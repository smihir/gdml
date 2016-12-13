import find_mxnet
import mxnet as mx
import logging
import os
import pickle
import socket
import sys

class gdSGD(mx.optimizer.Optimizer):
    """A very simple SGD optimizer with momentum and weight regularization.
    It sends the weights to a server located at a higher level in the server
    tree, and pulls back updated weights from the server.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, momentum=0.0, guf=1, **kwargs):
        super(gdSGD, self).__init__(**kwargs)
        self.global_update_frequency = guf
        self.current_update_count = 0
        self.momentum = momentum
        self.hierarchical = True
        self.sock = None

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return mx.ndarray.zeros(weight.shape, weight.context, dtype=weight.dtype)

    def update(self, index, weight, grad, state):
        """Update the parameters and send up the tree

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.ndarray.NDArray))
        assert(isinstance(grad, mx.ndarray.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        self.current_update_count += 1

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom[:] *= self.momentum
            mom[:] += -lr * (grad + wd * weight)
            weight[:] += mom
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + wd * weight)

        # now push the weight to root
        if self.sock is not None:
            if self.current_update_count % self.global_update_frequency == 0:
                try:
                    logging.info('send')
                    self.sock.send(pickle.dumps((index, weight)))
                    logging.info('send ok')
                    # zmq documentation says this recv is necessary.
                    # I have not tested without this recv.
                    # TODO(smihir): try to see if the recv can be removed.
                    try:
                        new = pickle.loads(self.sock.recv())
                        weight[:] = new
                    except Exception as ee:
                        logging.info('cannot load new weights {}'.format(str(ee)))

                except Exception as e:
                    logging.info('cannot send model upstream: {}'.format(str(e)))
                    pass


def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    kv.set_optimizer(mx.optimizer.Test())
    mx.optimizer.Optimizer.register(gdSGD)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
        # TODO: check epoch_size for 'dist_sync'
        epoch_size = args.num_examples / args.batch_size
        model_args['begin_num_update'] = epoch_size * args.load_epoch

    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= (kv.num_workers * args.num_datacenters)
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer          = 'gdSGD',
        **model_args)

    eval_metrics = ['accuracy', 'ce']

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 1))

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)
