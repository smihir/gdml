
import mxnet as mx
import numpy as np
import sys


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label
                , bucket_key, utt_id=None, utt_len=0, 
                effective_sample_count=None):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.utt_id = utt_id
        self.utt_len = utt_len
        self.effective_sample_count = effective_sample_count

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]