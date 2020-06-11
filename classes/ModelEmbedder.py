import tensorflow as tf
from constants import *


class ModelEmbedder(object):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def embed(self, data):
        return self.model(
            inputs={
                "tokens": tf.squeeze(tf.cast(data, tf.string)),
                "sequence_len": tf.constant(BATCH_SIZE * [MAX_LENGTH])
            },
            signature="tokens",
            as_dict=True
        )[self.model_name]
