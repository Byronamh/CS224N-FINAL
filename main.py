import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from constants import CSV_KEYS as C


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s[C['WORD']].values.tolist(),
                                                           s[C['POS']].values.tolist(),
                                                           s[C['TAG']].values.tolist())]
        self.grouped = self.data.groupby(C['ID']).apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


data = pd.read_csv("./data/training/ner.csv", encoding="latin1", error_bad_lines=False)
data = data.fillna(method="ffill")

words = list(set(data[C['WORD']].values))
words.append("ENDPAD")
n_words = len(words)
tags = list(set(data[C['TAG']].values))
n_tags = len(tags)
getter = SentenceGetter(data)
sent = getter.get_next()
sentences = getter.sentences
max_len = 50
tag2idx = {t: i for i, t in enumerate(tags)}
X = [[w[0] for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
X = new_X

y = [[tag2idx[w[2]] for w in s] for s in sentences]

from keras.preprocessing.sequence import pad_sequences

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=2018)
batch_size = 32

elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


def ElmoEmbedding(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size * [max_len])
    },
        signature="tokens",
        as_dict=True)["elmo"]


input_text = Input(shape=(max_len,), dtype=tf.string)
print(input_text)
embedding = elmo_model(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

'''
input = Input(shape=(140,))
model = Embedding(input_dim=n_words, output_dim=140, input_length=140)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
'''

model = Model(input_text, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
X_tr, X_val = X_tr[:1213 * batch_size], X_tr[-135 * batch_size:]
y_tr, y_val = y_tr[:1213 * batch_size], y_tr[-135 * batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                    batch_size=batch_size, epochs=3, verbose=1)
