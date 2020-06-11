import pandas as pd
import numpy as np

from classes.ModelEmbedder import ModelEmbedder
from constants import *

from classes.SentenceGetter import SentenceGetter
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow_hub as hub
import tensorflow as tf

tfv1 = tf.compat.v1
tfv1.disable_eager_execution()

data = pd.read_csv("./data/training/ner.csv", encoding="latin1", error_bad_lines=False)
data = data.fillna(method="ffill")

vocab = list(set(data[CSV_KEYS['WORD']].values))
vocab.append("ENDPAD")
tags = list(set(data[CSV_KEYS['TAG']].values))

getter = SentenceGetter(data)
sent = getter.get_next()
corpus = getter.sentences
tagsToBeMatched = {t: i for i, t in enumerate(tags)}
word_bank = []
for seq in [[w[0] for w in s] for s in corpus]:
    new_seq = []
    for i in range(MAX_LENGTH):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append(PADDING)
    word_bank.append(new_seq)
words = word_bank

tags = [[tagsToBeMatched[w[2]] for w in s] for s in corpus]
tags = pad_sequences(maxlen=MAX_LENGTH, sequences=tags, padding="post", value=tagsToBeMatched["O"])

words_tr, words_te, tags_tr, tags_te = train_test_split(words, tags, test_size=0.1, random_state=RANDOM_SEED)

sess = tfv1.Session()
tfv1.keras.backend.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess.run(tfv1.global_variables_initializer())
sess.run(tfv1.tables_initializer())

embedder = ModelEmbedder(elmo_model, 'elmo')

input_text = tfv1.keras.Input(shape=(MAX_LENGTH,), dtype=tf.string)
embedding = tfv1.keras.layers.Lambda(
    embedder.embed,
    output_shape=(None, 1024)
)(input_text)

hidden_lstm_layer = tfv1.keras.layers.Bidirectional(
    tfv1.keras.layers.LSTM(
        units=512,
        return_sequences=True,
        recurrent_dropout=0.2,
        dropout=0.2
    )
)(embedding)

hidden_rnn_layer = tfv1.keras.layers.Bidirectional(
    tfv1.keras.layers.LSTM(
        units=512,
        return_sequences=True,
        recurrent_dropout=0.2,
        dropout=0.2
    )
)(hidden_lstm_layer)

hidden_lstm_layer = tfv1.keras.layers.add(
    [hidden_lstm_layer, hidden_rnn_layer]
)

out = tfv1.keras.layers.TimeDistributed(
    tfv1.keras.layers.Dense(len(tags), activation="softmax")
)(hidden_lstm_layer)

model = tfv1.keras.Model(input_text, out)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

words_tr, words_val = words_tr[:1213 * BATCH_SIZE], words_tr[-135 * BATCH_SIZE:]
tags_tr, tags_val = tags_tr[:1213 * BATCH_SIZE], tags_tr[-135 * BATCH_SIZE:]

tags_tr = tags_tr.reshape(tags_tr.shape[0], tags_tr.shape[1], 1)
tags_val = tags_val.reshape(tags_val.shape[0], tags_val.shape[1], 1)
os.system('mkdir -p {}'.format(CHECKPOINT_PATH))

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    verbose=1
)
history = model.fit(
    np.array(words_tr),
    tags_tr,
    validation_data=(np.array(words_val), tags_val),
    batch_size=BATCH_SIZE,
    epochs=5,
    verbose=1,
    callbacks=[cp_callback]
)

if VERBOSE:

    i = 19
    prediction = model.predict(np.array(words_te[i:i + BATCH_SIZE]))[0]
    prediction = np.argmax(prediction, axis=-1)
    print(VERBOSE_TABLE_FORMAT_STRING.format(CSV_KEYS['WORD'], "Pred", "True"))
    print("-" * 30)
    for w, true, pred in zip(words_te[i], tags_te[i], prediction):
        if w != PADDING:
            print(VERBOSE_TABLE_FORMAT_STRING.format(w, tags[pred], tags[true]))
