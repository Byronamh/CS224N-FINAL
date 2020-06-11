from tika import parser
from constants import *
import tensorflow as tf
import numpy as np

files = [f for f in os.listdir('./data/training') if os.path.isfile(f)]
files = filter(lambda f: f.endswith(('.pdf', '.PDF')), files)

word_bank = []

for filename in files:
    fileData = parser.from_file('./data/training/' + filename)
    safe_text = str(fileData['content']).encode('utf-8', errors='ignore')
    safe_text = str(safe_text).replace('\\', '\\\\').replace('"', '\\"')

    safe_text = safe_text.replace('\n', '')
    sentences = safe_text.split('.')
    words = []
    for sentence in sentences:
        words = sentence.split(' ')
    new_seq = []
    for i in range(MAX_LENGTH):
        try:
            new_seq.append(words[i])
        except:
            new_seq.append(PADDING)
    word_bank.append(new_seq)
words = word_bank

latest = tf.train.latest_checkpoint(CHECKPOINT_PATH)  # load saved ELMo

model = tf.compat.v1.keras.Model()

# Load the previously saved weights
model.load_weights(latest)

i = 19
prediction = model.predict(np.array(words[i:i + BATCH_SIZE]))[0]
prediction = np.argmax(prediction, axis=-1)
print(VERBOSE_TABLE_FORMAT_STRING.format(CSV_KEYS['WORD'], "Pred"))
print("-" * 30)
for w, pred in prediction:
    if w != PADDING:
        print(VERBOSE_TABLE_FORMAT_STRING.format(w, pred))
