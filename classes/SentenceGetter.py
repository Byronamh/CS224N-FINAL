from constants import CSV_KEYS as C


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        self.grouped = self.data.groupby(C['ID']).apply(self.compress)
        self.sentences = [s for s in self.grouped]

    def compress(self, row):
        return [
            (word, position, tag) for word, position, tag in
            zip(
                row[C['WORD']].values.tolist(),
                row[C['POS']].values.tolist(),
                row[C['TAG']].values.tolist()
            )
        ]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
