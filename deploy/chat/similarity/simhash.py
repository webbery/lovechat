import re
from simhash import Simhash, SimhashIndex

class SimHashSpace():
    def __init__(self,hash_list):
        objs = [(str(k), Simhash(v)) for k, v in enumerate(hash_list)]
        self.index = SimhashIndex(objs)

    def _get_features_(self,s):
        width = 3
        s = s.lower()
        s = re.sub(r'[^\w]+', '', s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

    def find_similarity_indexes(self,sentence):
        indexes = self.index.get_near_dups(Simhash(self._get_features_(sentence)))
        return [int(idx) for idx in indexes]


if __name__ == "__main__":
    passq_hash = []
    for q in tqdm(new_corpus['question']):
        qv = Simhash(get_features(q))
        q_hash.append(qv)
    new_corpus['question_hash']=q_hash