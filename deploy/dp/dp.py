import numpy as np
np.set_printoptions(suppress=True)
import hanlp

class SyntacticParser():
    def __init__(self):
        print('begin syntatic')
        self.tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        print('begin syntactic_parser')
        self.syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
        print('begin tagger')
        self.tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
        print('finish syntatic')

    def parse(self,sentences):
        #if isinstance(sentences,list)==False: return None
        print(sentences)
        token = self.tokenizer(sentences)
        print(token)
        tags = self.tagger(token)
        print(tags)
        pairs = []
        #if isinstance(sentences[0],list)==True:
        #    pass
        #else:
        for idx in range(len(tags)):
            pairs.append((token[idx],tags[idx]))
        print(pairs)
        return self.syntactic_parser(pairs)

parser = SyntacticParser()