import hanlp


class NamedEntityRecognize():
    def __init__(self):
        self.recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)

    def recongize(self,sentences):
        return self.recognizer(sentences)

recognizer = NamedEntityRecognize()