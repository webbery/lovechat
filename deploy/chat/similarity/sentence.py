import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba as jieba

class SentenceSimenticModel(nn.Module):
    def __init__(self,features):
        super(SentenceSimenticModel, self).__init__()
        self.linear_left_1 = nn.Linear(features, 256)
        self.relu_left = nn.ReLU()
        self.linear_left_2 = nn.Linear(256, 128)
        self.dropout_left = nn.Dropout(0.5)
        self.linear_left_3 = nn.Linear(128, 256)

        self.linear_right_1 = nn.Linear(features, 256)
        self.relu_right = nn.ReLU()
        self.linear_right_2 = nn.Linear(256, 128)
        self.dropout_right = nn.Dropout(0.5)
        self.linear_right_3 = nn.Linear(128, 256)
        self.softsign = nn.Softsign()
#         self.classifier = torch.Linear(128,2)
        
    def forward(self, input_ids1, input_ids2, token_type_ids=None, position_ids=None, head_mask=None,
            labels=None):
#         print(input_ids1)
#         print('-----------')
        output1 = torch.FloatTensor(list(input_ids1))
        output1 = self.linear_left_2(self.relu_left(self.linear_left_1(output1)))
        output1 = self.linear_left_3(self.dropout_left(output1))
        
        if input_ids2 is not None:
            output2 = torch.FloatTensor(list(input_ids2))
            output2 = self.linear_right_2(self.relu_right(self.linear_right_1(output2)))
            output2 = self.linear_right_3(self.dropout_right(output2))

            cls_output = self.softsign(F.pairwise_distance(output2,output1))
            cls_output = cls_output.reshape(labels.shape)
            criterion = nn.MSELoss()
            loss = 0
            if labels is not None:
                loss = criterion(cls_output, labels)
            return loss, cls_output
        else:
            return output1

class SentenceSimentic():
    def __init__(self,model_file):
        features = 512
        self.model = SentenceSimenticModel(features)
        checkpoint = torch.load(model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.sentence2vec = SentenceTransformer('distiluse-base-multilingual-cased')


    def decompose(self,sentence):
        ## 
        if isinstance(sentence,str):
            sentences = [sentence]
        else:
            sentences = sentence
        vec = self.sentence2vec.encode(sentences)
        vec = self.model(vec,None)
        return vec.cpu().detach().numpy().tolist()

sentence2vec = SentenceSimentic('custom.pkl')

def __array2sentece__(arr):
    sentences = []
    for sentence in arr:
        line = ''
        for word in sentence:
            line += word + ' '
        sentences.append(line)
    return sentences

def list2words(sentences):
    result = []
    for sentence in sentences:
        l = jieba.lcut(sentence)
        result.append(l)
    return result

def get_similarity_with_tfidf(input,source,tops=10):
    '''
    source & input like this:
    [
        '这样啊，那我不打扰您了，您有空再来找我聊吧!',
        '别着急啊，有事慢慢来'
    ]
    '''
    print(input)
    print(source)
    input = list2words(input)
    source = list2words(source)


    sentences = __array2sentece__(source)
    new_input = __array2sentece__(input)
    sentences.append(new_input[0])
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(sentences).todense()
    values = cosine_similarity(tfidf_matrix)
    print(values)
    last_indx = len(source)
    print(values[last_indx])

    result = sorted(list(enumerate(values[last_indx])),key=lambda x:x[1],reverse=True)
    print(result)
    return result[1:(1+tops)]
