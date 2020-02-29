import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AdamW, BertPreTrainedModel,get_linear_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
import re
import logging

class BertSimilarityModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSimilarityModel, self).__init__(config)
        self.bert = BertModel(config)
        self.similarity = nn.Linear(config.hidden_size, 16)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            labels=None):
        # print(input_ids.shape)
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1] 
        cls_output = self.similarity(cls_output) # batch, 16
#         print(cls_output.shape)
        cls_output = torch.sigmoid(cls_output)
#         print(cls_output.shape,labels.shape)
        loss = 0
        if labels is not None:
            criterion = nn.BCELoss()
            loss = criterion(cls_output, labels)
        return loss, cls_output

class BertSimilarity():
    def __init__(self,model_file):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        device = torch.device("cpu")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_file)
        self.model = BertSimilarityModel.from_pretrained('bert-base-chinese').to(device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def __truncate_token__(self,row,minsize=50):
        tokens = []
        for s in row:
            if isinstance(s,str)!=True:
                break
            sentence = re.sub(r'[。，?]','',s)
            if len(sentence)>minsize: sentence = sentence[0:minsize]
            text = self.tokenizer.encode(sentence)
            tokens += text
        if len(tokens)>512:
            minsize-=10
            tokens = self.__truncate_token__(row,minsize)
        return tokens

    def find_most_similarity(self,input,candidates):
        sentences = [input]
        sentences += candidates
        tokens = self.__truncate_token__(sentences,60)
        text = torch.LongTensor(tokens).reshape((1,-1))
        with torch.no_grad():
            mask = (text != 0).float()
            _, pred = self.model(text, attention_mask=mask)
            index = np.argmax(pred,axis=1)
            
            logging.debug(index)
            logging.debug(pred)
            logging.debug(candidates)

            return candidates[index],float(pred[0][index])

