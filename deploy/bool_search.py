from collections import defaultdict
import json

#建立词汇-文档表
class BoolSearch():
    def __init__(self,docs=None):
        if docs==None: return
        self.words={}
        #二进制位数
        self.bits = len(docs)
        self.keys = set()
        for indx in range(self.bits):
            for word in docs[indx]: self.keys.add(word)
                    
        for word in self.keys:
            for indx in range(self.bits):
                if word not in self.words: self.words[word]=0
                self.words[word] <<= 1
                if word in docs[indx]:
                    self.words[word] |= 1
#         print(self.words)

    def __itr_dict__(self,doc):
        # 1.先遍历词典,判断词典中的词是否在新文档中
        for word in self.words.keys():
        # 2.如果在文档中
            self.words[word] <<= 1
            if word in doc:
        # 3.就向左移位并加一
                self.words[word] |= 1
        # 4.否则只向左移位
        # 5.然后从文档中移除这个词
                while word in doc: doc.remove(word)
                
    def __itr_doc__(self,doc):
        # 6.接着遍历文档中剩下的词,重复步骤3
        for word in doc:
            self.words[word] = 0
            self.words[word] <<= 1
            self.words[word] |= 1
                    
    def __all__(self,n):
        num = 1
        for i in range(n-1):
            num <<=1
            num |= 1
        return num
    
    def search(self,input):
        result = self.__all__(self.bits)
#         print(self.bits,result)
        for word in input:
            if not self.words.__contains__(word): continue
            result &= self.words[word]
        return result
    
    def show(self):
        print(self.words)
        
    def get_indexes(self,input):
        result = self.search(input)
        indexes = []
        for indx in range(self.bits):
            if result & 1: indexes.append(self.bits-indx -1)
            result >>=1
        return indexes
    
    def save(self,filepath):
        with open(filepath, 'w') as fw:
            json.dump({'bits':self.bits,'words':self.words},fw)
        
    def load(self,filepath):
        with open(filepath,'r') as f:
            s = json.load(f)
            self.bits = s['bits']
            self.words = s['words']

bool_search = BoolSearch()
bool_search.load('search_answer.jsn')