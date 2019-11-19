# 分割中文句子
import jieba
import re

def is_chinese(uchar):
    # 判断一个unicode是否是汉字
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
    
def is_number(uchar):
    # 判断一个unicode是否是数字
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    # 判断一个unicode是否是英文字母
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def is_legal(uchar):
    # 判断是否非汉字，数字和英文字符
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return False
    else:
        return True

def cut_chinese_sentences(doc,doc_type,):
    sentences = []
    try:
        if doc_type=="file":
            for line in doc.splitlines():
                sentences+=re.split("[。?？！!]+",line)
        else:
            ## array
            counter = 0
            for line in doc:
                # 捕获标点符号
                sentence=re.split("([，。?？！!\n\r]+)",line)
                frag_of_sentence = list(filter(lambda s: s and s.strip(), sentence))
                sentences += frag_of_sentence
                # print(sentences)
                counter += 1
                if counter>2: break
    except:
        print('except: ',doc)
    # sentences = [ for line in doc.splitlines()]
    # print(sentences)
    return sentences

def cut_and_segment(doc,lang='zh',doc_type='file',segment_type='str'):
    sentences = cut_chinese_sentences(doc,doc_type)
    # 构建带标点的句子
    sentences_with_punctuation=[]
    # 将句子带标点拼回来
    frag=[]
    sent = ''
    for idx in range(len(sentences)):
        if re.search('[，。?？！!\n\r]',sentences[idx])!=None:
            frag += [sent+sentences[idx]]
            sent=''
        else:
            sent += sentences[idx]
    sentences_with_punctuation+=frag
    sentences_of_split=[]
    for sentence in sentences:
        line = segment(sentence,segment_type)
        if line=="" or len(line)==0: continue
        sentences_of_split.append(line)
    
    # sentences_with_punctuation += frag
    return sentences_of_split,sentences_with_punctuation

def segment(sentence,type="str"):
    sentence_seged = jieba.cut(sentence.strip()) 
    stopwords = [",","，","。","?","？","（","）","(",")","《","》","、","/","！","「","」","“","”","：","<",">","－","／"," ","|","-","@","\r","\n","\r\n"]
    if type=="str":
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr
    else:
        outstr = []
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr.append(word)
        return outstr

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer()
# line = cut_and_segment("雨后")
# vec = tfidf.fit_transform(line)
# print(line)
