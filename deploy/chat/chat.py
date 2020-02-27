from deploy.chat.query.query import corpus
from deploy.chat.clawer import claw_answer

class ChatService():
    def __init__(self):
        pass

    def reply(self,input):
        sentence,score = corpus.find_sentence_by_similarity(input)
        # if score<0.1:
        #     sentence = claw_answer(input)
        return sentence, score

chat_service = ChatService()