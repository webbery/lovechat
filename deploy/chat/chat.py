# from deploy.chat.query.query import corpus
from deploy.chat.generator.inference import seq2seq
from deploy.chat.clawer import claw_answer

import logging

class ChatService():
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    def reply(self,input):
        # sentence,score = corpus.find_sentence_by_similarity(input)
        # if score<0.1:
        #     sentence = claw_answer(input)
        return seq2seq.generate(input), 0.5

chat_service = ChatService()