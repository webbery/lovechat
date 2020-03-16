from .model.net import Transformer
from .data_utils.utils import Config, CheckpointManager, SummaryManager
from .data_utils.vocab_tokenizer import Tokenizer, Vocabulary, keras_pad_fn, mecab_token_pos_flat_fn
import sys
import numpy as np
import json
import torch
from pathlib import Path

class Seq2Seq():
    def __init__(self):
        data_dir = Path('deploy/chat/generator/experiments/data')
        model_dir = Path('deploy/chat/generator/experiments')
        data_config = Config(json_path=data_dir / 'config.json')
        self.model_config = Config(json_path=model_dir / 'config.json')

        with open(data_config.token2idx_vocab, mode='rb') as io:
            token2idx_vocab = json.load(io)
        self.vocab = Vocabulary(token2idx = token2idx_vocab)
        self.tokenizer = Tokenizer(vocab=self.vocab, split_fn=mecab_token_pos_flat_fn, pad_fn=keras_pad_fn, maxlen=self.model_config.maxlen)
        self.model_config.vocab_size = len(self.vocab.token2idx)

        self.model = Transformer(config=self.model_config, vocab=self.vocab)
        checkpoint_manager = CheckpointManager(model_dir) # experiments/base_model
        checkpoint = checkpoint_manager.load_checkpoint('best.tar')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    def decoding_from_result(self,y_pred):
        list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()
        pred_token = self.tokenizer.decode_token_ids(list_of_pred_ids)
        pred_str = ''.join([token.split('/')[0] for token in pred_token[0][:-1]])
        return pred_str

    def generate(self,input):
        input_text = list(input)
        enc_input = torch.tensor(self.tokenizer.list_of_string_to_arr_of_pad_token_ids([input_text]))
        # print(enc_input)
        dec_input = torch.tensor([[self.vocab.token2idx[self.vocab.START_TOKEN]]])
        for i in range(self.model_config.maxlen):
            y_pred = self.model(enc_input.to(self.device), dec_input.to(self.device))
            y_pred_ids = y_pred.max(dim=-1)[1]
            if (y_pred_ids[0,-1] == self.vocab.token2idx[self.vocab.END_TOKEN]).to(torch.device('cpu')).numpy():
                return self.decoding_from_result(y_pred)
            dec_input = torch.cat([dec_input.to(torch.device('cpu')), y_pred_ids[0,-1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)
            if i ==self. model_config.maxlen - 1:
                return self.decoding_from_result(y_pred)

seq2seq = Seq2Seq()