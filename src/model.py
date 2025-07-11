import torch
from torch import nn
import torch.nn.functional as F
from src.sub_module import *
from src.model_layers import *
from loguru import logger


class Ent_Pretrainer(nn.Module):
    """Entity enhanced news encoder, V2. 
    """
    def __init__(self, config) -> None:
        super(Ent_Pretrainer, self).__init__()
        self.config = config
        self.bert = Bert_Layer(config)
        self.bert_sent = Pooler(pooler_type=config.pooler_type, config=config)
        
        self.drop = nn.Dropout(config.drop_rate)
        self.drop1 = nn.Dropout(p=0.1)
        self.lin = nn.Linear(self.config.word_embedding_dim, self.config.news_dim)
        self.ent_lin = nn.Linear(self.config.ent_dim, self.config.news_dim)
        self.ent_former = EntFormer(config)
        self.ent_former.load_pretrained_embedding()
        self.gelu = nn.LeakyReLU()

    def forward(self, data, mode='pretrain'):
        if mode == 'pretrain':
            bert_output = self.bert(data[0], data[1])
            bert_sents = self.bert_sent(data[1], bert_output)
            bert_moe_out = self.gelu(self.lin(bert_sents)) # (N, news_dim)
            general_ent = self.ent_former(data[2])[0] # (N, news_dim)
            ent_moe_out = self.gelu(self.ent_lin(general_ent))
            
            return (bert_moe_out, ent_moe_out, data[3], self.drop1(bert_moe_out), self.drop1(ent_moe_out))
        else:
            # downstream stage
            bert_output = self.bert(data[0], data[1])
            bert_sents = self.bert_sent(data[1], bert_output)
            bert_moe_out = self.drop(self.lin(bert_sents))
            bert_moe_out = self.gelu(bert_moe_out)

            # general_ent = self.general_embedding(data[3])
            # mix_ent = torch.stack((general_ent, data[2]))
            # mix_avg = torch.mean(mix_ent, dim=0)
            mix_avg = self.ent_former(data[2])[0]
            mix_avg = self.gelu(self.ent_lin(mix_avg))
            return bert_moe_out, mix_avg
            
    def get_embedding(self):
        raise NotImplementedError
    
    def _freeze_partial_bert(self):
        self.bert.freeze_partial()
    
    def _unfreeze_partial_bert(self):
        self.bert.unfreeze_partial()
