import torch
from torch import nn
from transformers import AutoConfig, AutoModel
import torch.nn.functional as F
from src.model_layers import AdditiveAttention, MultiHeadAttention
from loguru import logger

class Bert_Layer(torch.nn.Module):
    def __init__(self, args):
        super(Bert_Layer, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_path)
        self.config.output_hidden_states = True
        self.config.num_hidden_layers = args.num_hidden_layers
        self.bert_layer = AutoModel.from_pretrained(args.bert_path, config=self.config)
        self.device = args.device
        self.bert_layer = self.bert_layer.to(self.device)

    def forward(self, ids, masks):
        bert_output = self.bert_layer(input_ids=ids, attention_mask=masks, output_hidden_states=True)

        return bert_output
    
    def freeze_partial(self):
        for i in range(self.args.num_freeze_layers):
            for p in self.bert_layer.encoder.layer[i].parameters():
                p.requires_grad = False

        for p in self.bert_layer.embeddings.parameters():
            p.requires_grad = False
    
    def unfreeze_partial(self):
        for name ,param in self.bert_layer.named_parameters():
            param.requires_grad = True
    

class DotProductClickPredictor(nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
            X reps news embedding dimension
        Returns:
            (shape): batch_size, candidate_size
        """
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability
    
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type, config):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
            "attention",
            "gate"
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )
        self.config = config
        if self.pooler_type == 'attention':
            self.norm = nn.LayerNorm(self.config.word_embedding_dim)
            self.mh_att = MultiHeadAttention(self.config.word_embedding_dim, 6, 128, 128, self.config.enable_gpu)
            self.att_layer = AdditiveAttention(self.config.word_embedding_dim, self.config.news_dim)

        if self.pooler_type == 'gate':
            self.norm = nn.LayerNorm(self.config.word_embedding_dim)
            self.mh_att = MultiHeadAttention(self.config.word_embedding_dim, 6, 128, 128, self.config.enable_gpu)
            self.fc1 = nn.Linear(768, 300)
            self.fc2 = nn.Linear(300, 1, False)
        logger.debug(f"News encoder type: {pooler_type}")

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            if self.pooler_type == 'cls_before_pooler':
                return last_hidden[:, 0]
            else:
                return pooler_output
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "attention":
            token_vectors = hidden_states[-1]
            att_result = self.mh_att(token_vectors, mask=attention_mask)
            att_result = self.norm(token_vectors + att_result)
            pooled_result = self.att_layer(token_vectors, attention_mask)

            return pooled_result
        elif self.pooler_type == "gate":
            token_vectors = hidden_states[-1]
            pooled_result = self.mh_att(token_vectors, mask=attention_mask)

            h_tilde = self.norm(token_vectors + pooled_result)
            r = torch.softmax(self.fc2(torch.tanh(self.fc1(h_tilde))), dim=1)
            h = torch.bmm(r.transpose(-1, -2), h_tilde).squeeze(dim=1)
            return h
