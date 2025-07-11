import torch
from torch import nn
from itertools import accumulate
from src.sub_module import *
from src.model_layers import *
from src.model import *
from loguru import logger
from typing import List, Tuple
    

class UserBehavEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(UserBehavEncoder, self).__init__()
        self.config = config
        self.multihead_self_attention = MultiHeadAttention(
            config.news_dim, config.num_attention_heads, config.user_query_vector_dim,
            config.user_query_vector_dim, self.config.enable_gpu)

        self.additive_attention = AdditiveAttention(config.user_query_vector_dim*20,
                                                    config.news_dim)
        
        self.ent_multihead_self_attention = MultiHeadAttention(
            config.news_dim, config.num_attention_heads, config.user_query_vector_dim,
            config.user_query_vector_dim, self.config.enable_gpu)
        self.ent_additive_attention = AdditiveAttention(config.user_query_vector_dim*20,
                                                    config.news_dim)
        self.layer_norm = nn.LayerNorm(self.config.news_dim)


    def forward(self, user_vector, mask, entity_vector) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch_size, num_clicked_news_a_user, news_embedding_dim
        multihead_user_vector = self.multihead_self_attention(Q=user_vector, K=entity_vector, mask=mask)
        multihead_user_vector = self.layer_norm(multihead_user_vector + user_vector)
        # batch_size, news_embedding_dim
        final_user_vector = self.additive_attention(multihead_user_vector, mask)

        multihead_ent_vector = self.ent_multihead_self_attention(Q=entity_vector, K=user_vector, mask=mask)
        multihead_ent_vector = self.layer_norm(multihead_ent_vector + entity_vector)
        final_ent_vector = self.ent_additive_attention(multihead_ent_vector, mask)
        return final_user_vector, final_ent_vector
    


class IP2(nn.Module):
    def __init__(self, config):
        super(IP2, self).__init__()
        self.config = config
        self.user_encoder = UserBehavEncoder(config)
        self.news_encoder = Ent_Pretrainer(config)
        self.click_predictor = DotProductClickPredictor()
        self.hist_cast = nn.Linear(config.news_dim*2, 1)
        self.cand_cast = nn.Linear(config.news_dim*2, 1)
        
        
    def _chunk_news_encode(self, tokens, masks, group_ids, ents, cat):
        max_hist_len = max(group_ids)
        chunk_news_tokens = tokens.split(self.config.chunk_size)
        chunk_news_tokens_mask = masks.split(self.config.chunk_size)
        chunk_ent_embeddings = ents.split(self.config.chunk_size)
        chunk_category = cat.split(self.config.chunk_size)
        # chunk into pieces, then start to encode
        reshaped_embedding = torch.zeros((len(group_ids), max_hist_len, self.config.news_dim), device=tokens.device) # [N, max_history_news, news_dim]
        reshaped_masking = torch.zeros((len(group_ids), max_hist_len), device=tokens.device) # [N, max_history_news]
        reshaped_ents = torch.zeros((len(group_ids), max_hist_len, self.config.news_dim), device=tokens.device)

        chunk_embeddings = []
        chunk_ents = []
        for tk, tm, et, ct in zip(chunk_news_tokens, chunk_news_tokens_mask, chunk_ent_embeddings, chunk_category):
            resout = self.news_encoder((tk, tm, et, ct), mode="downstream")
            chunk_embeddings.append(resout[0])
            chunk_ents.append(resout[1])
        
        chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
        chunk_ents = torch.cat(chunk_ents, dim=0)
        ids_offset = [0] + list(accumulate(group_ids))
        for i, (j, k) in enumerate(zip(ids_offset, group_ids)):
            # i-index inside the minibatch; j-history news log length
            # front padding, e.g. 0,0,0,0,1,1,1,1
            # news embedding and entity will share the same mask
            reshaped_embedding[i][-k:] = chunk_embeddings[j:j+k]
            reshaped_ents[i][-k:] = chunk_ents[j:j+k]
            reshaped_masking[i][-k:] = 1
        return reshaped_embedding, reshaped_masking, reshaped_ents

    def _chunk_news_encode_reduced(self, tokens, masks, group_ids, ents, cat):
        chunk_news_tokens = tokens.split(self.config.chunk_size)
        chunk_news_tokens_mask = masks.split(self.config.chunk_size)
        chunk_ent_embeddings = ents.split(self.config.chunk_size)
        chunk_category = cat.split(self.config.chunk_size)
        # chunk into pieces, then start to encode
        # ! To avoid bottle-neck, set `chunk_size` at least 128.
        chunk_embeddings = []
        chunk_ents = []
        # * Do not need to allocate the memory prior manually.
        for tk, tm, et, ct in zip(chunk_news_tokens, chunk_news_tokens_mask, chunk_ent_embeddings, chunk_category):
            resout = self.news_encoder((tk, tm, et, ct), mode="downstream")
            chunk_embeddings.append(resout[0])
            chunk_ents.append(resout[1])
        
        reshaped_embedding = torch.cat(chunk_embeddings, dim=0)
        reshaped_ents = torch.cat(chunk_ents, dim=0)
        return reshaped_embedding, reshaped_ents

        
    def forward(self, data):
        hist_vec, hist_mask, ent_hist_vec = self._chunk_news_encode(data[0], data[2], data[4], data[6], data[10])
        user_vec, user_vec2 = self.user_encoder(hist_vec, hist_mask, ent_hist_vec) # (N, news_dim)
        impression_vec, im_ent_vec = self.news_encoder((data[1], data[3], data[7], data[11]), mode='downstream')

        cur_batchsize = len(data[4])
        # * adjust reshape strategy for last batch of data
        if cur_batchsize == self.config.batch_size:
            # normal occation
            impression_vec = impression_vec.reshape(self.config.batch_size, self.config.npratio+1, -1)
            im_ent_vec = im_ent_vec.reshape(self.config.batch_size, self.config.npratio+1, -1)
        else:
            # last batch in the dataset
            impression_vec = impression_vec.reshape(cur_batchsize, self.config.npratio+1, -1)
            im_ent_vec = im_ent_vec.reshape(cur_batchsize, self.config.npratio+1, -1)

        # this will return a (batch_size, candidate_size) prediction matrix.
        pred = self._predict(impression_vec, im_ent_vec, user_vec, user_vec2)
        return pred
        
    def _load_pretrain_encoder(self, statedict):
        logger.success("Pretrained news encoder loaded!")
        self.news_encoder.load_state_dict(statedict)
    
    def get_news_vector(self, news, group_ids):
        hist_vec, hist_mask, ent_hist_vec = self._chunk_news_encode(news[0], news[1], group_ids, news[2], news[3])
        return hist_vec, hist_mask, ent_hist_vec
    
    def get_news_vector_reduced(self, news, group_ids):
        hist_vec, ent_hist_vec = self._chunk_news_encode_reduced(news[0], news[1], group_ids, news[2], news[3])
        return hist_vec, ent_hist_vec
    
    def get_ent_vector(self, ents, group_ids):
        ent_hist_vec, ent_hist_mask = self._ent_transform(ents, group_ids)
        return ent_hist_vec, ent_hist_mask

    def get_user_vector(self, hist_vec, hist_mask, entity_vector):
        # N, news_embedding
        return self.user_encoder(hist_vec, hist_mask, entity_vector)

    def get_prediction(self, impression_vec, im_ent_vec, user_vec, user_vec2):
        pred = self._predict(impression_vec, im_ent_vec, user_vec, user_vec2)

        return pred
    
    def get_prediction_reduced(self, im_vec, im_ent_vec, group_ids, user_vec, user_vec2) -> List[torch.Tensor]:
        # aggregation
        user_alpha = torch.sigmoid(self.hist_cast(torch.concat((user_vec, user_vec2), dim=-1))) # shape [N, 1]
        combine_user = user_alpha * user_vec + (1-user_alpha) * user_vec2

        imp_alpha = torch.sigmoid(self.cand_cast(torch.concat((im_vec, im_ent_vec), dim=-1))) # shape [N, cadidate_size]
        combine_cadnews = im_vec * imp_alpha + im_ent_vec * (1-imp_alpha) # [x, news_dim]

        gather_index = torch.repeat_interleave(torch.arange(group_ids.shape[0], device=group_ids.device), group_ids)

        raw_pred = torch.matmul(combine_cadnews, combine_user.T)
        raw_pred = torch.gather(raw_pred, dim=1, index=gather_index.unsqueeze(-1)).squeeze()
        pred = raw_pred.split(group_ids.tolist())

        return pred


    def _predict(self, impression_vec, im_ent_vec, user_vec, user_vec2):
        if self.config.predict_type == 'avg':
            pred1 = self.click_predictor(impression_vec, user_vec)
            pred2 = self.click_predictor(im_ent_vec, user_vec2)
            pred = torch.mean(torch.stack((pred1, pred2)), dim=0)

        elif self.config.predict_type == 'cross':
            user_alpha = torch.sigmoid(self.hist_cast(torch.concat((user_vec, user_vec2), dim=-1))) # shape [N, 1]
            combine_user = user_alpha * user_vec + (1-user_alpha) * user_vec2

            imp_alpha = torch.sigmoid(self.cand_cast(torch.concat((impression_vec, im_ent_vec), dim=-1))) # shape [N, cadidate_size]
            combine_cadnews = impression_vec * imp_alpha + im_ent_vec * (1-imp_alpha)
            
            pred = self.click_predictor(combine_cadnews, combine_user)
        elif self.config.predict_type == 'max':
            pred1 = self.click_predictor(impression_vec, user_vec)
            pred2 = self.click_predictor(im_ent_vec, user_vec2)
            pred, _ = torch.max(torch.stack((pred1, pred2)), dim=0)

        elif self.config.predict_type == 'cast':
            raise NotImplementedError
        elif self.config.predict_type == 'text':
            pred = self.click_predictor(impression_vec, user_vec)
        elif self.config.predict_type == 'entity':
            pred = self.click_predictor(im_ent_vec, user_vec2)
        return pred


    def _freeze_partial_bert(self):
        self.news_encoder._freeze_partial_bert()

    def _unfreeze_partial_bert(self):
        self.news_encoder._unfreeze_partial_bert()
