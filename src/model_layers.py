import numpy as np
import torch
from torch import nn
import pickle
import torch.nn.functional as F
import math
import os

class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        # shape [N, candidate_size, 1]

        x = torch.bmm(x.permute(0, 2, 1), alpha) # (N, candidate_vector_dim, 1)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int, enable_gpu:bool):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, mask=None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size, seq_len, _ = Q.shape

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) #  [bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [bz, 20, seq_len, seq_len]

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  #self.layer_norm(output + residual)

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class EntFormer(nn.Module):
    """
    SEE: Signature Entity Encoder with Transformer
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transformer_layer = nn.TransformerEncoderLayer(config.ent_dim, 5, 400, 0.5, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, config.num_see_layers)
        self.pe = PositionalEncoding(config.ent_dim)
        # * dict size: # of entities + special token <pad> + # of categories
        self.embedding = nn.Embedding(config.emb_size, config.ent_dim, 0, max_norm=True)

    def forward(self, input_ids, mask=None):
        """Encoding a sequence of entities into an embedding

        Args:
            input_ids: input entity ids, shape [N, S]
            mask: input entity mask, shape [N, S]
        """
        embeddings = self.embedding(input_ids)
        pe_embedding = self.pe(torch.transpose(embeddings, 0, 1))
        # add the positional embeddings, shape [S, N, E]
        # ! `True` means mask out
        if mask is None:
            mask = (input_ids==0)
        encoded = self.transformer_encoder(pe_embedding, src_key_padding_mask=mask)
        # shape [S, N, E]
        return encoded
    
    def load_pretrained_embedding(self):
        if self.config.use_transe == False:
            return
        filepath = os.path.join(self.config.data_dir, self.config.dataset, self.config.test_dir, "entemb.pkl")        
        
        with open(filepath, 'rb') as f:
            pretrained_matrix = np.load(f, allow_pickle=True)

        assert pretrained_matrix.shape == self.embedding.weight.shape, "Pretrained embedding shape does not match model's embedding layer."
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_matrix, device=self.config.device, dtype=self.embedding.weight.dtype), freeze=True)

        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_matrix).type_as(self.embedding.weight))