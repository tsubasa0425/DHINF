import math
import torch.nn.init as init
from Layers import *


class RelationHGNN(nn.Module):

    def __init__(self, input_num, embed_dim, dropout=0.5, is_norm=False):
        super().__init__()
        self.embedding = nn.Embedding(input_num, embed_dim)
        self.hgnn = HypergraphConv(embed_dim, embed_dim)
        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(embed_dim)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, hypergraph):
        output_embeddings = self.hgnn(self.embedding.weight, hypergraph)
        output_embeddings = self.dropout(output_embeddings)
        if self.is_norm:
            output_embeddings = self.batch_norm(output_embeddings)
        return output_embeddings

class CascadeHGNN(nn.Module):
    def __init__(self, input_num, embed_dim, dropout=0.5, is_norm=False):
        super().__init__()
        self.embedding = nn.Embedding(input_num, embed_dim)
        self.hgnn = CascadeConv(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(embed_dim)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, hypergraph):
        user_embedding = self.hgnn(self.embedding.weight, hypergraph)
        user_embedding = self.dropout(user_embedding)
        if self.is_norm:
            user_embedding = self.batch_norm(user_embedding)
        return user_embedding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CoAttention(nn.Module):
    def __init__(self, user_size, emb_dim):
        super().__init__()
        self.pe_encoder = PositionalEncoding(d_model=emb_dim, dropout=0.1)
        # 影响力embedding，每个用户不同
        self.influence_embedding = nn.Embedding(num_embeddings=user_size, embedding_dim=emb_dim)
        sqrt3 = math.sqrt(3)
        self.co_attn_wts = nn.Parameter((sqrt3+sqrt3)*torch.rand(emb_dim, emb_dim)-sqrt3)
    def forward(self, examples, sender_social_embedding, sender_temporal_embedding, masks):
        sender_influence_embedding = self.influence_embedding(examples)   # (batch_size, seq_len, embed_size)
        sender_influence_embedding = torch.multiply(sender_influence_embedding, torch.unsqueeze(masks.float(), -1))
        sender_temporal_embedding = self.pe_encoder(sender_temporal_embedding)
        sender_temporal_embedding += sender_influence_embedding
        attn_act = torch.tanh(
            torch.sum(
                torch.multiply(
                    torch.tensordot(sender_social_embedding, self.co_attn_wts, dims=([2], [0])),
                    sender_temporal_embedding
                ), 2
            )
        )
        attn_alpha = nn.Softmax(dim=1)(attn_act)
        attended_embeddings = torch.multiply(sender_temporal_embedding, torch.unsqueeze(attn_alpha, -1))
        outputs = torch.sum(attended_embeddings, 1) #(batch_size, embed_size)
        return outputs

class Fusion1(nn.Module):
    def __init__(self, input_size, out = 1, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, user_embeddings):
        hg_weight = []
        hg_num = len(user_embeddings)
        for i in range(hg_num):
            w = self.linear2(torch.tanh(self.linear1(user_embeddings[i])))  # (user_size, 1)
            hg_weight.append(w)
        temp = hg_weight[0]
        for i in range(1, hg_num):
            temp = torch.cat((temp, hg_weight[i]), 1)
        temp = nn.functional.softmax(temp, dim=1)  #(usersize, hg_num)
        output_embeding = torch.mul(user_embeddings[0], temp.T[:][0].view(-1, 1))   # (user_size, emb_dim)
        for i in range(1, hg_num):
            output_embeding += torch.mul(user_embeddings[0], temp.T[:][0].view(-1, 1))
        return output_embeding

class DHINF(nn.Module):
    def __init__(self, user_size, emded_dim):
        super().__init__()
        self.user_size = user_size
        self.emd_dim = emded_dim
        self.relationHGNN = RelationHGNN(self.user_size, self.emd_dim)
        self.cascadeHGNN = CascadeHGNN(self.user_size, self.emd_dim)
        self.co_att = CoAttention(self.user_size, self.emd_dim)
        self.fus1 = Fusion1(emded_dim)
        self.user_social_embedding = nn.Embedding(self.user_size, self.emd_dim)
        self.user_temporal_embedding = nn.Embedding(self.user_size, self.emd_dim)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.user_social_embedding.weight)
        init.xavier_normal_(self.user_temporal_embedding.weight)

    def forward(self, social_hypergraph_list, cascade_hypergraph, examples, masks, lambda_u):
        user_embeddings_list = []
        for i in range(len(social_hypergraph_list)):
            user_embeddings = self.relationHGNN(social_hypergraph_list[i])
            user_embeddings_list.append(user_embeddings)
        user_social_embedding = self.fus1(user_embeddings_list)
        sender_social_embedding = self.user_social_embedding(examples)
        sender_social_embedding = torch.multiply(sender_social_embedding, torch.unsqueeze(masks.float(), -1))
        user_temporal_embedding = self.cascadeHGNN(cascade_hypergraph)
        sender_temporal_embedding = self.user_temporal_embedding(examples)
        sender_temporal_embedding = torch.multiply(sender_temporal_embedding, torch.unsqueeze(masks.float(), -1))
        h = self.co_att(examples, sender_social_embedding, sender_temporal_embedding, masks)  # (batch_size, emd_dim)
        pred = torch.matmul(h, self.user_social_embedding.weight.T)  # (batch_size, user_size)
        user_loss = 0.5 * lambda_u * torch.mean(
            torch.sum(
                torch.square(user_temporal_embedding - self.user_temporal_embedding.weight), 1
            ) + torch.sum(
                torch.square(user_social_embedding - self.user_social_embedding.weight), 1
            )
        )
        return pred, self.co_att.co_attn_wts, user_loss