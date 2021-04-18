# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F, Parameter
from params import Params
from dataset import Dataset

class DE_InteractE(torch.nn.Module):
    def __init__(self, dataset, params):

        super(DE_InteractE, self).__init__()
        self.dataset = dataset
        self.params = params

        # Creating static embeddings.
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        # Creating and initializing the temporal embeddings for the entities 
        self.create_time_embedds()
        
        # Setting the non-linearity to be used for temporal part of the embedding
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
        # interactE params
        params.perm = 2
        params.embedding_dim = params.s_emb_dim + params.t_emb_dim
        params.embedding_shape1 = 10
        params.input_drop = params.dropout
        params.hidden_drop = params.dropout
        params.feat_drop = params.dropout
        params.use_bias = True
        params.hidden_size = 12800
        params.kernel_size = 3
        params.num_filt = 32
        
        self.padding = 0
        
        self.inp_drop = torch.nn.Dropout(params.input_drop).cuda()
        self.hidden_drop = torch.nn.Dropout(params.hidden_drop).cuda()
        self.feature_map_drop = torch.nn.Dropout2d(params.feat_drop).cuda()
        self.loss = torch.nn.BCELoss().cuda()
        self.emb_dim1 = params.embedding_shape1
        self.emb_dim2 = params.embedding_dim // self.emb_dim1

        #print(params.s_emb_dim, params.t_emb_dim, params.embedding_dim, self.emb_dim1, self.emb_dim2, flush=True)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=params.use_bias).cuda()
        self.bn0 = torch.nn.BatchNorm2d(params.perm).cuda()
        self.bn1 = torch.nn.BatchNorm2d(params.num_filt * params.perm).cuda()
        self.bn2 = torch.nn.BatchNorm1d(params.embedding_dim).cuda()
        # self.register_parameter('b', Parameter(torch.zeros(dataset.numEnt())))
        self.fc = torch.nn.Linear(params.hidden_size,params.embedding_dim).cuda()
        
        self.chequer_perm = self.get_chequer_perm()
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.params.num_filt, 1, params.kernel_size,  params.kernel_size).cuda()))
        nn.init.xavier_normal_(self.conv_filt)
        
    def get_chequer_perm(self):
        ent_perm  = np.int32([np.random.permutation(self.params.embedding_dim) for _ in range(self.params.perm)])
        rel_perm  = np.int32([np.random.permutation(self.params.embedding_dim) for _ in range(self.params.perm)])

        comb_idx = []
        for k in range(self.params.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.emb_dim1):
                for j in range(self.emb_dim2):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx]+self.params.embedding_dim); rel_idx += 1;
                        else:
                            temp.append(rel_perm[k, rel_idx]+self.params.embedding_dim); rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx]+self.params.embedding_dim); rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                        else:
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx]+self.params.embedding_dim); rel_idx += 1;

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).cuda()
        return chequer_perm
        
    def create_time_embedds(self):
        
        # frequency embeddings for the entities
        self.m_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        # phi embeddings for the entities
        self.m_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        # amplitude embeddings for the entities
        self.m_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)
        
            
    def get_time_embedd(self, entities, year, month, day):

        y = self.y_amp(entities)*self.time_nl(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*self.time_nl(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day + self.d_phi(entities))

        return y+m+d

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)        
        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)
        
        h = torch.cat((h,h_t), 1)
        t = torch.cat((t,t_t), 1)
        return h,r,t
        
    def circular_padding_chw(self, batch, padding):
        upper_pad   = batch[..., -padding:, :]
        lower_pad   = batch[..., :padding, :]
        temp        = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad    = temp[..., -padding:]
        right_pad   = temp[..., :padding]
        padded      = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded
        
    def forward(self, heads, rels, tails, years, months, days):
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days)

        cat_inputs = torch.cat([h_embs, r_embs], dim=1)
        #print("cat:", cat_inputs.shape, flush=True)
        
        
        chequer_inputs = cat_inputs[:, self.chequer_perm]
        #print(self.chequer_perm.shape, flush=True)
        #print("chequer:", chequer_inputs.shape, flush=True)
        
        stacked_inputs = chequer_inputs.reshape((-1, self.params.perm, 2*self.emb_dim1, self.emb_dim2)) 

        #print("stacked:", stacked_inputs.shape, flush=True)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.circular_padding_chw(x, self.params.kernel_size // 2)
        
        #print("circular:", x.shape, flush=True)
        
        x= F.conv2d(x, self.conv_filt.repeat(self.params.perm, 1, 1, 1), padding=self.padding, groups=self.params.perm)
        
        #print("conv:", x.shape, flush=True)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)

        #print("before view:", x.shape, flush=True)
        x = x.view(x.shape[0], -1)
        #print("after view:", x.shape, flush=True)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        
        scores = x - t_embs

        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim = 1)
        
        return scores

