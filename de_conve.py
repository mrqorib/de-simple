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

class DE_ConvE(torch.nn.Module):
    def __init__(self, dataset, params):

        super(DE_ConvE, self).__init__()
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
        
        # convE params
        params.embedding_dim = params.s_emb_dim + params.t_emb_dim
        params.embedding_shape1 = 10
        params.input_drop = params.dropout
        params.hidden_drop = params.dropout
        params.feat_drop = params.dropout
        params.use_bias = True
        params.hidden_size = 4608
        
        self.inp_drop = torch.nn.Dropout(params.input_drop).cuda()
        self.hidden_drop = torch.nn.Dropout(params.hidden_drop).cuda()
        self.feature_map_drop = torch.nn.Dropout2d(params.feat_drop).cuda()
        self.loss = torch.nn.BCELoss().cuda()
        self.emb_dim1 = params.embedding_shape1
        self.emb_dim2 = params.embedding_dim // self.emb_dim1

        #print(params.s_emb_dim, params.t_emb_dim, params.embedding_dim, self.emb_dim1, self.emb_dim2, flush=True)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=params.use_bias).cuda()
        self.bn0 = torch.nn.BatchNorm2d(1).cuda()
        self.bn1 = torch.nn.BatchNorm2d(32).cuda()
        self.bn2 = torch.nn.BatchNorm1d(params.embedding_dim).cuda()
        # self.register_parameter('b', Parameter(torch.zeros(dataset.numEnt())))
        self.fc = torch.nn.Linear(params.hidden_size,params.embedding_dim).cuda()
        
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
        
    def forward(self, heads, rels, tails, years, months, days):
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days)

        #print("h_emb_shape:", h_embs.shape, flush=True)
        #print("t_emb_shape:", t_embs.shape, flush=True)

        
        h_embedded = h_embs.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_embedded = r_embs.view(-1, 1, self.emb_dim1, self.emb_dim2)

        #print("h_embedded:", h_embedded.shape, flush=True)
        #print("r_embedded:", r_embedded.shape, flush=True)

        
        stacked_inputs = torch.cat([h_embedded, r_embedded], 2)

        #print("stacked:", stacked_inputs.shape, flush=True)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
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

