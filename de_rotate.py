# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class DE_RotatE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_RotatE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.margin = params.margin
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        
        # code says -pi to pi, paper says 0 to 2* pi.
        nn.init.uniform_(self.ent_embs_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.ent_embs_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.rel_embs.weight, -math.pi, math.pi)
    
    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.uniform_(self.m_freq_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.d_freq_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.y_freq_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.m_freq_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.d_freq_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.y_freq_t.weight, -self.margin, self.margin)

        nn.init.uniform_(self.m_phi_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.d_phi_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.y_phi_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.m_phi_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.d_phi_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.y_phi_t.weight, -self.margin, self.margin)

        nn.init.uniform_(self.m_amps_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.d_amps_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.y_amps_h.weight, -self.margin, self.margin)
        nn.init.uniform_(self.m_amps_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.d_amps_t.weight, -self.margin, self.margin)
        nn.init.uniform_(self.y_amps_t.weight, -self.margin, self.margin)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        if h_or_t == "head":
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
        else:
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))
            
        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        h_re = self.ent_embs_h(heads)
        t_re = self.ent_embs_h(tails)
        h_im = self.ent_embs_t(tails)
        t_im = self.ent_embs_t(heads)

        phase_rel = self.rel_embs(rels)
        r_re = torch.cos(phase_rel)
        r_im = torch.sin(phase_rel)
        
        h_re = torch.cat((h_re, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_re = torch.cat((t_re, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_im = torch.cat((h_im, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_im = torch.cat((t_im, self.get_time_embedd(heads, years, months, days, "tail")), 1)
        
        return h_re, r_re, t_re, h_im, r_im, t_im
    
    def forward(self, heads, rels, tails, years, months, days):
        h_re, r_re, t_re, h_im, r_im, t_im = self.getEmbeddings(heads, rels, tails, years, months, days)

        # re_score = r_re * t_re + r_im * t_im
        # im_score = r_re * t_im - r_im * t_re
        
        # re_score = re_score - h_re
        # im_score = im_score - h_im
        
        re_score = h_re * r_re - h_im * r_im
        im_score = h_re * r_im + h_im * r_re
        re_score = re_score - t_re
        im_score = im_score - t_im

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        
        scores = self.margin - torch.sum(score, dim=1)
        
        return scores
        
