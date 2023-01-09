# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import pdb


def mixup_data(x, y, alpha=0.1, runs, use_cuda=True):
    for i in range(runs):
        output_x = torch.Tensor(0)
        output_x= output_x.numpy().tolist()
        output_y = torch.Tensor(0)
        output_y = output_y.numpy().tolist()
        batch_size = x.size()[0]
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)


def mixup_data_refactor( x, y, x_refactor, y_refactor, alpha, runs, use_cuda=True):
    for i in range(runs):
        output_x = torch.Tensor(0)
        output_x= output_x.numpy().tolist()
        output_y = torch.Tensor(0)
        output_y = output_y.numpy().tolist()
        batch_size = x.size()[0]
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x_refactor[index, :]
        mixed_y = lam * y + (1 - lam) * y_refactor[index, :]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
        
    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None): 

        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)
    
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())[0]

        logits=self.classifier(outputs)
        logits, labels = mixup_data(logits,labels) # Mixup Data 
        prob=torch.nn.functional.log_softmax(logits,-1)

        if labels is not None:
            loss = -torch.sum(prob*labels)
            return loss,prob
        else:
            return prob
      

        
 
        


