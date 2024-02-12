import torch
import torch.nn as nn   

from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool, GATConv

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

from sklearn.utils import shuffle
import collections
import pandas as pd
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def batch_norm_init(m):
    if isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()
        
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Gene_Embedding(nn.Module):
    def __init__(self, vocab_size= None,embed_size=None):
        super(Gene_Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dim = embed_size
        self.eps = 1e-12
        
    def forward(self, genes=None,scales=None):
        x = self.embedding(genes)
        x = self.unit(x)
        x *= scales.unsqueeze(-1) 
        return x
        
    def unit(self,x):
        return (x+self.eps)/(torch.norm(x,dim=2).unsqueeze(-1)+self.eps)


class Tokenizer():
    def __init__(
        self, Gene_vocab, shuf= True, pad_token=0, sep_token=1, unk_token=2, cls_token=3, mask_token=4, **kwargs):
        super().__init__()
        self.Gene_vocab = Gene_vocab
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.shuf = shuf
        
        self.special_tokens = {'UNK':self.unk_token, 'SEP':self.sep_token, 
                               'PAD':self.pad_token, 'CLS':self.cls_token,
                               'MASK':self.mask_token}
        

        self.symb_to_id = collections.OrderedDict([(SYMBOL, ID) for SYMBOL,ID in self.Gene_vocab.values])
        
    @property
    def vocab_size(self):
        return len(self.Gene_vacab)

    def get_vocab(self):
        return self.Gene_vocab 
    
    def tokenize(self, sample):
        pathway = sample['pathway']
        sample_Id = sample['Id']
        genes_scales = pd.Series(data = sample['scales'],index = sample['genes'])
        genes = list(genes_scales.index.astype(str))
        scales = list(genes_scales.values)
        
        if self.shuf:
            genes,scales = shuffle(genes, scales)
                
        token = {"pathway":pathway,"sample_Id":sample_Id,
                 "genes":genes,"scales":scales}
        
        return token
    
    def check_unk(self,genes):
        genes = [gene if gene is not None else self.special_tokens['UNK'] for gene in genes]
        return genes
    
    def check_mis_scale(self,scales):
        scales = [scale if scale > 1e-12 else 1.0 for scale in scales]
        return scales
                
    def convert_symb_to_id(self, symbs):
        return [self.symb_to_id.get(symb) for symb in symbs]


    def convert_id_to_symb(self, indices):
        return [list(self.symb_to_id.keys())[list(self.symb_to_id.values()).index(index)] for index in indices]
    
    def encode(self, sample, add_special_tokens = True, 
               max_length = 128, pad_to_max_length = True,
               gene_type = 'SYMBOL'):
        
        token = self.tokenize(sample)

        token['genes'] = self.convert_symb_to_id(token['genes'])
        
        token['genes'] = self.check_unk(token['genes'])
        token['scales'] = self.check_mis_scale(token['scales'])
        
        if add_special_tokens:
            token['genes'] = [self.special_tokens['CLS']] + token['genes'] + [self.special_tokens['SEP']]
            token['scales'] = [1] + token['scales'] + [1] 
        
        if pad_to_max_length:
            token['genes'] += [self.special_tokens['PAD']]*(max_length+2 - len(token['genes']))
            token['scales'] += [self.special_tokens['PAD']]*(max_length+2 - len(token['scales']))
            
        return token
    
    def encode2torch(self, sample, add_special_tokens = True, 
                    max_length = 128, pad_to_max_length = True,
                    gene_type = 'SYMBOL'):
        
        token = self.encode(sample, add_special_tokens = add_special_tokens
                            ,gene_type = gene_type)
            
        token['genes'] = torch.tensor(token['genes'])
        token['scales'] = torch.tensor(token['scales'], dtype=torch.float)
            
        return token
    
    
    def encode_pair(self, sample1, sample2, add_special_tokens = True, 
                    max_len = 128, pad_to_max_length = True,
                    return_attention_mask = False,
                    gene_type = 'ENTREZID'):
        
        token1 = self.tokenize(sample1)
        token2 = self.tokenize(sample2)
        pair_token = {}
        

        token1['genes'] = self.convert_symb_to_id(token1['genes'])
        token2['genes'] = self.convert_symb_to_id(token2['genes'])
        
        token1['genes'] = self.check_unk(token1['genes'])
        token2['genes'] = self.check_unk(token2['genes'])
        
        token1['scales'] = self.check_mis_scale(token1['scales'])
        token2['scales'] = self.check_mis_scale(token2['scales'])
        
        if add_special_tokens:
            token1['genes'] = [self.special_tokens['CLS']] + token1['genes'] + [self.special_tokens['SEP']]
            token2['genes'] = token2['genes'] + [self.special_tokens['SEP']]
            
            token1['scales'] = [1] + token1['scales'] + [1]
            token2['scales'] = token2['scales'] + [1] 
            
        pair_token['genes'] = token1['genes'] + token2['genes'] 
        pair_token['scales'] = token1['scales'] + token2['scales']
            
        if pad_to_max_length:
            pair_token['genes'] += [self.special_tokens['PAD']]*(max_len-len(pair_token['genes']))
            pair_token['scales'] += [self.special_tokens['PAD']]*(max_len-len(pair_token['scales']))
            
        pair_token['genes'] = torch.tensor(pair_token['genes'])
        pair_token['scales'] = torch.tensor(pair_token['scales'], dtype=torch.float)
        pair_token['sample_Ids'] = (sample1['Id'],sample2['Id'])
        pair_token['pathways'] = (sample1['pathway'],sample2['pathway'])
        
        return pair_token
    

class GEN(torch.nn.Module):
    def __init__(self, y_dim = 256, dropout_ratio = 0.3,
                 gnn = None, embedding = None, encoder = None):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding 
        self.gnn = gnn 
        
        self.dropout_ratio = dropout_ratio
        self.y_dim = y_dim
    
        self.do = nn.Dropout(self.dropout_ratio)
        
        self.regression = nn.Sequential(
            nn.Linear(self.y_dim, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 1)
        )
        self.regression.apply(xavier_init)
        
        
    def forward(self,x_drug,x_gexpr, x_genes_gexpr,x_methyl, x_genes_methyl):
        x_d = self.gnn(x_drug)     
        x_g = self.embedding(x_genes_gexpr, x_gexpr)
        x_m = self.embedding(x_genes_methyl, x_methyl)
        
        x = self.encoder(x_g,x_m,x_d)
        
        y = self.regression(x)
        
        return y

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Transformer_Encoder(nn.Module):
    def __init__(self, genes=300, x_dim = 64, y_dim = 512, dropout = 0.15, encoder = None):
        super(Transformer_Encoder, self).__init__()
        
        self.key_fc = nn.Linear(x_dim,y_dim)
        self.query_fc = nn.Linear(x_dim,y_dim)
        self.value_fc = nn.Linear(x_dim,y_dim)
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        self.encoder = encoder

        self.key_fc.apply(xavier_init)
        self.query_fc.apply(xavier_init)
        self.value_fc.apply(xavier_init)
                    
    def forward(self, g: torch.Tensor, m:torch.Tensor) -> torch.Tensor:
        x = torch.cat([g,m],dim=1)
        key = self.key_fc(x) 
        query = self.query_fc(x) 
        value = self.value_fc(x)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        att = self.softmax(scores)
        
        y = torch.matmul(att, value)
        y = self.layer_norm(y)+value
        
        if self.encoder is not None:
            y = self.encoder(y) + y
        
        return y

class Co_Att_Encoder(nn.Module):
    def __init__(self, genes=300, x_dim = 64, y_dim = 512, dropout = 0.15, encoder = None):
        super(Co_Att_Encoder, self).__init__()
        
        self.key_ge_fc = nn.Linear(x_dim,y_dim)
        self.query_ge_fc = nn.Linear(x_dim,y_dim)
        self.value_ge_fc = nn.Linear(x_dim,y_dim)
        
        
        self.key_me_fc = nn.Linear(x_dim,y_dim)
        self.query_me_fc = nn.Linear(x_dim,y_dim)
        self.value_me_fc = nn.Linear(x_dim,y_dim)
        
        
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        self.encoder = encoder        

        self.key_ge_fc = self.key_ge_fc.apply(xavier_init)
        self.query_ge_fc = self.query_ge_fc.apply(xavier_init)
        self.value_ge_fc = self.value_ge_fc.apply(xavier_init)
        self.key_me_fc = self.key_me_fc.apply(xavier_init)
        self.query_me_fc = self.query_me_fc.apply(xavier_init)
        self.value_me_fc = self.value_me_fc.apply(xavier_init)


    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        query_ge = self.query_ge_fc(x) 
        key_ge = self.key_ge_fc(x)
        value_ge = self.value_ge_fc(x)
        
        query_me = self.query_me_fc(m)
        key_me = self.key_me_fc(m) 
        value_me = self.value_me_fc(m)
        
        scores_ge = torch.matmul(query_ge, key_me.transpose(-2, -1)) \
                 / math.sqrt(query_ge.size(-1))
                 
        scores_me = torch.matmul(query_me, key_ge.transpose(-2, -1)) \
                 / math.sqrt(query_me.size(-1))
        
                
        att_ge = self.softmax(scores_ge) 
        att_me = self.softmax(scores_me)
        
        y_ge = torch.matmul(att_ge, value_me)
        y_me = torch.matmul(att_me, value_ge)
        
        y_ge = self.layer_norm(y_ge)+query_ge
        y_me = self.layer_norm(y_me)+query_me
        
        
        y = torch.cat([y_ge,y_me],dim=1)
        y = self.layer_norm(y)
        y = self.dropout(y)
        
        return y 


class Merged_Att_Encoder(nn.Module):
    def __init__(self, genes=300, x_dim = 64, y_dim = 512, dropout = 0.15, encoder = None):
        super(Merged_Att_Encoder, self).__init__()
        
        self.key_ge_fc = nn.Linear(x_dim,y_dim)
        self.query_ge_fc = nn.Linear(x_dim,y_dim)
        self.value_ge_fc = nn.Linear(x_dim,y_dim)
        
        
        self.key_me_fc = nn.Linear(x_dim,y_dim)
        self.query_me_fc = nn.Linear(x_dim,y_dim)
        self.value_me_fc = nn.Linear(x_dim,y_dim)
        
        
        self.layer_norm2 = nn.LayerNorm(genes*2)
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        self.encoder = encoder           

        self.key_ge_fc = self.key_ge_fc.apply(xavier_init)
        self.query_ge_fc = self.query_ge_fc.apply(xavier_init)
        self.value_ge_fc = self.value_ge_fc.apply(xavier_init)
        self.key_me_fc = self.key_me_fc.apply(xavier_init)
        self.query_me_fc = self.query_me_fc.apply(xavier_init)
        self.value_me_fc = self.value_me_fc.apply(xavier_init)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        co_x = torch.cat([x,m],dim=1)
        
        query_ge = self.query_ge_fc(x) 
        key_ge = self.key_ge_fc(co_x) 
        value_ge = self.value_ge_fc(co_x)
        
        query_me = self.query_me_fc(m) 
        key_me = self.key_me_fc(co_x)
        value_me = self.value_me_fc(co_x)
        
        scores_ge = torch.matmul(query_ge, key_me.transpose(-2, -1)) \
                 / math.sqrt(query_ge.size(-1))
                 
        scores_me = torch.matmul(query_me, key_ge.transpose(-2, -1)) \
                 / math.sqrt(query_me.size(-1))
        
                
        att_ge = self.softmax(scores_ge) 
        att_me = self.softmax(scores_me)
        
        y_ge = torch.matmul(att_ge, value_me)
        y_me = torch.matmul(att_me, value_ge)
        
        y_ge = self.layer_norm(y_ge)+query_ge
        y_me = self.layer_norm(y_me)+query_me
        
        
        y = torch.cat([y_ge,y_me],dim=1)
        y = self.layer_norm(y)
        y = self.dropout(y)
        
        return y 
      
class hierarchical_Encoder(nn.Module):
    def __init__(self, genes=300, x_dim = 64, y_dim = 512, dropout = 0.15, encoder = None):
        super(hierarchical_Encoder, self).__init__()
        
        self.key_ge_fc = nn.Linear(x_dim,y_dim)
        self.query_ge_fc = nn.Linear(x_dim,y_dim)
        self.value_ge_fc = nn.Linear(x_dim,y_dim)
        
        
        self.key_me_fc = nn.Linear(x_dim,y_dim)
        self.query_me_fc = nn.Linear(x_dim,y_dim)
        self.value_me_fc = nn.Linear(x_dim,y_dim)

        self.key_hi_fc = nn.Linear(y_dim,y_dim)
        self.query_hi_fc = nn.Linear(y_dim,y_dim)
        self.value_hi_fc = nn.Linear(y_dim,y_dim)        
        
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        self.encoder = encoder        
        
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        query_ge = self.query_ge_fc(x) 
        key_ge = self.key_ge_fc(x) 
        value_ge = self.value_ge_fc(x)
        
        query_me = self.query_me_fc(m) 
        key_me = self.key_me_fc(m) 
        value_me = self.value_me_fc(m)
        
        scores_ge = torch.matmul(query_ge, key_me.transpose(-2, -1)) \
                 / math.sqrt(query_ge.size(-1))
                 
        scores_me = torch.matmul(query_me, key_ge.transpose(-2, -1)) \
                 / math.sqrt(query_me.size(-1))
        
                
        att_ge = self.softmax(scores_ge) 
        att_me = self.softmax(scores_me)
        
        y_ge = torch.matmul(att_ge, value_me)
        y_me = torch.matmul(att_me, value_ge)
        
        y_ge = self.layer_norm(y_ge)+query_ge
        y_me = self.layer_norm(y_me)+query_me
        
        
        y = torch.cat([y_ge,y_me],dim=1)

        query_hi = self.query_hi_fc(y) 
        key_hi = self.key_hi_fc(y) 
        value_hi = self.value_hi_fc(y)
        scores_hi = torch.matmul(query_hi, key_hi.transpose(-2, -1)) \
                 / math.sqrt(query_hi.size(-1))       
                
        att_hi = self.softmax(scores_hi) 
        y_hi = torch.matmul(att_hi, value_hi)
        y_hi = self.layer_norm(y_hi)+query_hi
        y = self.layer_norm(y_hi)
        y = self.dropout(y)
        
        return y 

class CTT_Att_Encoder(nn.Module):  #Concatenate & Self vectors
    def __init__(self, genes, x_dim, y_dim, dropout = 0.15, encoder = None):
        super(CTT_Att_Encoder, self).__init__()
        
        self.key_ge_fc = nn.Linear(x_dim,y_dim)
        self.query_ge_fc = nn.Linear(x_dim,y_dim)
        self.value_ge_fc = nn.Linear(x_dim,y_dim)
        
        
        self.key_me_fc = nn.Linear(x_dim,y_dim)
        self.query_me_fc = nn.Linear(x_dim,y_dim)
        self.value_me_fc = nn.Linear(x_dim,y_dim)

        self.key_co_fc = nn.Linear(x_dim,y_dim)
        self.query_co_fc = nn.Linear(x_dim,y_dim)
        self.value_co_fc = nn.Linear(x_dim,y_dim)        
        
        self.layer_norm2 = nn.LayerNorm(genes*2)
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        self.encoder = encoder           

        self.key_ge_fc = self.key_ge_fc.apply(xavier_init)
        self.query_ge_fc = self.query_ge_fc.apply(xavier_init)
        self.value_ge_fc = self.value_ge_fc.apply(xavier_init)
        self.key_me_fc = self.key_me_fc.apply(xavier_init)
        self.query_me_fc = self.query_me_fc.apply(xavier_init)
        self.value_me_fc = self.value_me_fc.apply(xavier_init)
        self.key_co_fc = self.key_co_fc.apply(xavier_init)
        self.query_co_fc = self.query_co_fc.apply(xavier_init)
        self.value_co_fc = self.value_co_fc.apply(xavier_init)

    def forward(self, g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        co_x = torch.cat([g,m],dim=1)
        
        query_ge = self.query_ge_fc(g) 
        key_ge = self.key_ge_fc(g)
        value_ge = self.value_ge_fc(g)
        
        query_me = self.query_me_fc(m) 
        key_me = self.key_me_fc(m) 
        value_me = self.value_me_fc(m)

        query_co = self.query_co_fc(co_x)
        key_co = self.key_co_fc(co_x) 
        value_co= self.value_co_fc(co_x)

        scores_ge = torch.matmul(query_ge, key_me.transpose(-2, -1)) \
                 / math.sqrt(query_ge.size(-1))
                 
        scores_me = torch.matmul(query_me, key_ge.transpose(-2, -1)) \
                 / math.sqrt(query_me.size(-1))
        scores_co = torch.matmul(query_co, key_co.transpose(-2, -1)) \
                 / math.sqrt(query_co.size(-1))        
                
        att_ge = self.softmax(scores_ge) 
        att_me = self.softmax(scores_me)
        att_co = self.softmax(scores_co)
        
        y_ge = torch.matmul(att_ge, value_me)
        y_me = torch.matmul(att_me, value_ge)
        y_co = torch.matmul(att_co, value_co)
        
        y_ge = self.layer_norm(y_ge)+query_ge
        y_me = self.layer_norm(y_me)+query_me
        y_co = self.layer_norm(y_co)+query_co
        
        
        y = torch.cat([y_ge,y_me,y_co],dim=1)
        y = self.layer_norm(y)
        y = self.dropout(y)
        
        return y 

class COT_Att_Encoder(nn.Module): #Co-attention & Self vectors
    def __init__(self, genes, x_dim, y_dim, dropout = 0.15, encoder = None):
        super(COT_Att_Encoder, self).__init__()
        
        self.key_ge_fc = nn.Linear(x_dim,y_dim)
        self.query_ge_fc = nn.Linear(x_dim,y_dim)
        self.value_ge_fc = nn.Linear(x_dim,y_dim)
        
        
        self.key_me_fc = nn.Linear(x_dim,y_dim)
        self.query_me_fc = nn.Linear(x_dim,y_dim)
        self.value_me_fc = nn.Linear(x_dim,y_dim)

        self.key_co_fc = nn.Linear(x_dim,y_dim)
        self.query_co_fc = nn.Linear(x_dim,y_dim)
        self.value_co_fc = nn.Linear(x_dim,y_dim)        
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        self.encoder = encoder           

        self.key_ge_fc = self.key_ge_fc.apply(xavier_init)
        self.query_ge_fc = self.query_ge_fc.apply(xavier_init)
        self.value_ge_fc = self.value_ge_fc.apply(xavier_init)
        self.key_me_fc = self.key_me_fc.apply(xavier_init)
        self.query_me_fc = self.query_me_fc.apply(xavier_init)
        self.value_me_fc = self.value_me_fc.apply(xavier_init)
        self.key_co_fc = self.key_co_fc.apply(xavier_init)
        self.query_co_fc = self.query_co_fc.apply(xavier_init)
        self.value_co_fc = self.value_co_fc.apply(xavier_init)

    def forward(self, g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        query_ge = self.query_ge_fc(g)
        key_ge = self.key_ge_fc(g)
        value_ge = self.value_ge_fc(g)
        
        query_me = self.query_me_fc(m) 
        key_me = self.key_me_fc(m) 
        value_me = self.value_me_fc(m)


        scores_ge = torch.matmul(query_ge, key_me.transpose(-2, -1)) \
                 / math.sqrt(query_ge.size(-1))                 
        scores_me = torch.matmul(query_me, key_ge.transpose(-2, -1)) \
                 / math.sqrt(query_me.size(-1))
        
        scores_coge = torch.matmul(query_ge, key_me.transpose(-2, -1)) \
                 / math.sqrt(query_ge.size(-1))                 
        scores_come = torch.matmul(query_me, key_ge.transpose(-2, -1)) \
                 / math.sqrt(query_me.size(-1))

                
        att_ge = self.softmax(scores_ge) 
        att_me = self.softmax(scores_me)

        att_coge = self.softmax(scores_coge)
        att_come = self.softmax(scores_come)
        
        y_ge = torch.matmul(att_ge, value_me)
        y_me = torch.matmul(att_me, value_ge)
        y_coge = torch.matmul(att_coge, value_me)
        y_come = torch.matmul(att_come, value_ge)
        
        y_ge = self.layer_norm(y_ge)+query_ge
        y_me = self.layer_norm(y_me)+query_me
        y_coge = self.layer_norm(y_coge)+query_ge
        y_come = self.layer_norm(y_come)+query_me
        
        
        y = torch.cat([y_ge,y_me,y_coge,y_come],dim=1)
        y = self.layer_norm(y)
        y = self.dropout(y)
        
        return y 

class Main_Encoder(nn.Module):
    def __init__(self, cell_encoder = None, d_dim = 128, 
                 genes=300, y_dim=512, dropout = 0.15):
        
        super(Main_Encoder, self).__init__()
        self.cell_encoder = cell_encoder
        
        self.feed1 = nn.Linear(d_dim,y_dim)
        self.feed2 = nn.Linear(y_dim,y_dim)
        self.activation = GELU() 
        
        self.layer_norm1 = nn.LayerNorm(y_dim*2)
        self.dropout = nn.Dropout(dropout)

        self.feed1.apply(xavier_init)
        self.feed2.apply(xavier_init)
        
    def forward(self, x_g: torch.Tensor,x_m:torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        x1 = self.cell_encoder(x_g,x_m)
        y1, _ = torch.max(x1, dim= 1)  # [batchsize, embeddings_dim]
        
        y2 = self.feed2(self.activation(self.feed1(x_d)))
        
        y = torch.cat([y1,y2],dim = 1)
        y = self.dropout(self.layer_norm1(y))
        
        return y
    
class GEN_WO_GeneVec(torch.nn.Module):
    def __init__(self,gcn = None, gexpr_dim=100, dropout_drug = 0.1, dropout_cell = 0.1,
                 dropout_reg = 0.1, d_dim = 128*3, y_dim = 512):#
        super().__init__()
        self.d_dim = d_dim
        self.y_dim = y_dim
        self.gcn = gcn
        
        self.fc_d1 = nn.Linear(d_dim,y_dim)
        self.fc_d2 = nn.Linear(y_dim,y_dim)
        self.activation = GELU() 
        
        self.fc_g1 = nn.Linear(gexpr_dim,y_dim)
        self.fc_g2 = nn.Linear(y_dim,y_dim)
        
        self.layer_norm1 = nn.LayerNorm(y_dim*2)
        self.dropout_drug = nn.Dropout(dropout_drug)
        self.dropout_cell = nn.Dropout(dropout_cell)
        
        self.regression = nn.Sequential(
                nn.Linear(self.y_dim*2, 512),
                nn.ELU(),
                nn.Dropout(p=dropout_reg),
                nn.Linear(512, 512),
                nn.ELU(),
                nn.Dropout(p=dropout_reg),
                nn.Linear(512, 1)
            )
        
    def forward(self,x_feat=None,x_adj=None,x_gexpr=None):
        x = self.gcn(x_feat)
        
        x = self.fc_d1(x)
        x = self.activation(x)
        x = self.dropout_drug(x)
        x = self.fc_d2(x)
        
        x_gexpr = self.fc_g1(x_gexpr)
        x_gexpr = self.activation(x_gexpr) 
        x_gexpr = self.dropout_cell(x_gexpr) 
        x_gexpr = self.fc_g2(x_gexpr)
        
        x = torch.cat([x,x_gexpr], dim=1) #Concatenate()([x,x_gexpr])
        x = self.layer_norm1(x)
        
        y = self.regression(x)
        
        return y


class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug, do):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(do)
        
        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), 
                                      nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)
            
        
        self.convs_drug.apply(xavier_init)
        self.bns_drug.apply(batch_norm_init)

    def forward(self, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.dropout(x_drug)
        return x_drug


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        if self.bias is not None:
            support += self.bias
        
        #output = torch.matmul(adj.permute(0,2,1), support)
        output = torch.matmul(adj, support)
        
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
        self.NEG_INF = -1e38

    def max_pooling(self,x, adj):
        node_num = x.shape[1]
        features = torch.unsqueeze(x,1).repeat(1, node_num, 1, 1) \
                    + torch.unsqueeze((1.0 - adj) * self.NEG_INF, -1)
        return torch.max(features,2)[0]
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = (torch.sum(x,dim=1)/x.shape[1])
        
        return x
