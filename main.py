import numpy as np
from models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy.stats import pearsonr

import torch.nn as nn
import torch

from torch import optim
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_condition else "cpu")

gene_vocab = pd.read_csv('./data/ccle_gene_vocabulary_promoter.csv',sep=',')
vocab_size = gene_vocab.shape[0]
tokenizer = Tokenizer(gene_vocab,shuf =True)

threshold = None
personalized_genes = False
random_genes = False


nb_epoch=250

gnn_dropout = 0.1
att_dropout = 0.1
fc_dropout = 0.1

nGenes = 300
lr = 0.0001
embed_size = 128
batch_size = 128

layer_drug = 3
dim_drug = 128
nhid = layer_drug*dim_drug

att_dim = 1000

#dataset = 'CTRP'
dataset = 'CCLE'
#dataset = 'GDSC'

name += '_'+dataset

title = name+'_Adim_'+str(att_dim)+'_Ddim_'+str(dim_drug)+'_nGenes_'+str(nGenes)+'_GNN_do'+str(gnn_dropout)+'_att_do_'+str(att_dropout)+'_lr_'+str(lr)

if dataset == 'GDSC':
    Gene_expression_file = './data/GDSC/GDSC_gexpression.csv'
    Methylation_file = './data/GDSC/GDSC_methylation_987cell.csv'
    Drug_info_file = './data/GDSC/GDSC_drug_information.csv'
    Drug_feature_file = './data/GDSC/drug_graph_feat'
    cancer_response_exp_file = './data/GDSC/GDSC2_ic50.csv'

    drugid2pubchemid, drug_pubchem_id_set, gexpr_feature,methyl_feature, _, experiment_data \
                                                    = get_drug_cell_info(Drug_info_file,Drug_feature_file,
                                                                         Gene_expression_file,Methylation_file,cancer_response_exp_file,
                                                                         norm = False)

    data_idx = get_idx(drugid2pubchemid, drug_pubchem_id_set, gexpr_feature,methyl_feature,experiment_data)

    drug_dict = np.load('./data/GDSC/GDSC_drug_feature_graph.npy', allow_pickle=True).item()
    overlapped_genes = set(gene_vocab['SYMBOL']).intersection(gexpr_feature.columns)
    gexpr_feature = gexpr_feature.dropna(how='all')[list(overlapped_genes)]
    overlapped_genes = set(gene_vocab['SYMBOL']).intersection(methyl_feature.columns)
    methyl_feature = methyl_feature.dropna(how='all')[list(overlapped_genes)]

    over_under_ids_df_gexpr,over_under_genes_df_gexpr = get_gene_set(tokenizer, gexpr_feature, nGenes, random_genes)
    over_under_ids_df_methyl,over_under_genes_df_methyl = get_gene_set(tokenizer, methyl_feature, nGenes, random_genes)    
    input_df_gexpr = get_gnn_input_df(data_idx,drug_dict,gexpr_feature,over_under_ids_df_gexpr,over_under_genes_df_gexpr)
    input_df_methyl = get_gnn_input_df(data_idx,drug_dict,methyl_feature,over_under_ids_df_methyl,over_under_genes_df_methyl)
    input_df_gexpr = input_df_gexpr[input_df_gexpr['drug id'] != '84691']
    input_df_methyl = input_df_methyl[input_df_methyl['drug id'] != '84691']

else: 
    if dataset == 'CCLE':
        nb_epoch = 200
        gexpr_feature = pd.read_csv("./data/CCLE/ccle_gexpr_promoter.csv",index_col = 0)
        methyl_feature = pd.read_csv("./data/CCLE/ccle_methylation.csv",index_col=0)
        gexpr_feature = gexpr_feature.T
        methyl_feature=methyl_feature.T
        assert methyl_feature.shape[0]==gexpr_feature.shape[0]
        methyl_feature = methyl_feature.loc[list(gexpr_feature.index)]

        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(gexpr_feature.values)
        gexpr_feature = pd.DataFrame(data=scalerGDSC.transform(gexpr_feature.values),
                                     index = gexpr_feature.index,
                                     columns = gexpr_feature.columns)

        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(methyl_feature.values)
        methyl_feature = pd.DataFrame(data=scalerGDSC.transform(methyl_feature.values),
                                     index = methyl_feature.index,
                                     columns = methyl_feature.columns)

        data_idx = pd.read_csv("./data/CCLE/CCLE_cell_drug.csv",index_col=0)
        data_idx = data_idx.dropna(axis=0)

        drug_dict = np.load("./data/CCLE/drug_feature_graph.npy", allow_pickle=True).item()

        drug_name = list(drug_dict.keys())
        data_idx = data_idx[data_idx['drug_name'].isin(drug_name)==True]
        data_idx = data_idx.values
        
        nb_celllines = len(set([item[0] for item in data_idx]))
        nb_drugs = len(set([item[1] for item in data_idx]))
        print('%d instances across %d cell lines and %d drugs were generated.'
            %(len(data_idx),nb_celllines,nb_drugs))

        overlapped_genes = set(gene_vocab['SYMBOL']).intersection(gexpr_feature.columns)
        gexpr_feature = gexpr_feature.dropna(how='all')[list(overlapped_genes)]
        overlapped_genes = set(gene_vocab['SYMBOL']).intersection(methyl_feature.columns)
        methyl_feature = methyl_feature.dropna(how='all')[list(overlapped_genes)]

        over_under_ids_df_gexpr,over_under_genes_df_gexpr = get_gene_set(tokenizer, gexpr_feature,
                                                            nGenes, random_genes)
        over_under_ids_df_methyl,over_under_genes_df_methyl = get_gene_set(tokenizer, methyl_feature,
                                                            nGenes, random_genes)    

        input_df_gexpr = get_gnn_input_df(data_idx,drug_dict,gexpr_feature,over_under_ids_df_gexpr,over_under_genes_df_gexpr)
        input_df_methyl = get_gnn_input_df(data_idx,drug_dict,methyl_feature,over_under_ids_df_methyl,over_under_genes_df_methyl)

        all_samples =  gexpr_feature.index

save_path = './results/weights/'
img_path = './results/imgs/'
result_path = './results/'

total_train_pcc = []
total_val_pcc = []
total_test_pcc = []
total_train_r2 = []
total_val_r2 = []
total_test_r2 = []
total_train_losses = []
total_test_losses = []
total_val_losses = []

kfold = KFold(n_splits=5,shuffle=True,random_state=0)
att_dim = 1000


for fold, (train2_index,test2_index) in enumerate(kfold.split(input_df_gexpr)):
    fold +=1
    patience = 10

    train2_index, val2_index = train_test_split(train2_index, test_size=0.05)
    train_df_gexpr = input_df_gexpr.iloc[train2_index].reset_index(drop=True)
    val_df_gexpr = input_df_gexpr.iloc[val2_index].reset_index(drop=True)
    test_df_gexpr = input_df_gexpr.iloc[test2_index].reset_index(drop=True)
    train_df_methyl = input_df_methyl.iloc[train2_index].reset_index(drop=True)
    val_df_methyl = input_df_methyl.iloc[val2_index].reset_index(drop=True)
    test_df_methyl = input_df_methyl.iloc[test2_index].reset_index(drop=True)


    train_dataloader = get_gnn_dataloader(train_df_gexpr,train_df_methyl, batch_size=batch_size)
    validation_dataloader = get_gnn_dataloader(val_df_gexpr,val_df_methyl, batch_size=batch_size)
    test_dataloader = get_gnn_dataloader(test_df_gexpr, test_df_methyl,batch_size=batch_size)

    gene_embedding = Gene_Embedding(vocab_size= vocab_size,embed_size=embed_size)

    gnn = GNN_drug(layer_drug = layer_drug, dim_drug = dim_drug, do = gnn_dropout)

    cell_encoder = Transformer_Encoder(genes = nGenes, x_dim= embed_size, y_dim = att_dim, 
                                        dropout = att_dropout, encoder = C_EnC)

    drug_encoder = Transformer_Encoder(genes = nGenes, x_dim= nhid, y_dim = att_dim,
                                        dropout = att_dropout, encoder = D_EnC)

    encoder = Main_Encoder(cell_encoder = cell_encoder, d_dim = nhid,
                            genes=nGenes, y_dim=att_dim, dropout = att_dropout)

    model = GEN(y_dim = att_dim*2, dropout_ratio = fc_dropout,
                gnn = gnn, embedding = gene_embedding, encoder = encoder)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) #0.0001
    mse = nn.MSELoss()

    train_pcc = []
    val_pcc = []
    test_pcc = []

    train_r2 = []
    val_r2 = []
    test_r2 = []

    best_pcc = 0
    train_loss = []
    test_loss = []
    val_loss = []
    for ep in range(nb_epoch):
        true_Y = []
        pred_Y = []


        model.train()
        for step, (x_drug, x_genes_gexpr, x_gexpr, x_genes_methyl, x_methyl, y) in enumerate(train_dataloader):
            if len(y) >1:
                optimizer.zero_grad()

                x_drug = x_drug.to(device)
                x_gexpr = x_gexpr.to(device)
                x_genes_gexpr = x_genes_gexpr.to(device)
                x_methyl = x_methyl.to(device)
                x_genes_methyl = x_genes_methyl.to(device)
                y = y.to(device).float()

                pred_y = model(x_drug,x_gexpr, x_genes_gexpr,x_methyl,x_genes_methyl)

                loss = mse(pred_y.view(-1),y)

                loss.backward()
                optimizer.step()

                pred_y = pred_y.view(-1).detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                true_Y += list(y)
                pred_Y += list(pred_y)

            # if (step+1) %500 ==0:
            #     print(title)
            #     print("training step: ", step)
            #     print("step_training loss: ", loss.item())
            #     overall_pcc = pearsonr(pred_y,y)[0]
            #     print("The overall Pearson's correlation is %.4f."%overall_pcc)

        loss_train = mean_squared_error(true_Y, pred_Y)
        pcc_train = pearsonr(true_Y, pred_Y)[0]
        r2_train = r2_score(true_Y, pred_Y)
        # print("Train avg_loss: ", loss_train)
        # print("Train avg_pcc: ", pcc_train)
        # print("Train r2: ", r2_train)

        train_pcc.append(pcc_train)
        train_loss.append(loss_train)
        train_r2.append(r2_train)

        total_val_loss = 0.
        sum_pcc = 0.
        true_Y = []
        pred_Y = []

        model.eval()
        for step, (x_drug, x_genes_gexpr, x_gexpr, x_genes_methyl, x_methyl, y) in enumerate(validation_dataloader):
            if len(y) >1:
                x_drug = x_drug.to(device)
                x_gexpr = x_gexpr.to(device)
                x_genes_gexpr = x_genes_gexpr.to(device)
                x_methyl = x_methyl.to(device)
                x_genes_methyl = x_genes_methyl.to(device)
                y = y.to(device).float()

                pred_y = model(x_drug,x_gexpr, x_genes_gexpr,x_methyl,x_genes_methyl)

                loss = mse(pred_y.view(-1),y)

                pred_y = pred_y.view(-1).detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                true_Y += list(y)
                pred_Y += list(pred_y)

                total_val_loss += loss.item()


        loss_val = mean_squared_error(true_Y, pred_Y)
        pcc_val = pearsonr(true_Y, pred_Y)[0]
        r2_val = r2_score(true_Y, pred_Y)

        # print("Validation avg_loss: ", loss_val)
        # print("Validation avg_pcc: ", pcc_val)
        # print("Validation r2: ", r2_val)
        val_loss.append(loss_val)
        val_pcc.append(pcc_val)
        val_r2.append(r2_val)

        if best_pcc < val_r2[-1]:
            best_pcc = val_r2[-1]
            torch.save(model.state_dict(),save_path+str(fold)+title+'.pt')
            # print('Best Val r2 ', best_pcc)

        true_Y = []
        pred_Y = []

        model.eval()

        for step, (x_drug, x_genes_gexpr, x_gexpr, x_genes_methyl, x_methyl, y) in enumerate(test_dataloader):
            if len(y) >1:
                x_drug = x_drug.to(device)
                x_gexpr = x_gexpr.to(device)
                x_genes_gexpr = x_genes_gexpr.to(device)
                x_methyl = x_methyl.to(device)
                x_genes_methyl = x_genes_methyl.to(device)
                y = y.to(device).float()

                pred_y = model(x_drug,x_gexpr, x_genes_gexpr,x_methyl,x_genes_methyl)
                loss = mse(pred_y.view(-1),y)

                pred_y = pred_y.view(-1).detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                true_Y += list(y)
                pred_Y += list(pred_y)

        loss_test = mean_squared_error(true_Y, pred_Y)
        pcc_test = pearsonr(true_Y, pred_Y)[0]
        r2_test = r2_score(true_Y, pred_Y)

        # print("Test avg_loss: ", loss_test)
        # print("Test avg_pcc: ", pcc_test)
        # print("Test r2: ", r2_test)

        test_pcc.append(pcc_test)
        test_loss.append(loss_test)
        test_r2.append(r2_test)

        # if (ep+1) %50 ==0:
        #     input_title = str(fold)+'_Loss_'+title
        #     show_picture(train_loss,val_loss, test_loss, input_title)
        #     input_title = str(fold)+'_PCC_'+title
        #     show_picture(train_pcc,val_pcc, test_pcc, input_title)
        #     input_title = str(fold)+'_r2_'+title
        #     show_picture(train_r2,val_r2, test_r2, input_title)


        # print("#################### epoch ############################ ",ep)

    model.load_state_dict(torch.load(save_path+str(fold)+title+'.pt'))
    torch.save(model.state_dict(), save_path+title+'_final.pt')
    true_Y = []
    pred_Y = []

    model.eval()

    for step, (x_drug, x_genes_gexpr, x_gexpr, x_genes_methyl, x_methyl, y) in enumerate(test_dataloader):
        if len(y) >1:
            x_drug = x_drug.to(device)
            x_gexpr = x_gexpr.to(device)
            x_genes_gexpr = x_genes_gexpr.to(device)
            x_methyl = x_methyl.to(device)
            x_genes_methyl = x_genes_methyl.to(device)
            y = y.to(device).float()

            pred_y = model(x_drug,x_gexpr, x_genes_gexpr,x_methyl,x_genes_methyl)
            loss = mse(pred_y.view(-1),y)

            pred_y = pred_y.view(-1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            true_Y += list(y)
            pred_Y += list(pred_y)


    loss_test = mean_squared_error(true_Y, pred_Y)
    pcc_test = pearsonr(true_Y, pred_Y)[0]
    r2_test = r2_score(true_Y, pred_Y)

    print("Test avg_loss: ", loss_test)
    print("Test avg_pcc: ", pcc_test)
    print("Test r2: ", r2_test)

    test_pcc.append(pcc_test)
    test_loss.append(loss_test)
    test_r2.append(r2_test)

    input_title = str(fold)+'_fold_'+str(ep)+'_epoch_'+'_Loss_'+title
    show_picture(train_loss,val_loss, test_loss, input_title,path=img_path, save=True)
    input_title =  str(fold)+'_fold_'+str(ep)+'_epoch_'+'_PCC_'+title
    show_picture(train_pcc,val_pcc, test_pcc, input_title,path=img_path, save=True)
    input_title =  str(fold)+'_fold_'+str(ep)+'_epoch_'+'_r2_'+title
    show_picture(train_r2,val_r2, test_r2, input_title,path=img_path, save=True)

    total_train_pcc.append(train_pcc)
    total_val_pcc.append(val_pcc)

    total_train_r2.append(train_r2)
    total_val_r2.append(val_r2)

    total_train_losses.append(train_loss)
    total_val_losses.append(val_loss)

    total_test_pcc.append(pcc_test)
    total_test_r2.append(r2_test)
    total_test_losses.append(loss_test)

    df_test_pcc = pd.DataFrame(data = total_test_pcc)
    df_test_r2 = pd.DataFrame(data = total_test_r2)

    df_test_losses = pd.DataFrame(data = total_test_losses)

    df_test_pcc.to_csv(result_path+title+'_pcc.csv')
    df_test_r2.to_csv(result_path+title+'_r2.csv')

    df_test_losses.to_csv(result_path+title+'_loss.csv')
