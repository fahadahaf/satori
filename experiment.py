import gensim
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from fastprogress import progress_bar
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from random import randint
from sklearn import metrics
from torch.backends import cudnn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations

#local imports
from datasets import DatasetLoadAll, DatasetLazyLoad, DatasetEmbd
from extract_motifs import get_motif
from models import Basset, AttentionNet
from utils import get_params_dict




###########################################################################################################################
#--------------------------------------------Train and Evaluate Functions-------------------------------------------------#
###########################################################################################################################
def trainRegular(model, device, iterator, optimizer, criterion, useEmb=False):
    model.train()
    running_loss = 0.0
    train_auc = []
    for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
        #pdb.set_trace()
        if useEmb:
            data, target = data.to(device,dtype=torch.long), target.to(device,dtype=torch.long)
        else:
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        #loss = F.binary_cross_entropy(outputs, target)
        labels = target.cpu().numpy()
        
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(outputs)
        pred = pred.cpu().detach().numpy()
        #print(pred)
        try:
            train_auc.append(metrics.roc_auc_score(labels, pred[:,1]))
        except:
            train_auc.append(0.0)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #return outputs
    return running_loss/len(iterator),train_auc


def evaluateRegular(net, device, iterator, criterion, out_dirc=None, getCNN=False, storeCNNout = False, getSeqs = False, useEmb=False):
    #pdb.set_trace()
    running_loss = 0.0
    valid_auc = [] 
    roc = np.asarray([[],[]]).T
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    per_batch_info = {}

    net.eval()
    CNNlayer = net.layer1[0:3]
    CNNlayer.eval()
    
    with torch.no_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
            if useEmb:
                data, target = data.to(device,dtype=torch.long), target.to(device,dtype=torch.long)
            else:
                data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
            # Model computations
            outputs = net(data)
            loss = criterion(outputs, target)
            softmax = torch.nn.Softmax(dim=1)
            labels=target.cpu().numpy()
            pred = softmax(outputs)
            pred=pred.cpu().detach().numpy()
            label_pred = np.column_stack((labels,pred[:,1]))
            per_batch_labelPreds[batch_idx] = label_pred
            roc = np.row_stack((roc,label_pred))

            try:
                valid_auc.append(metrics.roc_auc_score(labels, pred[:,1]))
            except:
                valid_auc.append(0.0)
            running_loss += loss.item()
        
            if getCNN:
                try: #if the network has an embedding layer (input must be embedded as well)
                    data = net.embedding(data)
                    outputCNN = CNNlayer(data.permute(0,2,1))
                except:
                    outputCNN = CNNlayer(data)
                if storeCNNout:
                    if not os.path.exists(out_dirc):
                        os.makedirs(out_dirc)	
                    with open(out_dirc+'/CNNout_batch-'+str(batch_idx)+'.pckl','wb') as f:
	                    pickle.dump(outputCNN.cpu().detach().numpy(),f)
                    per_batch_CNNoutput[batch_idx] = out_dirc+'/CNNout_batch-'+str(batch_idx)+'.pckl'
                else:
                    per_batch_CNNoutput[batch_idx] = outputCNN.cpu().detach().numpy()
            
            if getSeqs:
                per_batch_testSeqs[batch_idx] = np.column_stack((headers,seqs))
            
    labels = roc[:,0]
    preds = roc[:,1]
    valid_auc = metrics.roc_auc_score(labels,preds)
        
    return running_loss/len(iterator),valid_auc,roc,per_batch_labelPreds,per_batch_CNNoutput,per_batch_testSeqs 
###########################################################################################################################
#---------------------------------------------------------End-------------------------------------------------------------#
###########################################################################################################################
def get_indices(dataset_size, test_split, output_dir, shuffle_data=True, seed_val=100, mode='train'):
    indices = list(range(dataset_size))
    split_val = int(np.floor(test_split*dataset_size))
    if shuffle_data:
        np.random.seed(seed_val)
        np.random.shuffle(indices)
    train_indices, test_indices, valid_indices = indices[2*split_val:], indices[:split_val], indices[split_val:2*split_val]
    #--save indices for later use, when testing for example---#
    if mode=='train':
        np.savetxt(output_dir+'/valid_indices.txt', valid_indices, fmt='%s')
        np.savetxt(output_dir+'/test_indices.txt', test_indices, fmt='%s')
        np.savetxt(output_dir+'/train_indices.txt', train_indices, fmt='%s')
    else:
        try:
            valid_indices = np.loadtxt(output_dir+'/valid_indices.txt', dtype=int)
            test_indices = np.loadtxt(output_dir+'/test_indices.txt', dtype=int)
            train_indices = np.loadtxt(output_dir+'/train_indices.txt', dtype=int)
        except:
            print("Error! looks like you haven't trained the model yet. Rerun with --mode train.")
    return train_indices, test_indices, valid_indices


def load_datasets(arg_space, use_embds, batchSize, kmer_len=None, embd_size=None, embd_window=None):
    """
    Loads and processes the data.
    """
    input_prefix = arg_space.inputprefix
    output_dir = 'results/'+arg_space.directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #save arguments to keep record
    with open(output_dir+'/arguments.txt','w') as f:
	    f.writelines(str(arg_space))
    test_split = arg_space.splitperc/100
    if arg_space.verbose:
        print("test/validation split val: %.2f"%test_split)

    modelwv = None
    if not use_embds:
        if arg_space.deskLoad:
            final_dataset = DatasetLazyLoad(input_prefix)
        else:
            final_dataset = DatasetLoadAll(input_prefix)
        #train_indices, test_indices, valid_indices = get_indices(len(final_dataset), test_split, output_dir)
    else:
        w2v_path = arg_space.wvPath+'/' if arg_space.wvPath[-1]!='/' else arg_space.wvPath #'Word2Vec_Models/'
        w2v_filename = 'Word2Vec_Model_kmerLen'+str(kmer_len)+'_win'+str(embd_window)+'_embSize'+str(embd_size)
        modelwv = Word2Vec.load(w2v_path+w2v_filename)
        data_all = DatasetLoadAll(input_prefix, for_embeddings=True)
        final_dataset = pd.merge(data_all.df_seq_final, data_all.df, on='header')[['header','sequence',7]]
        final_dataset = DatasetEmbd(final_dataset.values.tolist(), modelwv, kmer_len)
        #train_indices, test_indices, valid_indices = get_indices(len(final_dataset), test_split, output_dir)
    #pdb.set_trace()    
    train_indices, test_indices, valid_indices = get_indices(len(final_dataset), test_split, output_dir, mode=arg_space.mode)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(final_dataset, batch_size = batchSize, sampler = train_sampler)
    test_loader = DataLoader(final_dataset, batch_size = batchSize, sampler = test_sampler)
    valid_loader = DataLoader(final_dataset, batch_size = batchSize, sampler = valid_sampler)
    return train_loader, test_loader, valid_loader, modelwv, output_dir


def run_experiment(device, arg_space, params):
    """
    Run the main experiment, that is, load the data and train-test the model and generate/store results.
    Args:
        device: (torch.device) Specifies the device (either gpu or cpu).
        arg_space: ArgParser object containing all the user-specified arguments.
        params: (dict) Dictionary of hyperparameters. 
    """
    net_type = arg_space.netType

    num_labels = params['num_classes']
    get_CNNout = params['get_CNNout']
    get_sequences = params['get_seqs']
    batch_size = params['batch_size']
    max_epochs = params['num_epochs']
    use_embds = arg_space.useEmbeddings
    kmer_len, embd_size, embd_window = [None]*3
    if use_embds:
        kmer_len = params['embd_kmersize']
        embd_size = params['embd_size']
        embd_window = params['embd_window']

    prefix = 'modelRes' #Using generic, not sure if we need it as an argument or part of the params dict
    train_loader, test_loader, valid_loader, modelwv, output_dir = load_datasets(arg_space, use_embds, batch_size, kmer_len, embd_size, embd_window)
    #print(params)
    if net_type == 'basset':
        if arg_space.verbose:
            print("Using Basset-like model.")
        net = Basset(params, wvmodel=modelwv, useEmbeddings=use_embds).to(device)
    else:
        if arg_space.verbose:
            print("Using Attention-based model.")
        net = AttentionNet(params, wvmodel=modelwv, useEmbeddings=use_embds, device=device).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(net.parameters())
    ##-------Main train/test loop----------##
    if arg_space.mode == 'train':
        best_valid_loss = np.inf
        best_valid_auc = np.inf
        for epoch in progress_bar(range(1, max_epochs + 1)):
            res_train = trainRegular(net, device, train_loader, optimizer, criterion, useEmb=use_embds)
            res_valid = evaluateRegular(net, device, valid_loader, criterion, useEmb=use_embds)
            res_train_auc = np.asarray(res_train[1]).mean()
            res_train_loss = res_train[0]
            res_valid_auc = np.asarray(res_valid[1]).mean()
            res_valid_loss = res_valid[0]
            if res_valid_loss < best_valid_loss:
                best_valid_loss = res_valid_loss
                best_valid_auc = res_valid_auc
                if arg_space.verbose:
                    print("Best Validation Loss: %.3f and AUC: %.2f"%(best_valid_loss, best_valid_auc), "\n")
                torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':res_valid_loss
                        },output_dir+'/'+prefix+'_model')
    try:    
        checkpoint = torch.load(output_dir+'/'+prefix+'_model')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except:
        print("No pre-trained model found at %s! Please run with --mode set to train."%output_dir)
        return
    #print(net)
    res_test = evaluateRegular(net, device, test_loader, criterion, output_dir+"/Stored_Values",
                               getCNN=get_CNNout, storeCNNout=arg_space.storeCNN, getSeqs=get_sequences, useEmb=use_embds)
    test_loss = res_test[0]
    labels = res_test[2][:,0]
    preds = res_test[2][:,1]
    auc_test = metrics.roc_auc_score(labels, preds)
    if arg_space.verbose:
        print("Test Loss: %.3f and AUC: %.2f"%(test_loss, auc_test), "\n")
    auprc_test = metrics.average_precision_score(labels,preds)
    some_res = [['Test_Loss','Test_AUC', 'Test_AUPRC']]
    some_res.append([test_loss,auc_test,auprc_test])
    #---Calculate roc and prc values---#
    fpr,tpr,thresholds = metrics.roc_curve(labels,preds)
    precision,recall,thresholdsPR = metrics.precision_recall_curve(labels,preds)
    roc_dict = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}
    prc_dict = {'precision':precision, 'recall':recall, 'thresholds':thresholdsPR}
    #---Store results----#
    with open(output_dir+'/'+prefix+'_roc.pckl','wb') as f:
        pickle.dump(roc_dict,f)
    with open(output_dir+'/'+prefix+'_prc.pckl','wb') as f:
        pickle.dump(prc_dict,f)
    np.savetxt(output_dir+'/'+prefix+'_results.txt',some_res,fmt='%s',delimiter='\t')

    CNNWeights = net.layer1[0].weight.cpu().detach().numpy()
    return res_test, CNNWeights


def motif_analysis(res_test, CNNWeights, argSpace, for_negative=False):
    """
    Infer regulatory motifs by analyzing the first CNN layer filters.
    Args:
        res_test: (list) Returned by the experiment function after testing the model.
        CNNWeights: (numpy.ndarray) Weights of the first CNN layer.
        argSpace: The ArgParser object containing values of all the user-specificed arguments.
        for_negative: (bool) Determines if the motif analysis is for the positive or negative set.
    """
    output_dir = 'results/'+argSpace.directory
    if not os.path.exists(output_dir):
        print("Error! output directory doesn't exist.")
        return   
    NumExamples = 0
    pos_score_cutoff = argSpace.scoreCutoff
    k = 0 #batch number
    per_batch_labelPreds = res_test[3][k]
    #per_batch_Embdoutput = res_test[5][k]
    CNNoutput = res_test[4][k]
    if argSpace.storeCNN:
        with open(CNNoutput,'rb') as f:
            CNNoutput = pickle.load(f)
    Seqs = np.asarray(res_test[-1][k])
    if for_negative:
        tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
    else:
        tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>(1-pos_score_cutoff))]
    NumExamples += len(tp_indices)	
    CNNoutput = CNNoutput[tp_indices]
    Seqs = Seqs[tp_indices]

    for k in range(1,len(res_test[3])):
        if argSpace.verbose:
            print("batch number: ",k)
            
        per_batch_labelPreds = res_test[3][k]
        per_batch_CNNoutput = res_test[4][k]
        if argSpace.storeCNN:
            with open(per_batch_CNNoutput,'rb') as f:
                per_batch_CNNoutput = pickle.load(f)
        
        per_batch_seqs = np.asarray(res_test[-1][k])
        if for_negative:
            tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
        else:
            tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>(1-pos_score_cutoff))]
        NumExamples += len(tp_indices)
        CNNoutput = np.concatenate((CNNoutput,per_batch_CNNoutput[tp_indices]),axis=0)
        Seqs = np.concatenate((Seqs,per_batch_seqs[tp_indices]))
    if argSpace.tfDatabase == None:
        dbpath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/motif_databases/CIS-BP/Homo_sapiens.meme'
    else:
        dbpath = argSpace.tfDatabase

    if argSpace.tomtomPath == None:
        tomtomPath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/meme-5.0.3/src/tomtom'
    else:
        tomtomPath = argSpace.tomtomPath
    if for_negative:
        motif_dir = output_dir + '/Motif_Analysis_Negative'
    else:
        motif_dir = output_dir + '/Motif_Analysis'
    get_motif(CNNWeights, CNNoutput, Seqs, dbpath, dir1 = motif_dir, embd=argSpace.useEmbeddings,
                data='DNA', tomtom=tomtomPath, tomtompval=argSpace.tomtomPval, tomtomdist=argSpace.tomtomDist) 
    return motif_dir, NumExamples