#from __future__ import print_function
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations

from fastprogress import progress_bar
from scipy.special import comb  
import pdb

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

#from my_classes import BassetDataset
#from strat_sampler import StratifiedSampler
from torch.backends import cudnn
import numpy as np
from sklearn import metrics
from random import randint 

from tqdm import tqdm
import math
import random

import pickle

#from gensim.models import Word2Vec
#from gensim.models.word2vec import LineSentence
#import gensim

#from skorch import NeuralNetRegressor
#from sklearn.model_selection import RandomizedSearchCV 

#from sklearn.model_selection import ParameterGrid
import os
import sys
from extract_motifs_deepRAM_withActivationScore import get_motif

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bottleneck as bn
from argparse import ArgumentParser 

from multiprocessing import Process
from multiprocessing import Pool

from scipy.stats import fisher_exact 
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

import time
####################################################################################################################
##################################--------------Argument Parsing--------------######################################
def parseArgs():
    """Parse command line arguments
    
    Returns
    -------
    a : argparse.ArgumentParser
    
    """
    parser = ArgumentParser(description='Main deepSAMIREI script.')
    parser.add_argument('-v', '--verbose',dest='verbose', action='store_true', 
                        default=False, help="verbose output [default is quiet running]")
    
    parser.add_argument('-o','--outDir',dest='directory',type=str,
                        action='store',help="output directory", default='')
    parser.add_argument('-m','--mode', dest='mode',type=str,
                        action='store',help="Mode of operation: train or test.", default='train')     
    parser.add_argument('--deskload', dest='deskLoad',
                        action='store_true',default=False,
                        help="Load dataset from desk. If false, the data is converted into tensors and kept in main memory (not recommended for large datasets).")  
    parser.add_argument('-w','--numworkers',dest='numWorkers',type=int,
                        action='store',help="Number of workers used in data loader. For loading from the desk, use more than 1 for faster fetching.", default=1)        
    #parser.add_argument('-c','--cvtype',dest='cvtype',type=str,action='store',
    #                    help="Type of cross validation to use. Options are: chrom (leave-one-chrom-out), nfolds (N fold CV), and none (No CV, just regular train test split)",
    #                    default='none')
    #parser.add_argument('--numfolds',dest='numfolds',type=int,action='store',
    #                    help="Number of folds for N fold CV", default=10)
    parser.add_argument('--splittype',dest='splitType',type=str, action='store',
                        help="Either to use a percantage of data for valid,test or use specific chromosomes. In the later case, provide chrA,chrB for valid,test. Default value is percent and --splitperc value will be used.", default='percent')
    parser.add_argument('--splitperc',dest='splitperc',type=float, action='store',
                        help="Pecentages of test, and validation data splits, eg. 10 for 10 percent data used for testing and validation.", default=10)
    parser.add_argument('--motifanalysis', dest='motifAnalysis',
                        action='store_true',default=False,
                        help="Analyze CNN filters for motifs and search them against known TF database.")
    parser.add_argument('--scorecutoff',dest='scoreCutoff',type=float,
                        action='store',default=0.65,
                        help="In case of binary labels, the positive probability cutoff to use.")
    parser.add_argument('--tomtompath',dest='tomtomPath',
                        type=str,action='store',default=None,
                        help="Provide path to where TomTom (from MEME suite) is located.") 
    parser.add_argument('--database',dest='tfDatabase',type=str,action='store',
                        help="Search CNN motifs against known TF database. Default is Human CISBP TFs.", default=None)
    parser.add_argument('--annotate',dest='annotateTomTom',type=str,action='store',
                        default=None, help="Annotate tomtom motifs. The options are: 1. path to annotation file, 2. No (not to annotate the output) 3. None (default where human CISBP annotations are used)")                    
    parser.add_argument('-a','--attnfigs', dest='attnFigs',
                        action='store_true',default=False,
                        help="Generate Attention (matrix) figures for every test example.")
    parser.add_argument('-i','--interactions', dest='featInteractions',
                        action='store_true',default=False,
                        help="Self attention based feature(TF) interactions analysis.")
    parser.add_argument('-b','--background', dest='intBackground',type=str,
                        action='store',default=None,
                        help="Background used in interaction analysis: shuffle (for di-nucleotide shuffled sequences with embedded motifs.), negative (for negative test set). Default is not to use background (and significance test).")
    parser.add_argument('--attncutoff', dest='attnCutoff',type=float,
                        action='store',default=0.04,
                        help="Attention (probability) cutoff value to use while searching for maximum interaction. A value (say K) greater than 1.0 will mean using top K interaction values.") #In human promoter DHSs data analysis, lowering the cutoff leads to more TF interactions. 
    parser.add_argument('--intseqlimit', dest='intSeqLimit',type=int,
                        action='store',default = -1,
                        help="A limit on number of input sequences to test. Default is -1 (use all input sequences that qualify).")
    parser.add_argument('-s','--store', dest='storeInterCNN',
                        action='store_true',default=False,
                        help="Store per batch attention and CNN outpout matrices. If false, the are kept in the main memory.")
    parser.add_argument('--considertophit', dest='considerTopHit',
                        action='store_true',default=False,
                        help="Consider only the top matching TF/regulatory element for a filter (from TomTom results).") #This is particularly useful when we have lots of TF hits with fewer (1 or two) filters per TF. Using the top TF match will lead to fewer interactions.
    parser.add_argument('--numlabels', dest='numLabels',type=int,
                        action='store',default = 2,
                        help="Number of labels. 2 for binary (default). For multi-class, multi label problem, can be more than 2. ")
    parser.add_argument('--tomtomdist', dest='tomtomDist',type=str,
                        action='store',default = 'pearson',
                        help="TomTom distance parameter (pearson, kullback, ed etc). Default is pearson. See TomTom help from MEME suite.")
    parser.add_argument('--attrbatchsize', dest='attrBatchSize',type=int,
                        action='store',default = 12,
                        help="Batch size used while calculating attributes. Default is 12.")
    parser.add_argument('--tomtompval', dest='tomtomPval',type=float,
                        action='store',default = 0.05,
                        help="Adjusted p-value cutoff from TomTom. Default is 0.05.")
    parser.add_argument('--testall', dest='testAll',
                        action='store_true',default=False,
                        help="Test on the entire dataset (default False). Useful for interaction/motif analysis.")
    parser.add_argument('--useall', dest='useAll',
                        action='store_true',default=False,
                        help="Use all examples in multi-label problem. Default is False.")
    parser.add_argument('--precisionlimit', dest='precisionLimit',type=float,
                        action='store',default = 0.50,
                        help="Precision limit to use for selecting examples in case of multi-label problem.")					
    parser.add_argument('inputprefix', type=str,
                        help="Input file prefix for the bed/text file and the corresponding fasta file (sequences).")
    parser.add_argument('hparamfile',type=str,
                        help='Name of the hyperparameters file to be used.')
    
    
    
    args = parser.parse_args()
    #if not validateArgs( args ):
    #    raise Exception("Argument Errors: check arguments and usage!")
    return args



####################################################################################################################

####################################################################################################################
##############################################--------Main Network Class--------####################################

#courtesy https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa
class soft_exponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential,self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
       
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
       
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
	
class Generalized_Net(nn.Module):
	def __init__(self, params, genPAttn=True, wvmodel=None, numClasses = 2):
		super(Generalized_Net, self).__init__()
		
		#if embedding:
		#model1 = gensim.models.Word2Vec.load(word2vec_model)
		
		self.embSize = params['embd_size']#embSize
		self.numMultiHeads = 8#numMultiHeads
		self.SingleHeadSize = params['singlehead_size']#SingleHeadSize
		self.MultiHeadSize = params['multihead_size']#MultiHeadSize
		self.usepooling = params['usepooling']
		self.pooling_val = params['pooling_val']
		self.readout_strategy = params['readout_strategy']
		self.kmerSize = params['kmer_size']
		self.useRNN = params['use_RNN']
		self.useCNN = params['use_CNN']
		self.usePE = params['use_posEnc']
		self.useCNNpool = params['use_CNNpool']
		self.RNN_hiddenSize = params['RNNhiddenSize']
		self.numCNNfilters = params['CNN_filters']
		self.filterSize = params['CNN_filtersize']
		self.CNNpoolSize = params['CNN_poolsize']
		self.numClasses = numClasses
		#self.seqLen
		
		self.genPAttn = genPAttn
		
		#weights = torch.FloatTensor(wvmodel.wv.vectors)
		#self.embedding = nn.Embedding.from_pretrained(weights, freeze=False) #False before
		if wvmodel == None:
			self.numInputChannels = params['input_channels'] #number of channels, one hot encoding
		else:
			self.numInputChannels = self.embSize
		
		if self.usePE:
			self.pe = PositionalEncoding(d_model = self.numInputChannels, dropout=0.1)
		
		if self.useCNN and self.useCNNpool:
			self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters,
										 kernel_size=self.filterSize, padding=6),nn.BatchNorm1d(num_features=self.numCNNfilters),
										 nn.ReLU(),nn.MaxPool1d(kernel_size=self.CNNpoolSize))
			#self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.embSize, out_channels=200, kernel_size=13, padding=6),nn.BatchNorm1d(num_features=200),nn.Softplus(),nn.MaxPool1d(kernel_size=6))
			#self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.embSize, out_channels=200, kernel_size=13, padding=6),nn.BatchNorm1d(num_features=200),soft_exponential(200,alpha=0.05),nn.MaxPool1d(kernel_size=6))
			self.dropout1 = nn.Dropout(p=0.2)
        
		if self.useCNN and self.useCNNpool == False:
			self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters,
										 kernel_size=self.filterSize, padding=6),
										 nn.BatchNorm1d(num_features=self.numCNNfilters),nn.ReLU())
			#self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.embSize, out_channels=200, kernel_size=13, padding=6),nn.BatchNorm1d(num_features=200),nn.Softplus())
			#self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.embSize, out_channels=200, kernel_size=13, padding=6),nn.BatchNorm1d(num_features=200),soft_exponential(200,alpha=0.05),)
			self.dropout1 = nn.Dropout(p=0.2)
		
		if self.useRNN:
			self.RNN = nn.LSTM(self.numInputChannels if self.useCNN==False else self.numCNNfilters, self.RNN_hiddenSize, num_layers=2, bidirectional=True)
			self.dropoutRNN = nn.Dropout(p=0.4)
			self.Q = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			self.K = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			self.V = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
		
		if self.useRNN == False and self.useCNN == False:
			self.Q = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			self.K = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			self.V = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
		
		if self.useRNN == False and self.useCNN == True:
			self.Q = nn.ModuleList([nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			self.K = nn.ModuleList([nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			self.V = nn.ModuleList([nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
			
		
		#num_vectors = (600-self.kmerSize) + 1
		
		#if self.usepooling==True:
		#	self.MAXPOOL = nn.ModuleList([nn.MaxPool1d(kernel_size=self.pooling_val) for i in range(0,self.numMultiHeads)])
		#	num_vectors = int(num_vectors/self.pooling_val) 
		
		self.RELU = nn.ModuleList([nn.ReLU() for i in range(0,self.numMultiHeads)])
		
		self.MultiHeadLinear = nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.MultiHeadSize)#50
		
		self.MHReLU = nn.ReLU()
		
		#self.alpha = nn.Linear(in_features=50, out_features=50)
		
		
		#if self.readout_strategy == 'flatten':
		#	fc1_in_size = num_vectors*self.MultiHeadSize#591*32 #591 words/kmers/low-D vectors each of size 32 (hard-coding for now)
		#	self.fc1 = nn.Linear(in_features=fc1_in_size, out_features=1000)
		#	self.relu4 = nn.ReLU()
		#	self.dropout4 = nn.Dropout(p=0.4)
	
		#	self.fc2 = nn.Linear(in_features=1000, out_features=self.MultiHeadSize) 
		#	self.relu5 = nn.ReLU()
		#	self.dropout5 = nn.Dropout(p=0.4)
	
		self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=self.numClasses)
		
	def attention(self, query, key, value, mask=None, dropout=0.0):
		"Compute 'Scaled Dot Product Attention'"
		d_k = query.size(-1)
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
		#if mask is not None:
		#    scores = scores.masked_fill(mask == 0, -1e9)
		p_attn = F.softmax(scores, dim = -1)
		#scores = F.softmax(scores, dim = -1)
		# (Dropout described below)
		p_attn = F.dropout(p_attn, p=dropout,training=self.training)
		#scores = F.dropout(p_attn,p=dropout)
		return torch.matmul(p_attn, value), p_attn
	
	def forward(self, inputs):
		#pdb.set_trace()
		
		#output1 = self.embedding(inputs)
		
		#output = self.embedding(inputs)
		output = inputs
		if self.usePE:
			output = self.pe(output)
		
		if self.useCNN:
			#output = output.permute(0,2,1)
			output = self.layer1(output)
			output = self.dropout1(output)
			output = output.permute(0,2,1)
		
		if self.useRNN:
			output, _ = self.RNN(output)
			F_RNN = output[:,:,:self.RNN_hiddenSize]
			R_RNN = output[:,:,self.RNN_hiddenSize:] #before I wrote :self.RNN_hiddenSize for the reverse part too (forwarnRNNonly results are based on that). That is basically forward RNN concatenated with itself (perhaps very similar to single direction LSTM)
			output = torch.cat((F_RNN,R_RNN),2)
			output = self.dropoutRNN(output)
		
		pAttn_concat = torch.Tensor([]).to(device)
		attn_concat = torch.Tensor([]).to(device)
		for i in range(0,self.numMultiHeads):
			query, key, value = self.Q[i](output), self.K[i](output), self.V[i](output)
			
			#attnOut, attn = self.attention(query, key, value, dropout=0.2)
			attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
			attnOut = self.RELU[i](attnOut)
			if self.usepooling:
				attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
			attn_concat = torch.cat((attn_concat,attnOut),dim=2)
			if self.genPAttn:
				pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)
			#else:
			#	pAttn_concat = torch.cat(pAttn_concat,torch.tensor([0]))
		
		output = self.MultiHeadLinear(attn_concat)
		
		output = self.MHReLU(output)
		
		
		#output2 = self.alpha(output1)
		
		#output = output + output2
	
		#output = output.reshape(output.size(0), -1)
		#output = (output-output.mean())/output.std()
		if self.readout_strategy == 'normalize':
			output = output.sum(axis=1)
			output = (output-output.mean())/output.std()
		#elif self.readout_strategy == 'flatten':
		#	output = output.reshape(output.size(0), -1)
		#	
		#	output = self.fc1(output)
		#	output = self.relu4(output)
		#	output = self.dropout4(output)
	
		#	output = self.fc2(output)
		#	output = self.relu5(output)
		#	output = self.dropout5(output)
	
		output = self.fc3(output)
	
		assert not torch.isnan(output).any()
		
		return output
    
######################################################################################################################   

######################################################################################################################
#####################################-------------------Data Processing Code------------##############################

class ProcessedDataVersion2(Dataset):
    def __init__(self, df_path, num_labels = 2, for_embeddings=False):
        self.DNAalphabet = {'A':'0', 'C':'1', 'G':'2', 'T':'3'}
        df_path = df_path.split('.')[0] #just in case the user provide extension
        self.df_all = pd.read_csv(df_path+'.txt',delimiter='\t',header=None)
        self.df_seq = pd.read_csv(df_path+'.fa',header=None)
        strand = self.df_seq[0][0][-3:] #can be (+) or (.) 
        self.df_all['header'] = self.df_all.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+strand, axis=1)
        
        self.chroms = self.df_all[0].unique()
        self.df_seq_all = pd.concat([self.df_seq[::2].reset_index(drop=True), self.df_seq[1::2].reset_index(drop=True)], axis=1, sort=False)
        self.df_seq_all.columns = ["header","sequence"]
        #self.df_seq_all['chrom'] = self.df_seq_all['header'].apply(lambda x: x.strip('>').split(':')[0])
        self.df_seq_all['sequence'].apply(lambda x: x.upper())
        self.num_labels = num_labels
        
        self.df = self.df_all
        self.df_seq_final = self.df_seq_all
            

        self.df = self.df.reset_index()
        self.df_seq_final = self.df_seq_final.reset_index()
        #self.df['header'] = self.df.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+'('+x[5]+')', axis=1)
        if for_embeddings == False:
            self.One_hot_Encoded_Tensors = []
            self.Label_Tensors = []
            self.Seqs = []
            self.Header = []
            for i in progress_bar(range(0,self.df.shape[0])): #tqdm() before
                if self.num_labels == 2:
                    y = self.df[self.df.columns[-2]][i]
                else:
                    y = np.asarray(self.df[self.df.columns[-2]][i].split(',')).astype(int)
                    y = self.one_hot_encode_labels(y)
                header = self.df['header'][i]
                self.Header.append(header)
                X = self.df_seq_final['sequence'][self.df_seq_final['header']==header].array[0].upper()
                X = X.replace('N',list(self.DNAalphabet.keys())[randint(0,3)])
                X = X.replace('N',list(self.DNAalphabet.keys())[random.choice([0,1,2,3])])
                X = X.replace('S',list(self.DNAalphabet.keys())[random.choice([1,2])])
                X = X.replace('W',list(self.DNAalphabet.keys())[random.choice([0,3])])
                X = X.replace('K',list(self.DNAalphabet.keys())[random.choice([2,3])])
                X = X.replace('Y',list(self.DNAalphabet.keys())[random.choice([1,3])])
                X = X.replace('R',list(self.DNAalphabet.keys())[random.choice([0,2])])
                X = X.replace('M',list(self.DNAalphabet.keys())[random.choice([0,1])])
                self.Seqs.append(X)
                X = self.one_hot_encode(X)
                self.One_hot_Encoded_Tensors.append(torch.tensor(X))
                self.Label_Tensors.append(torch.tensor(y))
        
    def __len__(self):
        return self.df.shape[0]
    
    def get_all_data(self):
        return self.df, self.df_seq_final
    
    def get_all_chroms(self):
        return self.chroms
    
    def one_hot_encode(self,seq):
        mapping = dict(zip("ACGT", range(4)))    
        seq2 = [mapping[i] for i in seq]
        return np.eye(4)[seq2].T.astype(np.long)
  
    def one_hot_encode_labels(self,y):
        lbArr = np.zeros(self.num_labels)
        lbArr[y] = 1
        return lbArr.astype(np.float)
    
    def __getitem__(self, idx):
        return self.Header[idx],self.Seqs[idx],self.One_hot_Encoded_Tensors[idx],self.Label_Tensors[idx]


class ProcessedDataVersion2A(Dataset):
    def __init__(self, df_path, num_labels = 2, for_embeddings=False):
        self.DNAalphabet = {'A':'0', 'C':'1', 'G':'2', 'T':'3'}
        df_path = df_path.split('.')[0] #just in case the user provide extension
        self.df_all = pd.read_csv(df_path+'.txt',delimiter='\t',header=None)
        self.df_seq = pd.read_csv(df_path+'.fa',header=None)
        strand = self.df_seq[0][0][-3:] #can be (+) or (.) 
        self.df_all['header'] = self.df_all.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+strand, axis=1)

        
        self.chroms = self.df_all[0].unique()
        self.df_seq_all = pd.concat([self.df_seq[::2].reset_index(drop=True), self.df_seq[1::2].reset_index(drop=True)], axis=1, sort=False)
        self.df_seq_all.columns = ["header","sequence"]
        #self.df_seq_all['chrom'] = self.df_seq_all['header'].apply(lambda x: x.strip('>').split(':')[0])
        self.df_seq_all['sequence'].apply(lambda x: x.upper())
        self.num_labels = num_labels
        
        self.df = self.df_all
        self.df_seq_final = self.df_seq_all
            

        self.df = self.df.reset_index()
        self.df_seq_final = self.df_seq_final.reset_index()
        #self.df['header'] = self.df.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+'('+x[5]+')', axis=1)
        
    def __len__(self):
        return self.df.shape[0]
    
    def get_all_data(self):
        return self.df, self.df_seq_final
    
    def get_all_chroms(self):
        return self.chroms
    
    def one_hot_encode(self,seq):
        mapping = dict(zip("ACGT", range(4)))    
        seq2 = [mapping[i] for i in seq]
        return np.eye(4)[seq2].T.astype(np.long)
  
    def one_hot_encode_labels(self,y):
        lbArr = np.zeros(self.num_labels)
        lbArr[y] = 1
        return lbArr.astype(np.float)
    
    def __getitem__(self, idx):
        if self.num_labels == 2:
            y = self.df[self.df.columns[-2]][idx]
        else:
            y = np.asarray(self.df[self.df.columns[-2]][idx].split(',')).astype(int)
            y = self.one_hot_encode_labels(y)
        header = self.df['header'][idx]
        X = self.df_seq_final['sequence'][self.df_seq_final['header']==header].array[0].upper()
        X = X.replace('N',list(self.DNAalphabet.keys())[randint(0,3)])
        X = X.replace('N',list(self.DNAalphabet.keys())[random.choice([0,1,2,3])])
        X = X.replace('S',list(self.DNAalphabet.keys())[random.choice([1,2])])
        X = X.replace('W',list(self.DNAalphabet.keys())[random.choice([0,3])])
        X = X.replace('K',list(self.DNAalphabet.keys())[random.choice([2,3])])
        X = X.replace('Y',list(self.DNAalphabet.keys())[random.choice([1,3])])
        X = X.replace('R',list(self.DNAalphabet.keys())[random.choice([0,2])])
        X = X.replace('M',list(self.DNAalphabet.keys())[random.choice([0,1])])
        seq = X 
        X = self.one_hot_encode(X)
        return header,seq,torch.tensor(X),torch.tensor(y)

        
#######################################################################################################################

#######################################################################################################################
################################--------------Train and Evaluate Functions---------------##############################

def trainRegularMC(model, device, iterator, optimizer, criterion):
    model.train()
    running_loss = 0.0
    train_auc = []
    all_labels = []
    all_preds = []
    count = 0
    #pdb.set_trace()
    for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
        #pdb.set_trace()
        data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        #loss = F.binary_cross_entropy(outputs, target)
        
        labels = target.cpu().numpy()
        
		#softmax = torch.nn.Softmax(dim=0) #along columns
        #should be using sigmoid here
        sigmoid = torch.nn.Sigmoid()
			
		#pred = softmax(outputs)
        pred = sigmoid(outputs)
        pred = pred.cpu().detach().numpy()


        all_labels+=labels.tolist()
        all_preds+=pred.tolist()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

    for j in range(0, len(all_labels[0])):
        cls_labels = np.asarray(all_labels)[:,j]
        pred_probs = np.asarray(all_preds)[:,j]
        auc_score = metrics.roc_auc_score(cls_labels.astype(int),pred_probs)
        train_auc.append(auc_score)
    
 
    return running_loss/len(train_loader),train_auc

def trainRegular(model, device, iterator, optimizer, criterion):
    model.train()
    running_loss = 0.0
    train_auc = []
    for batch_idx, (headers, seqs, data, target) in enumerate(iterator):
        #pdb.set_trace()
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
    return running_loss/len(train_loader),train_auc




def evaluateRegularMC(net, iterator, criterion, out_dirc, getPAttn=False, storePAttn = False, getCNN=False, storeCNNout = False, getSeqs = False):
    
    running_loss = 0.0
    valid_auc = []
    
    net.eval()
    
    roc = np.asarray([[],[]]).T
    PAttn_all = {}
    all_labels = []
    all_preds = []
    
    
    running_loss = 0.0
    valid_auc = []
    
    net.eval()
    
    #embd = net.embedding #get the embeddings, we need that for first conv layer
    CNNlayer = net.layer1[0:3] #first conv layer without the maxpooling part

    #embd.eval()
    CNNlayer.eval()
    
    roc = np.asarray([[],[]]).T
    PAttn_all = {}
    
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    per_batch_info = {}
    
    
    with torch.no_grad():
    
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):

            data, target = data.to(device,dtype=torch.float), target.to(device, dtype=torch.float)
            # Model computations
            outputs = net(data)
            PAttn = [] #just a dummy
            
            loss = criterion(outputs, target)
            #loss = F.binary_cross_entropy(outputs, target)
            
            labels=target.cpu().numpy()
            
            #softmax = torch.nn.Softmax(dim=0) #along columns
            #should be using sigmoid here
            sigmoid = torch.nn.Sigmoid()
			
			#pred = softmax(outputs)
            pred = sigmoid(outputs)
            pred = pred.cpu().detach().numpy()
        
            #pred = pred.cpu().detach().numpy()
        
            all_labels+=labels.tolist()
            all_preds+=pred.tolist()
            
            label_pred = {'labels':labels,'preds':pred}
            
            per_batch_labelPreds[batch_idx] = label_pred
            
            #softmax = torch.nn.Softmax(dim=1)
            
            
            #labels=target.cpu().numpy()
            
            #pred = softmax(outputs)
            #print(pred)
            #valid_auc.append(metrics.roc_auc_score(labels, pred))
            
            #pred = torch.argmax(pred, dim=1)
            if getPAttn == True:
                if storePAttn == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)					
                    with open(output_dir+'/PAttn_batch-'+str(batch_idx)+'.pckl','wb') as f:
	                    pickle.dump(PAttn.cpu().detach().numpy(),f)
                    PAttn_all[batch_idx] = output_dir+'/PAttn_batch-'+str(batch_idx)+'.pckl' #paths to the pickle PAttention
                else:
                    PAttn_all[batch_idx] = PAttn.cpu().detach().numpy()
            #pred = torch.argmax(pred, dim=1)
            #pred=pred.cpu().detach().numpy()
            
            if getCNN == True:
                outputCNN = CNNlayer(data)
                if storeCNNout == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)	
                    with open(output_dir+'/CNNout_batch-'+str(batch_idx)+'.pckl','wb') as f:
                        pickle.dump(outputCNN.cpu().detach().numpy(),f)
                    per_batch_CNNoutput[batch_idx] = output_dir+'/CNNout_batch-'+str(batch_idx)+'.pckl'
                else:
                    per_batch_CNNoutput[batch_idx] = outputCNN.cpu().detach().numpy()
                    
                    
            if getSeqs == True:
                per_batch_testSeqs[batch_idx] = np.column_stack((headers,seqs))
            
            #label_pred = np.column_stack((labels,pred[:,1]))
            #roc = np.row_stack((roc,label_pred))
            #print(pred)
            #try:
            #    valid_auc.append(metrics.roc_auc_score(labels, pred[:,1]))
            #except:
            #    valid_auc.append(0.0)
            

            
            running_loss += loss.item()
    
    for j in range(0, len(all_labels[0])):
        cls_labels = np.asarray(all_labels)[:,j]
        pred_probs = np.asarray(all_preds)[:,j]
        auc_score = metrics.roc_auc_score(cls_labels.astype(int),pred_probs)
        valid_auc.append(auc_score)
        
    return running_loss/len(iterator),valid_auc,roc,PAttn_all,per_batch_labelPreds,per_batch_CNNoutput,per_batch_testSeqs  


             

def evaluateRegular(net, iterator, criterion,out_dirc,getPAttn=False, storePAttn = False, getCNN=False, storeCNNout = False, getSeqs = False):
    
    running_loss = 0.0
    valid_auc = []
    
    net.eval()
    
    #embd = net.embedding #get the embeddings, we need that for first conv layer
    CNNlayer = net.layer1[0:3] #first conv layer without the maxpooling part

    #embd.eval()
    CNNlayer.eval()
    
    roc = np.asarray([[],[]]).T
    PAttn_all = {}
    
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    per_batch_info = {}
    
    with torch.no_grad():
    
        for batch_idx, (headers, seqs, data, target) in enumerate(iterator):

            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
            # Model computations
            outputs = net(data)
            PAttn = []
            
            loss = criterion(outputs, target)
            #loss = F.binary_cross_entropy(outputs, target)
            
            softmax = torch.nn.Softmax(dim=1)
            
            
            labels=target.cpu().numpy()
            
            pred = softmax(outputs)
            
            #print(pred)
            #valid_auc.append(metrics.roc_auc_score(labels, pred))
            if getPAttn == True:
                if storePAttn == True:
	                output_dir = out_dirc
	                if not os.path.exists(output_dir):
	                    os.makedirs(output_dir)	
	                with open(output_dir+'/PAttn_batch-'+str(batch_idx)+'.pckl','wb') as f:
	                    pickle.dump(PAttn.cpu().detach().numpy(),f)
	                PAttn_all[batch_idx] = output_dir+'/PAttn_batch-'+str(batch_idx)+'.pckl' #paths to the pickle PAttention
                else:
                    PAttn_all[batch_idx] = PAttn.cpu().detach().numpy()
            #pred = torch.argmax(pred, dim=1)
            pred=pred.cpu().detach().numpy()
            
            label_pred = np.column_stack((labels,pred[:,1]))
            per_batch_labelPreds[batch_idx] = label_pred
            roc = np.row_stack((roc,label_pred))
            
            #print(pred)
            try:
                valid_auc.append(metrics.roc_auc_score(labels, pred[:,1]))
            except:
                valid_auc.append(0.0)
            
            
            running_loss += loss.item()
            
            
            #outputEmb = embd(data)
            #outputEmb = outputEmb.permute(0,2,1) #to make it compatible with next layer (CNN)
            outputCNN = CNNlayer(data).cpu().detach().numpy()


            #per_batch_Embdoutput[batch_idx] = outputEmb
            #per_batch_CNNoutput[batch_idx] = outputCNN
            if getCNN == True:
                outputCNN = CNNlayer(data)
                if storeCNNout == True:
                    output_dir = out_dirc
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)	
                    with open(output_dir+'/CNNout_batch-'+str(batch_idx)+'.pckl','wb') as f:
	                    pickle.dump(outputCNN.cpu().detach().numpy(),f)
                    per_batch_CNNoutput[batch_idx] = output_dir+'/CNNout_batch-'+str(batch_idx)+'.pckl'
                else:
                    per_batch_CNNoutput[batch_idx] = outputCNN.cpu().detach().numpy()


            #batch_test_indices = test_indices[batch_idx*batchSize:(batch_idx*batchSize)+batchSize]

            #batch_test_seqs = np.asarray(data_all.df_seq_all['sequence'][batch_test_indices])
            #batch_test_targets = np.asarray(data_all.df_all[7][batch_test_indices])
            
            if getSeqs == True:
                per_batch_testSeqs[batch_idx] = np.column_stack((headers,seqs))
            #per_batch_info[batch_idx] = [batch_test_targets,target]
            
            
    labels = roc[:,0]
    preds = roc[:,1]
    valid_auc = metrics.roc_auc_score(labels,preds)
        
    return running_loss/len(iterator),valid_auc,roc,PAttn_all,per_batch_labelPreds,per_batch_CNNoutput,per_batch_testSeqs    
    

    
########################################################################################################################

########################################################################################################################
############################---------------Embedding Class and Functions---------------#################################
########################################################################################################################     
##code borrowed from DeepRAM work
#class dataset_embd(Dataset):

    #def __init__(self,xy=None,model=None,kmer_len=5,stride=2):
      
        #self.kmer_len= kmer_len
        #self.stride= stride
        #data=[el[0] for el in xy]
        #words_doc= self.Gen_Words(data,self.kmer_len,self.stride)
##         print(words_doc[0])
        #x_data=[self.convert_data_to_index(el,model.wv) for el in words_doc]
        ##print(x_data.shape)
       
        
        #self.x_data=np.asarray(x_data,dtype=np.float32)
        #self.seq_data = np.asarray([el[0] for el in xy])
        #self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
        #self.x_data = torch.LongTensor(self.x_data)
        #self.y_data = torch.from_numpy(self.y_data)
        #self.len=len(self.x_data)
      

    #def __getitem__(self, index):
        #return self.seq_data[index], self.x_data[index], self.y_data[index]

    #def __len__(self):
        #return self.len
      
    #def Gen_Words(self,pos_data,kmer_len,s):
        #out=[]
        
        #for i in pos_data:

            #kmer_list=[]
            #for j in range(0,(len(i)-kmer_len)+1,s):

                  #kmer_list.append(i[j:j+kmer_len])
                
            #out.append(kmer_list)
            
        #return out
    
    #def convert_data_to_index(self, string_data, wv):
        #index_data = []
        #for word in string_data:
            #if word in wv:
                #index_data.append(wv.vocab[word].index)
        #return index_data

#def Gen_Words(pos_data,kmer_len,s):
        #out=[]
        
        #for i in pos_data:

            #kmer_list=[]
            #for j in range(0,(len(i)-kmer_len)+1,s):

                  #kmer_list.append(i[j:j+kmer_len])
                
            #out.append(kmer_list)
            
        #return out

#def convert_data_to_index(string_data, wv):
        #index_data = []
        #for word in string_data:
            #if word in wv:
                #index_data.append(wv.vocab[word].index)
            #else:
                #print(word)
        #return index_data

########################################################################################################################

########################################################################################################################
##########################------------------Other Functions--------------------#########################################
########################################################################################################################
#courtesy: https://gist.github.com/tomerfiliba/3698403
#get top N values from an ND array
def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:] #argparsort originally but new bottleneck doesn't have that
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


#--------Get shuffled background-----------#
from deeplift_dinuc_shuffle import *
from Bio.motifs import minimal

def get_shuffled_background(tst_loader,argspace): #this function randomly embed the consensus sequence of filter motifs in shuffled input sequences
	labels_array = np.asarray([i for i in range(0,argSpace.numLabels)])
	final_fa = []
	final_bed = []
	for batch_idx, (headers,seqs,_,batch_targets) in enumerate(tst_loader):
		for i in range (0, len(headers)):
			header = headers[i]
			seq = seqs[i]
			targets = batch_targets[i]
			dinuc_shuffled_seq = dinuc_shuffle(seq)
			hdr = header.strip('>').split('(')[0]
			chrom = hdr.split(':')[0]
			start,end = hdr.split(':')[1].split('-')
			final_fa.append([header,seq])
			
			if type(targets) == torch.Tensor:
				targets = targets.cpu().detach().numpy()
				
			target = targets.astype(int)
			
			labels = ','.join(labels_array[np.where(target==1)].astype(str)) 
			
			final_bed.append([chrom,start,end,'.',labels])
	
	
	final_fa_to_write = []
	#--load motifs
	with open(argspace.directory+'/Motif_Analysis/filters_meme.txt') as f:
		filter_motifs = minimal.read(f)
	
	motif_len = filter_motifs[0].length
	
	seq_numbers = [i for i in range(0,len(final_bed))]	
	seq_positions = [i for i in range(0,len(final_fa[0][1])-motif_len)] #can't go beyond end of sequence so have to subtract motif_len
	for i in progress_bar(range(0, len(filter_motifs))):
		motif = filter_motifs[i]
		consensus = motif.consensus
		num_occ = motif.num_occurrences
		random_seqs = random.choices(seq_numbers, k = num_occ) #randomly picking num_occ sequences (note that num_occ can be greater than population in this case since a single sequence can have multile occurence of a filter activation)
		#print(num_occ, len(seq_positions))
		random_positions = random.choices(seq_positions, k = num_occ) #randomly pick a position for a motif to occur
		
		for seq_index, pos in zip(random_seqs,random_positions):
			seq = final_fa[seq_index][1]
			seq = seq[:pos]+str(motif.consensus)+seq[pos+len(motif.consensus):]
			
			final_fa[seq_index][1] = seq
	
	out_directory = argspace.directory+'/Temp_Data'
	if not os.path.exists(out_directory):
		os.makedirs(out_directory)
	
	np.savetxt(out_directory+'/'+'shuffled_background.fa',np.asarray(final_fa).flatten(),fmt='%s')
	np.savetxt(out_directory+'/'+'shuffled_background.txt',np.asarray(final_bed), fmt='%s',delimiter='\t')
	
	return out_directory+'/'+'shuffled_background' #name of the prefix to use
	#
	#return final_bed,final_fa
		


def get_shuffled_background_v2(tst_loader, num_filters, motif_len, argspace): #this function uses the actual filter activation k-mers to embed them randomly in the shuffled input sequences (instead of using the filter's consensus)
	labels_array = np.asarray([i for i in range(0,argSpace.numLabels)])
	final_fa = []
	final_bed = []
	for batch_idx, (headers,seqs,_,batch_targets) in enumerate(tst_loader):
		for i in range (0, len(headers)):
			header = headers[i]
			seq = seqs[i]
			targets = batch_targets[i]
			dinuc_shuffled_seq = dinuc_shuffle(seq)
			hdr = header.strip('>').split('(')[0]
			chrom = hdr.split(':')[0]
			start,end = hdr.split(':')[1].split('-')
			final_fa.append([header,seq])
			
			if type(targets) == torch.Tensor:
				targets = targets.cpu().detach().numpy()
				
			target = targets.astype(int)
			
			labels = ','.join(labels_array[np.where(target==1)].astype(str)) 
			
			final_bed.append([chrom,start,end,'.',labels])
	
	
	final_fa_to_write = []
	#--load motifs
	#with open(argspace.directory+'/Motif_Analysis/filters_meme.txt') as f:
	#	filter_motifs = minimal.read(f)
	
	#motif_len = filter_motifs[0].length
	
	seq_numbers = [i for i in range(0,len(final_bed))]	
	seq_positions = [i for i in range(0,len(final_fa[0][1])-motif_len)] #can't go beyond end of sequence so have to subtract motif_len
	for i in progress_bar(range(0, num_filters)):
		motif = np.loadtxt(argspace.directory+'/Motif_Analysis/filter'+str(i)+'_logo.fa',dtype=str)        #filter_motifs[i]
		#consensus = motif.consensus
		num_occ = int(len(motif)/2) #fasta file is twice the size (with header and seq on separate lines)
		random_seqs = random.choices(seq_numbers, k = num_occ) #randomly picking num_occ sequences (note that num_occ can be greater than population in this case since a single sequence can have multile occurence of a filter activation)
		#print(num_occ, len(seq_positions))
		random_positions = random.choices(seq_positions, k = num_occ) #randomly pick a position for a motif to occur
		
		count = 1
		for seq_index, pos in zip(random_seqs,random_positions):
			emb_kmer = motif[count]
			seq = final_fa[seq_index][1]
			seq = seq[:pos]+emb_kmer+seq[pos+len(emb_kmer):]
			
			final_fa[seq_index][1] = seq
			count += 2
	
	out_directory = argspace.directory+'/Temp_Data'
	if not os.path.exists(out_directory):
		os.makedirs(out_directory)
	
	np.savetxt(out_directory+'/'+'shuffled_background.fa',np.asarray(final_fa).flatten(),fmt='%s')
	np.savetxt(out_directory+'/'+'shuffled_background.txt',np.asarray(final_bed), fmt='%s',delimiter='\t')
	
	return out_directory+'/'+'shuffled_background' #name of the prefix to use
		
		
	######--------For random embedding of filter seqs-------------######
###########background v3#########
def get_random_seq(pwm,alphabet=['A','C','G','T']):
	seq = ''
	for k in range(0,pwm.shape[0]):
		nc = np.random.choice(alphabet,1,p=pwm[k,:])
		seq += nc[0]
	return seq
		
	
def get_shuffled_background_v3(tst_loader,argspace): #this function is similar to the first one (get_shuffled_background()) however, instead of using consensus, it generates a random sequences (of same size as the PWM) based on the probability distributions in the matrix
	labels_array = np.asarray([i for i in range(0,argSpace.numLabels)])
	final_fa = []
	final_bed = []
	for batch_idx, (headers,seqs,_,batch_targets) in enumerate(tst_loader):
		for i in range (0, len(headers)):
			header = headers[i]
			seq = seqs[i]
			targets = batch_targets[i]
			dinuc_shuffled_seq = dinuc_shuffle(seq)
			hdr = header.strip('>').split('(')[0]
			chrom = hdr.split(':')[0]
			start,end = hdr.split(':')[1].split('-')
			final_fa.append([header,seq])
			
			if type(targets) == torch.Tensor:
				targets = targets.cpu().detach().numpy()
				
			target = targets.astype(int)
			
			labels = ','.join(labels_array[np.where(target==1)].astype(str)) 
			
			final_bed.append([chrom,start,end,'.',labels])
	
	
	final_fa_to_write = []
	#--load motifs
	with open(argspace.directory+'/Motif_Analysis/filters_meme.txt') as f:
		filter_motifs = minimal.read(f)
	
	motif_len = filter_motifs[0].length
	
	seq_numbers = [i for i in range(0,len(final_bed))]	
	seq_positions = [i for i in range(0,len(final_fa[0][1])-motif_len)] #can't go beyond end of sequence so have to subtract motif_len
	for i in progress_bar(range(0, len(filter_motifs))):
		motif = filter_motifs[i]
		pwm = np.column_stack((motif.pwm['A'],motif.pwm['C'],motif.pwm['G'],motif.pwm['T']))
		#consensus = motif.consensus
		
		num_occ = motif.num_occurrences
		random_seqs = random.choices(seq_numbers, k = num_occ) #randomly picking num_occ sequences (note that num_occ can be greater than population in this case since a single sequence can have multile occurence of a filter activation)
		#print(num_occ, len(seq_positions))
		random_positions = random.choices(seq_positions, k = num_occ) #randomly pick a position for a motif to occur
		
		for seq_index, pos in zip(random_seqs,random_positions):
			consensus = get_random_seq(pwm) #this will get us a random sequence generated based on the prob distribution of the PWM
			seq = final_fa[seq_index][1]
			seq = seq[:pos]+str(consensus)+seq[pos+len(consensus):]
			
			final_fa[seq_index][1] = seq
	
	out_directory = argspace.directory+'/Temp_Data'
	if not os.path.exists(out_directory):
		os.makedirs(out_directory)
	
	np.savetxt(out_directory+'/'+'shuffled_background.fa',np.asarray(final_fa).flatten(),fmt='%s')
	np.savetxt(out_directory+'/'+'shuffled_background.txt',np.asarray(final_bed), fmt='%s',delimiter='\t')
	
	return out_directory+'/'+'shuffled_background' #name of the prefix to use
	#
	#return final_bed,final_fa
	
	

#########################################################################################################################


##############################----------------Data Processing-----------------##########################################
argSpace = parseArgs()



inp_file_prefix = argSpace.inputprefix
#num_labels = int(sys.argv[2]) #num labels

#########################################################################################################################
#################################-------------------HyperParam Setup--------------------#################################
#########################################################################################################################

###Fixed parameters####
#stride = 1 #embeddings related
#batchSize = 172
#maxEpochs = 30

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
cudnn.benchmark = True



#w2v_path = 'Word2Vec_Models/'
#######################

#################--------------Main Loop-------------------#####################
param_data = np.loadtxt(argSpace.hparamfile,dtype=str)
output_dir = argSpace.directory


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#save arguments to keep record
with open(argSpace.directory+'/arguments.txt','w') as f:
	f.writelines(str(argSpace))

if argSpace.verbose:
	print("Output Directory: ",output_dir)


params = {}
for entry in param_data:
	if entry[1] == 'False':
		params[entry[0]] = False
	elif entry[1] == 'True':
		params[entry[0]] = True
	else:
		try:
			params[entry[0]] = int(entry[1])
		except:
			params[entry[0]] = entry[1]

if argSpace.verbose:
	print("HyperParameters are: ")
	print(params)

#prefix = params['prefix']
num_labels = argSpace.numLabels


genPAttn = params['get_pattn']
getCNNout = params['get_CNNout']
getSequences = params['get_seqs']

batchSize = params['batch_size']
maxEpochs = params['num_epochs']

#if 'CNN' in :
#	genPAttn = True
#else:
#	genPAttn = False

#w2v_filename = 'Word2Vec_Model_kmerLen'+str(params['kmer_size'])+'_win'+str(params['embd_window'])+'_embSize'+str(params['embd_size'])
#modelwv = Word2Vec.load(w2v_path+w2v_filename)

modelwv = None #for now not using word2vec embedings



#some params
test_split = argSpace.splitperc/100 #we need 0.10 for 10% #10% valid,test split
print("test/validation split val: ", test_split)
shuffle_data = True  #internal param
seed_val = 100	#internal param



use_Embds = False #for now, no embeddings
if use_Embds==False:
	if argSpace.deskLoad == False:
		data_all = ProcessedDataVersion2(inp_file_prefix,num_labels) #get all data
	else:
		data_all = ProcessedDataVersion2A(inp_file_prefix,num_labels)	
	
	dataset_size = len(data_all)
	indices = list(range(dataset_size))

	split_val = int(np.floor(test_split*dataset_size))
	if shuffle_data == True:
		np.random.seed(seed_val)
		np.random.shuffle(indices)
	
	if argSpace.splitType != 'percent':
		chrValid, chrTest = argSpace.splitType.split(',') #chr8 for valid and chr18 for test (kundaje paper in plos One). we can try chr4,19,chr16 etc for test since chr18 has the second lowest examples
		df_tmp = data_all.df
		
		test_indices = df_tmp[df_tmp[0]==chrTest].index.to_list()
		valid_indices = df_tmp[df_tmp[0]==chrValid].index.to_list()
		train_indices = df_tmp[~df_tmp[0].isin([chrValid,chrTest])].index.to_list()
		
		
	else:
		if argSpace.mode == 'train':
			train_indices, test_indices, valid_indices = indices[2*split_val:], indices[:split_val], indices[split_val:2*split_val]
			#save test and valid indices
			np.savetxt(argSpace.directory+'/valid_indices.txt',valid_indices,fmt='%s')
			np.savetxt(argSpace.directory+'/test_indices.txt',test_indices,fmt='%s')
			np.savetxt(argSpace.directory+'/train_indices.txt',train_indices,fmt='%s')
		else:
			train_indices = np.loadtxt(argSpace.directory+'/train_indices.txt',dtype=int)
			test_indices = np.loadtxt(argSpace.directory+'/test_indices.txt',dtype=int)
			valid_indices = np.loadtxt(argSpace.directory+'/valid_indices.txt',dtype=int)
		
	train_sampler = SubsetRandomSampler(train_indices)
	test_sampler = SubsetRandomSampler(test_indices)
	valid_sampler = SubsetRandomSampler(valid_indices)
	
	train_loader = DataLoader(data_all, batch_size = batchSize, sampler = train_sampler, num_workers = argSpace.numWorkers)
	test_loader = DataLoader(data_all, batch_size = batchSize, sampler = test_sampler, num_workers = argSpace.numWorkers)
	valid_loader = DataLoader(data_all, batch_size = batchSize, sampler = valid_sampler, num_workers = argSpace.numWorkers)
	
	if argSpace.testAll:
		test_loader = DataLoader(data_all, batch_size = batchSize, sampler=train_sampler, num_workers = argSpace.numWorkers)


net = Generalized_Net(params, genPAttn, wvmodel = modelwv, numClasses = num_labels).to(device) 

if num_labels == 2:
	criterion = nn.CrossEntropyLoss(reduction='mean')
else:
	criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(net.parameters())#, lr=0.002, weight_decay=0.01) #lr =0.002 #try it without lr and weight decay
#def evaluateRegular(net, iterator, criterion,out_dir,getPAttn=False, storePAttn = False, getCNN=False, storeCNNout = False, getSeqs = False):

##Saved Model Directory##
saved_model_dir = output_dir+'/Saved_Model'
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
#########################    

###################-------------Training and Testing------------------#############
if argSpace.mode == 'train':    
	best_valid_loss = np.inf
	best_valid_auc = np.inf
	for epoch in progress_bar(range(1, maxEpochs + 1)):#tqdm
		if num_labels == 2:
			res_train = trainRegular(net, device, train_loader, optimizer, criterion)
		else:
			res_train = trainRegularMC(net, device, train_loader, optimizer, criterion)
		res_train_loss = res_train[0]
		res_train_auc = np.asarray(res_train[1]).mean()
		
		if argSpace.verbose:
			print("Train Results (Loss and AUC): ", res_train_loss, res_train_auc)
	    
		if num_labels == 2:
			res_valid = evaluateRegular(net, valid_loader, criterion, output_dir+"/Stored_Values", getPAttn=False,
										storePAttn = False, getCNN = False,
										storeCNNout = False, getSeqs = False) #evaluateRegular(net,valid_loader,criterion)
			
			res_valid_loss = res_valid[0]
			res_valid_auc = res_valid[1]   
		else:
			res_valid = evaluateRegularMC(net, valid_loader, criterion, output_dir+"/Stored_Values", getPAttn=False,
										storePAttn = False, getCNN = False,
										storeCNNout = False, getSeqs = False) #evaluateRegular(net,valid_loader,criterion)
			
			res_valid_loss = res_valid[0]
			res_valid_auc = np.mean(res_valid[1])  
	        
		if res_valid_loss < best_valid_loss:
			best_valid_loss = res_valid_loss
			best_valid_auc = res_valid_auc
			#if valid_chrom not in auc_dict:
			if argSpace.verbose:
				print("Best Validation (Loss and AUC): ",res_valid[0],res_valid_auc,"\n")
			torch.save({'epoch': epoch,
	                       'model_state_dict': net.state_dict(),
	                       'optimizer_state_dict':optimizer.state_dict(),
	                       'loss':res_valid_loss
	                       },saved_model_dir+'/model')

try:    
	checkpoint = torch.load(saved_model_dir+'/model')
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
except:
	print("No pre-trained model found! Please run with --mode set to train.")

if num_labels == 2:
	res_test = evaluateRegular(net, test_loader, criterion, output_dir+"/Stored_Values", getPAttn = False, #getPAttn
										storePAttn = argSpace.storeInterCNN, getCNN = getCNNout,
										storeCNNout = argSpace.storeInterCNN, getSeqs = getSequences)
										
	test_loss = res_test[0]
	test_auc = res_test[1]
	labels = res_test[2][:,0]
	preds = res_test[2][:,1]
	if argSpace.verbose:
		print("Test Loss and AUC: ",test_loss, test_auc)
	labels = res_test[2][:,0]
	preds = res_test[2][:,1]
	fpr,tpr,thresholds = metrics.roc_curve(labels,preds)
	roc_dict = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}
	with open(output_dir+'/ROC_dict.pckl','wb') as f:
		pickle.dump(roc_dict,f)
	some_res = [['Test_Loss','Test_AUC']]
	some_res.append([[test_loss,test_auc]])
	np.savetxt(output_dir+'/loss_and_auc.txt',some_res,delimiter='\t',fmt='%s')
else:
	res_test = evaluateRegularMC(net, test_loader, criterion, output_dir+"/Stored_Values", getPAttn = False, #genPAttn,
										storePAttn = argSpace.storeInterCNN, getCNN = getCNNout,
										storeCNNout = argSpace.storeInterCNN, getSeqs = getSequences)
	test_loss = res_test[0]
	test_auc = res_test[1]
	if argSpace.verbose:
		print("Test Loss and mean AUC: ",test_loss, np.mean(test_auc))

	np.savetxt(output_dir+'/per_class_AUC.txt',test_auc,delimiter='\t',fmt='%s')


#print(lohahi)


###############################################################################
##############----------Motif Interaction Analysis--------------###############
# running_loss/len(iterator),valid_auc,roc,PAttn_all,per_batch_labelPreds,per_batch_CNNoutput,per_batch_testSeqs

CNNWeights = net.layer1[0].weight.cpu().detach().numpy()
Prob_Attention_All = res_test[3] #for a single batch the dimensions are: BATCH_SIZE x NUM_FEATURES x (NUM_MULTI_HEADS x NUM_FEATURES) #NUM_FEATURES can be num_kmers or num_kmers/convolutionPoolsize (for embeddings)
pos_score_cutoff = argSpace.scoreCutoff

def motif_analysis(res_test, output_dir, argSpace, for_background = False):
	NumExamples = 0
	pos_score_cutoff = argSpace.scoreCutoff
	k = 0 #batch number
	per_batch_labelPreds = res_test[4][k]
	#per_batch_Embdoutput = res_test[5][k]
	CNNoutput = res_test[5][k]
	if argSpace.storeInterCNN:
		with open(CNNoutput,'rb') as f:
			CNNoutput = pickle.load(f)
	Seqs = np.asarray(res_test[6][k])
	
	
	if num_labels == 2:
		if for_background and argSpace.intBackground == 'negative':
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
		elif for_background and argSpace.intBackground == 'shuffle':
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0])]
		else:
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
		
	else:
		tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
		#--Do I need to do the following? that is enrich motifs only in those sequences which qualify based on the precision value--#
		#--Problem with that is how do I handle this approach for background sequences?
		#if argSpace.useAll == True:
		#	tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
		#else:
		#	tp_indices=[]                                                                                                                                                                                    
		#	batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
		#	batch_preds = per_batch_labelPreds['preds']                                                                                                                                                                           
		#	for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
		#		ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
		#		ex_preds = batch_preds[e]                                                                                                                                                                                   
		#		ex_preds = np.asarray([i>=argSpace.precisionLimit for i in ex_preds]).astype(int)                                                                                                                                               
		#		tp_indices.append(e)
	
	NumExamples += len(tp_indices)
		
	CNNoutput = CNNoutput[tp_indices]
	Seqs = Seqs[tp_indices]
	
	for k in range(1,len(res_test[4])):
		if argSpace.verbose:
			print("batch number: ",k)
			
		per_batch_labelPreds = res_test[4][k]
		#per_batch_Embdoutput = res_test[5][k]
		per_batch_CNNoutput = res_test[5][k]
		with open(per_batch_CNNoutput,'rb') as f:
			per_batch_CNNoutput = pickle.load(f)
		
		per_batch_seqs = np.asarray(res_test[6][k])
	
		if num_labels == 2:
			if for_background and argSpace.intBackground == 'negative':
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
			elif for_background and argSpace.intBackground == 'shuffle':
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0])]
			else:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
		else:
			tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
			#if argSpace.useAll == True:
			#	tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
			#else:
			#	tp_indices=[]                                                                                                                                                                                    
			#	batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
			#	batch_preds = per_batch_labelPreds['preds']                                                                                                                                                                           
			#	for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
			#		ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
			#		ex_preds = batch_preds[e]                                                                                                                                                                                   
			#		ex_preds = np.asarray([i>=argSpace.precisionLimit for i in ex_preds]).astype(int)                                                                                                                                               
			#		tp_indices.append(e)
		
		NumExamples += len(tp_indices)
		
		CNNoutput = np.concatenate((CNNoutput,per_batch_CNNoutput[tp_indices]),axis=0)
		Seqs = np.concatenate((Seqs,per_batch_seqs[tp_indices]))

	#pdb.set_trace()
	#get_motif(CNNWeights, CNNoutput, Seqs, dir1 = 'Interactions_Test_noEmbdAttn_'+prefix,embd=True,data='DNA',kmer=kmer_len,s=stride,tomtom='/s/jawar/h/nobackup/fahad/MEME_SUITE/meme-5.0.3/src/tomtom') 
	
	if argSpace.tfDatabase == None:
		dbpath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/motif_databases/CIS-BP/Homo_sapiens.meme'
	else:
		dbpath = argSpace.tfDatabase
	
	if argSpace.tomtomPath == None:
		tomtomPath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/meme-5.0.3/src/tomtom'
	else:
		tomtomPath = argSpace.tomtomPath
	
	if for_background and argSpace.intBackground != None:
		motif_dir = output_dir + '/Motif_Analysis_Negative'
	else:
		motif_dir = output_dir + '/Motif_Analysis'
	
	#dbpath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/motif_databases/CIS-BP/Homo_sapiens.meme'
	#dbpath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/motif_databases/Homo_sapiens_testDFIM.meme'
	#for AT
	#dbpath = '/s/jawar/h/nobackup/fahad/MEME_SUITE/motif_databases/ARABD/ArabidopsisDAPv1.meme'
	
	get_motif(CNNWeights, CNNoutput, Seqs, dbpath, dir1 = motif_dir, embd=False,data='DNA',tomtom=tomtomPath,tomtompval=argSpace.tomtomPval,tomtomdist=argSpace.tomtomDist) 
	
	###-----------------Adding TF details to TomTom results----------------###
	if argSpace.annotateTomTom != 'No':
		tomtom_res = np.loadtxt(motif_dir+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
		if argSpace.annotateTomTom == None:
			database = np.loadtxt('../Basset_Splicing_IR-iDiffIR/Analysis_For_none_network-typeB_lotus_posThresh-0.60/MEME_analysis/Homo_sapiens_2019_01_14_4_17_pm/TF_Information_all_motifs.txt',dtype=str,delimiter='\t')
		else:
			database = argSpace.annotateTomTom
			                              
		final = []                                     
		for entry in tomtom_res[1:]:
			motifID = entry[1]                         
			res = np.argwhere(database[:,3]==motifID)
			TFs = ','.join(database[res.flatten(),6])
			final.append(entry.tolist()+[TFs])
	                                   
		np.savetxt(motif_dir+'/tomtom/tomtom_annotated.tsv',final,delimiter='\t',fmt='%s')
	return motif_dir, NumExamples


if argSpace.motifAnalysis:
	motif_dir,numPosExamples = motif_analysis(res_test, output_dir, argSpace, for_background = False)
	
	if argSpace.intBackground == 'negative':#!= None:
		motif_dir_neg,numNegExamples = motif_analysis(res_test, output_dir, argSpace, for_background = True)
		
	elif argSpace.intBackground == 'shuffle':
		bg_prefix = get_shuffled_background_v3(test_loader,argSpace) #get_shuffled_background_v2(test_loader, params['CNN_filters'], params['CNN_filtersize'], argSpace)  the version 2 doesn't work that well (average AUC 0.50 and 0 motifs in background)
		#data_bg = ProcessedDataVersion2(bg_prefix,num_labels)
		if argSpace.deskLoad == False:
			data_bg = ProcessedDataVersion2(bg_prefix,num_labels) #get all data
		else:
			data_bg = ProcessedDataVersion2A(bg_prefix,num_labels)
			
		test_loader_bg = DataLoader(data_bg,batch_size=batchSize,num_workers=argSpace.numWorkers)	
		if num_labels==2:
			res_test_bg = evaluateRegular(net, test_loader_bg, criterion, out_dirc = output_dir+"/Temp_Data/Stored_Values", getPAttn = False,#genPAttn,
											storePAttn = argSpace.storeInterCNN, getCNN = getCNNout,
											storeCNNout = argSpace.storeInterCNN, getSeqs = getSequences)
			motif_dir_neg,numNegExamples = motif_analysis(res_test_bg, output_dir, argSpace, for_background = True) #in this case the background comes from shuffled sequences and won't be using negative predictions
		else:
			res_test_bg = evaluateRegularMC(net, test_loader_bg, criterion, out_dirc = output_dir+"/Temp_Data/Stored_Values", getPAttn = False,#genPAttn,
											storePAttn = argSpace.storeInterCNN, getCNN = getCNNout,
											storeCNNout = argSpace.storeInterCNN, getSeqs = getSequences)
			motif_dir_neg,numNegExamples = motif_analysis(res_test_bg, output_dir, argSpace, for_background = True) #for_background doesn't matter in this case since num_labels are greater than 2
		
		
			
		
		
else: #this is when we want to skip motif analysis (if its already done) we will need motif directories for downstream analyses
	motif_dir = output_dir + '/Motif_Analysis'
	if argSpace.intBackground != None:
		motif_dir_neg = output_dir + '/Motif_Analysis_Negative'
	
	##########################################################################
####################################################################################

####################################################################################
#########------------Attention Probabilities Analysis-----------------##############

#if argSpace.attnFigs:
	#Attn_dir = output_dir + '/Attention_Figures'

	#if not os.path.exists(Attn_dir):
	    #os.makedirs(Attn_dir)
	
	#plt.close('all')
	#plt.rcParams["figure.figsize"] = (9,5)
	
	#count = 0
	#for k in range(0,len(Prob_Attention_All)): #going through all batches
		#if count == argSpace.intSeqLimit:
			#break
		

		#PAttn = Prob_Attention_All[k]
		#if argSpace.storeInterCNN:
			#with open(PAttn,'rb') as f:
				#PAttn = pickle.load(f)
		
		
		#per_batch_labelPreds = res_test[4][k]
		#if num_labels == 2:
			#tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)] #pos classified examples > 0.6
		#else:
			#tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
		##tp_indices = tp_indices[:10] #get top 5 from each batch
		
		#feat_size = PAttn.shape[1]
		
		##ex = 165#7
		##plt.imshow(PAttn[ex,:,99*i:99*(i+1)])
		#for ex in tp_indices:
			#count += 1
			#if count == argSpace.intSeqLimit:
				#break
			#plt.close('all')
			#fig, ax = plt.subplots(nrows=2, ncols=4)
		
			#for i in range(0,8):
				#plt.subplot(2,4,i+1)
				#attn_mat = PAttn[ex,:,feat_size*i:feat_size*(i+1)]
				#plt.imshow(attn_mat)
				#max_loc = np.unravel_index(attn_mat.argmax(), attn_mat.shape)
				#plt.title('Single Head #'+str(i)+' Max Loc: '+str(max_loc),size=6)
				#plt.grid(False)
			
			##plt.clf()
			##print('Done for: ',str(ex))
			
			#plt.savefig(Attn_dir+'/'+'Batch-'+str(k)+'_PosExample-'+str(ex)+'_AttenMatrices.pdf')
		
		#print("Done for batch: ",k)
		#plt.close('all')
#######################################################################################

########################################################################################
################------------------Interactions Analysis-----------------################
#-----------Calculating Positive and Negative population------------#
per_batch_labelPreds = res_test[4]
#print(lohahi)

#	batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
#	batch_preds = per_batch_labelPreds['preds']                                                                                                                                                                           
#	for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
#		ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
#		ex_preds = batch_preds[e]                                                                                                                                                                                   
#		ex_preds = np.asarray([i>=argSpace.precisionLimit for i in ex_preds]).astype(int)                                                                                                                                               
#		tp_indices.append(e)

numPosExamples = 0
if argSpace.useAll == False and argSpace.numLabels != 2:
	bg_indices_multiLabel = []

for j in range(0,len(per_batch_labelPreds)):
	if numPosExamples == argSpace.intSeqLimit:
			break
	
	batch_values = per_batch_labelPreds[j]
	if num_labels != 2:
		batch_preds = batch_values['preds']
		batch_values = batch_values['labels']
	for k in range(0,len(batch_values)):
		if num_labels == 2:
			if (batch_values[k][0]==1 and batch_values[k][1]>pos_score_cutoff):
				numPosExamples += 1
		else:
			if argSpace.useAll == True:
				numPosExamples += 1
			else:
				ex_labels = batch_values[k].astype(int)                                                                                                                                                                     
				ex_preds = batch_preds[k]                                                                                                                                                                                   
				ex_preds = np.asarray([i>=0.5 for i in ex_preds]).astype(int)     
				prec = metrics.precision_score(ex_labels,ex_preds)                                                                                                                                           
				if prec >= argSpace.precisionLimit:
					numPosExamples += 1
					bg_indices_multiLabel.append([j*batchSize + k, prec])
		if numPosExamples == argSpace.intSeqLimit:
			break


if argSpace.useAll == False and argSpace.numLabels != 2:
	bg_indices_multiLabel = np.asarray(bg_indices_multiLabel)[:,0].astype(int)

numNegExamples = 0

for j in range(0,len(per_batch_labelPreds)):
	if numNegExamples == argSpace.intSeqLimit:
			break
	
	batch_values = per_batch_labelPreds[j]
	if num_labels != 2:
		batch_values = batch_values['labels']
	for k in range(0,len(batch_values)):
		if num_labels == 2:
			if argSpace.intBackground == 'negative':
				if (batch_values[k][0]==0 and batch_values[k][1]<(1-pos_score_cutoff)):
					numNegExamples += 1
			elif argSpace.intBackground == 'shuffle':
				numNegExamples += 1
			
		else:
			numNegExamples += 1
		if numNegExamples == argSpace.intSeqLimit:
			break

if argSpace.useAll == False and argSpace.numLabels != 2:
	numNegExamples = len(bg_indices_multiLabel)

print('Positive and Negative Population: ',	numPosExamples, numNegExamples)		
#-------------------------------------------------------------------#
def get_filters_in_individual_seq(sdata):
	header,num_filters,filter_data_dict,CNNfirstpool = sdata
	s_info_dict = {}
	for j in range(0,num_filters):
		filter_data = filter_data_dict['filter'+str(j)] #np.loadtxt(motif_dir+'/filter'+str(j)+'_logo.fa',dtype=str)
		for k in range(0,len(filter_data),2):
			hdr = filter_data[k].split('_')[0]
			if hdr == header:
				#print(hdr,header)
				pos = int(filter_data[k].split('_')[-2])
				act_val = float(filter_data[k].split('_')[-1])
				pooled_pos = int(pos/CNNfirstpool)
				key = pos#pooled_pos #we are no longer dealing with attention so lets use the actual position of the filter activation instead
				#key = 'filter'+str(j)
				if key not in s_info_dict:
					#s_info_dict[key] = [(pos,act_val)]
					s_info_dict[key] = [('filter'+str(j),act_val)]
				else:
					if 'filter'+str(j) not in s_info_dict[key][0]:
					#if pos not in s_info_dict[key][0]:
						if act_val > s_info_dict[key][0][1]:
							#s_info_dict[key].append(('filter'+str(j),act_val))
							#s_info_dict[key] = [(pos,act_val)]#
							s_info_dict[key] = [('filter'+str(j),act_val)]
	#print({header: s_info_dict},hdr,header)
	
	return {header: s_info_dict}
						
def get_filters_in_seq_dict(all_seqs,motif_dir,num_filters,CNNfirstpool,numWorkers=1):
	filter_data_dict = {}
	for i in range(0,num_filters):
		filter_data = np.loadtxt(motif_dir+'/filter'+str(i)+'_logo.fa',dtype=str)
		filter_data_dict['filter'+str(i)] = filter_data
	
	seq_info_dict = {}
	
	sdata = []
	for i in range(0,all_seqs.shape[0]):
		header = all_seqs[i][0]
		sdata.append([header,num_filters,filter_data_dict,CNNfirstpool])
	#count = 0
	with Pool(processes = numWorkers) as pool:
		result = pool.map(get_filters_in_individual_seq,sdata,chunksize=1)
		#pdb.set_trace()
		#count += 1
		#if count %10 == 0:
		#	print(count)
		for subdict in result:
			seq_info_dict.update(subdict)

	return seq_info_dict
	
	
#############----------DFIM based interactions---------------#############

#import shap
from deeplift.visualization import viz_sequence
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

from Bio.SeqUtils import GC
from sklearn.preprocessing import normalize



def evaluateRegularBatch(net, batch, criterion):
    
	running_loss = 0.0
	valid_auc = []
    
	net.eval()
    
	with torch.no_grad():
		
		headers, seqs, data, target = batch
		data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
		# Model computations
		outputs = net(data)
		
		loss = criterion(outputs, target)
		#loss = F.binary_cross_entropy(outputs, target)
		
		softmax = torch.nn.Softmax(dim=1)
		
		
		labels=target.cpu().numpy()
		
		pred = softmax(outputs)


		pred=pred.cpu().detach().numpy()
		
		label_pred = np.column_stack((labels,pred[:,1]))
		#per_batch_labelPreds[batch_idx] = label_pred
		#roc = np.row_stack((roc,label_pred))
		
		#print(pred)
		try:
			valid_auc.append(metrics.roc_auc_score(labels, pred[:,1]))
		except:
			valid_auc.append(0.0)
		
		
		running_loss += loss.item()
		
		headers_seqs = np.column_stack((headers,seqs))

            
           
        
	return running_loss,valid_auc,label_pred,headers_seqs
	
	
	
	
def evaluateRegularBatchMC(net, batch, criterion):
    
	running_loss = 0.0
	valid_auc = []
	per_batch_labelPreds = {}
	net.eval()
    
	with torch.no_grad():	
		headers, seqs, data, target = batch
		data, target = data.to(device,dtype=torch.float), target.to(device, dtype=torch.float)
		
		outputs = net(data)
		
		loss = criterion(outputs, target)
		
		labels=target.cpu().numpy()
		
		#softmax = torch.nn.Softmax(dim=0) #along columns
		#pred = softmax(outputs)
		
		sigmoid = torch.nn.Sigmoid()
		pred = sigmoid(outputs)
		
		pred = pred.cpu().detach().numpy()
		
		label_pred = {'labels':labels,'preds':pred}
		
		per_batch_labelPreds = label_pred
	
		headers_seqs = np.column_stack((headers,seqs))
            

            
		running_loss += loss.item()
        
	return running_loss,valid_auc,per_batch_labelPreds,headers_seqs
	
	

def get_seq_index(source_seq):
	nc_dict = {'A':0,'C':1,'G':2,'T':3}
	return [nc_dict[nc] for nc in source_seq]

#process_motif(seq,srcPos,seq_GC)
#def process_motif(seq,source,seq_GC)
def process_motif(seq,srcPos,fltrSize,seq_GC): 
	source_seq = seq[srcPos:srcPos+fltrSize]
	source_seq_ind = get_seq_index(source_seq)
	
	prob_dist = [(1-seq_GC)/2, seq_GC/2, seq_GC/2, (1-seq_GC)/2] # prob. dist for [A,C,G,T]
	
	res = np.random.choice(['A','C','G','T'], fltrSize, p=prob_dist)
		
	return ''.join(res),source_seq,source_seq_ind
		
def one_hot_encode(seq):
	mapping = dict(zip("ACGT", range(4)))    
	seq2 = [mapping[i] for i in seq]
	return np.eye(4)[seq2].T.astype(np.long)	

def generate_reference(seqLen, seq_GC=0.46):

	prob_dist = [(1-seq_GC)/2, seq_GC/2, seq_GC/2, (1-seq_GC)/2] # prob. dist for [A,C,G,T]
	
	res = np.random.choice(['A','C','G','T'], seqLen, p=prob_dist)
	
	return res
	
#net = Generalized_Net(params, genPAttn, wvmodel = modelwv, numClasses = num_labels).to(device) 
#try:    
	#checkpoint = torch.load(saved_model_dir+'/model')
	#net.load_state_dict(checkpoint['model_state_dict'])
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	#epoch = checkpoint['epoch']
	#loss = checkpoint['loss']
#except:
	#print("No pre-trained model found! Please run with --mode set to train.")
  



Intr_dir = output_dir + '/Interactions'

if not os.path.exists(Intr_dir):
	os.makedirs(Intr_dir)



#metadata = pd.read_csv('/s/jawar/h/nobackup/fahad/Human_Chromatin/Kundaji_DFIM_Data/dfim/data/embedded_motif_ours/sim_metadata.txt',sep='\t')



net = Generalized_Net(params, genPAttn, wvmodel = modelwv, numClasses = num_labels)
try:    
	checkpoint = torch.load(saved_model_dir+'/model')
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
except:
	print("No pre-trained model found! Please run with --mode set to train.")

model = net.to(device)


model.eval()

torch.backends.cudnn.enabled=False
	
#dl = IntegratedGradients(model)#DeepLiftShap(model)#IntegratedGradients(model)#DeepLift(model)

#final_res = [['batch','example','source','target','FIS','FISbg']]

#from https://github.com/pytorch/captum/issues/171
def model_wrapper(inputs, targets, TPs):
	#pdb.set_trace()
	#targets, TPs = add_args
	output = model(inputs)
	output = output[:,TPs]
	targets = targets[:,TPs]
    #print(inputs.shape, output.shape, targets.shape)
    # element-wise multiply outputs with one-hot encoded targets 
    # and compute sum of each row
    # This sums the prediction for all markers which exist in the cell
    #rx = output * targets
    #print(rx)
	return torch.sum(output * targets, dim=0)
    

#use the following
if num_labels == 2:
	dl = IntegratedGradients(model)
else:
	dl = IntegratedGradients(model_wrapper)
#attributions = dl.attribute(test_points,baseline,additional_forward_args=target[i].unsqueeze(dim=0))

######-------------------------Some Notes---------------------------############
#1. For a single position, I am selecting filter with the highest activation
#   This is different than SATORI since there I considered all filters.
#   I am doing this to reduce the overhead while calculating all interactions

################################################################################
from datetime import datetime
startTime = datetime.now()

#print(lohahi)
for_background = False
num_filters = params['CNN_filters']
CNNfirstpool = params['CNN_poolsize'] 
CNNfiltersize = params['CNN_filtersize']
sequence_len = len(res_test[-1][0][1])
#GC_content of the train set sequences
GC_content = GC(''.join(train_loader.dataset.df_seq_final['sequence'][train_indices].values))/100 #0.46 #argSpace.gcContent

Filter_Intr_Keys = {}
count_index = 0
for i in range(0,num_filters):
	for j in range(0,num_filters):
		if i == j:
			continue
		intr = 'filter'+str(i)+'<-->'+'filter'+str(j)
		rev_intr = 'filter'+str(j)+'<-->'+'filter'+str(i)
		
		if intr not in Filter_Intr_Keys and rev_intr not in Filter_Intr_Keys:
			Filter_Intr_Keys[intr] = count_index
			count_index += 1



Filter_Intr_Attn = np.ones((len(Filter_Intr_Keys),numPosExamples))*-1
Filter_Intr_Pos = np.ones((len(Filter_Intr_Keys),numPosExamples)).astype(int)*-1

col_index = 0
tp_pos_dict = {}
for batch_idx, batch in enumerate(test_loader):
	
	if col_index >= numPosExamples:
			break
	
	if num_labels == 2:
		res_test = evaluateRegularBatch(net,batch,criterion)
	else:
		res_test = evaluateRegularBatchMC(net,batch,criterion)
	
	Seqs = res_test[-1]
	per_batch_labelPreds = res_test[-2]
	
	
	headers,seqs,datapoints,target = batch

	if num_labels == 2:
		if for_background and argSpace.intBackground != None:
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
		else:
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
	else:
		#tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
		if argSpace.useAll == True:
			tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
		else:
			tp_indices=[] 
			TPs = {}                                                                                                                                                                                   
			batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
			batch_preds = per_batch_labelPreds['preds']                                                                                                                                                                           
			for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
				ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
				ex_preds = batch_preds[e]                                                                                                                                                                                   
				ex_preds = np.asarray([i>=0.5 for i in ex_preds]).astype(int)    
				prec = metrics.precision_score(ex_labels,ex_preds)         
				if prec >= argSpace.precisionLimit:   
					TP = [i for i in range(0,ex_labels.shape[0]) if (ex_labels[i]==1 and ex_preds[i]==1)] #these are going to be used in calculating attributes: average accross only those columns which are true positives                                                                                                                               
					tp_indices.append(e)
					tp_pos_dict[headers[e]] = TP
					TPs[e] = TP
	print(len(tp_indices))
	#pos_ex_ind = [(i,res_test[2][i][1],target[i]) for i in range(0,res_test[2].shape[0]) if res_test[2][i][1]>0.5 and target[i]==1] 
	
	Seqs_tp = Seqs[tp_indices]
	
	seq_info_dict = get_filters_in_seq_dict(Seqs_tp,motif_dir,num_filters,CNNfirstpool,numWorkers=argSpace.numWorkers)

	
	if num_labels == 2:
		datapoints, target = datapoints.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
	else:
		datapoints, target = datapoints.to(device,dtype=torch.float), target.to(device,dtype=torch.float)

	for ind in range(0, len(tp_indices)):#i_pos in range(0,len(pos_ex_ind)):
		i = tp_indices[ind]
		#i = pos_ex_ind[i_pos][0]
		#baseline = torch.zeros(1, 4, 600)#datapoints[:100]#torch.zeros(3, 4, 600).to(device)
		test_points = datapoints[i]#i
		
		##this baseline assumes all 0s
		#baseline = test_points*0#
		
		##this baseline should be used with deepLIFTSHAP
		#baseline = datapoints[:100]#test_points*0
		
		#-----------------------------------------------------------------------------------#
		#---This can be put outside the loop since baseline won't change for fixed GC (at least this case, simulated)---#
		##Kundaje et al mentioned that they used baseline based on the background GC content
		baseline_seq = generate_reference(test_points.shape[-1], seq_GC = GC_content) #0.46 was for the simulated data
		baseline = one_hot_encode(baseline_seq) 
		
		baseline = torch.Tensor(baseline).to(device, dtype=torch.float)
		#------------------------------------------------------------------------------------#
		
		test_points = test_points.unsqueeze(dim=0)
		baseline = baseline.unsqueeze(dim=0)
		
		#suggestion from captum developer
		
		
		if num_labels == 2:
			attributions = dl.attribute(test_points,baseline,target=target[i])
		else:
			attributions = dl.attribute(test_points,baseline,additional_forward_args=(target[i].unsqueeze(dim=0),TPs[i]))
		
		#pdb.set_trace()
		res = attributions.squeeze(dim=0).cpu().detach().numpy()
		#res = normalize(res, norm='l2')
		
		
		
		#res_max = np.max(res,axis=0) don't need the max value; instead we need to pick the corresponding nucleotide value 
		
		#--to visualize and save the attribution across input--#
		#viz_sequence.plot_weights(res,subticks_frequency=50,figsize=(20,4))
		#plt.savefig('somefile.png')
		#------------------------------------------------------#
		
		
			
		
		#-------after mutating source--------#
		header = headers[i]
		seq = seqs[i]
		
		
		seq_GC = GC(seq)/100.0 #GC is from Bio.Utils #this is used to mutate source motif (GC content of the current input seq)
		
		
		
		

		pos_and_filters = seq_info_dict[header]
		
		
		tpnt_tuple = torch.tensor([]).to(device)
		bsln_tuple = torch.tensor([]).to(device)
		if num_labels==2:
			trgt_tuple = torch.tensor([]).to(device,dtype=torch.long)
		else:
			trgt_tuple = torch.tensor([]).to(device,dtype=torch.float)
		#intr_tuple = []
		#pos_tuple = []
		#trgPos_tuple = []
		#trgInd_tuple = []
		count_part = 0
		srcPos_info = {} #keeps track of interactions info
		count_srcInd = 0
		covered_intr_tuple = []
		for srcPos in pos_and_filters:

			#moving source part here
			source_mutated, source_seq, source_ind = process_motif(seq,srcPos,CNNfiltersize,seq_GC)
			seq_mutated = seq[:srcPos] + source_mutated + seq[srcPos+CNNfiltersize:]	
			mut_data = one_hot_encode(seq_mutated)
			test_points_mut = torch.Tensor(mut_data).to(device, dtype=torch.float)

			#-----------------------------------------------------------------------------------#
			##Kundaje et al mentioned that they used baseline based on the background GC content
			baseline_mut_seq = generate_reference(test_points_mut.shape[-1], seq_GC = GC_content) #0.46 was for the simulated data
			baseline_mut = one_hot_encode(baseline_mut_seq) 
		
			baseline_mut = torch.Tensor(baseline_mut).to(device, dtype=torch.float)
			#------------------------------------------------------------------------------------#

			test_points_mut = test_points_mut.unsqueeze(dim=0)
			baseline_mut = baseline_mut.unsqueeze(dim=0)

			tpnt_tuple = torch.cat((tpnt_tuple,test_points_mut),dim=0)
			bsln_tuple = torch.cat((bsln_tuple,baseline_mut),dim=0)
			trgt_tuple = torch.cat((trgt_tuple,target[i].unsqueeze(dim=0)))
			
			#print(pos_and_filters[srcPos])
			intr_tuple = []
			pos_tuple = []
			trgPos_tuple = []
			trgInd_tuple = []
			for trgPos in pos_and_filters:
				if abs(srcPos-trgPos) < CNNfirstpool: #Similar to what I do in self-attention; discard attention values at row == column (this is after pooling, where poolsize should be used)
					continue
				srcFilter = pos_and_filters[srcPos][0][0]
				trgFilter = pos_and_filters[trgPos][0][0]
				
				intr = srcFilter + '<-->' + trgFilter
				if intr not in Filter_Intr_Keys:
					intr = trgFilter + '<-->' + srcFilter
					
				if intr in covered_intr_tuple: #if already covered
					continue
				
				
				if srcFilter == trgFilter: #they can't be the same for interaction
					continue
				
				intr_tuple.append(intr)

				#for tracking
				covered_intr_tuple.append(intr)

				pos_diff = abs(srcPos-trgPos)
				pos_tuple.append(pos_diff)
				
				target_mutated, target_seq, target_ind = process_motif(seq,trgPos,CNNfiltersize,seq_GC) ##not needed but we need some of the info later
				
				trgPos_tuple.append(trgPos)
				trgInd_tuple.append(target_ind)
				
				count_part += 1
			srcPos_info[count_srcInd] = [intr_tuple,pos_tuple,trgPos_tuple,trgInd_tuple]
			count_srcInd += 1
		print('# filter comparisons: ',count_part)
		
		for bsize in range(0,len(srcPos_info),argSpace.attrBatchSize): #trying 96 at a time, otherwise it runs out of CUDA memory (too many tensors to test). 172 also fails. I guess 120 will work
			start = bsize
			end = min([bsize+argSpace.attrBatchSize,len(srcPos_info)])
			
			if num_labels == 2:
				attributions_mut = dl.attribute(tpnt_tuple[start:end],bsln_tuple[start:end],target=trgt_tuple[start:end])
			else:
				attributions_mut = dl.attribute(tpnt_tuple[start:end],bsln_tuple[start:end],additional_forward_args=(trgt_tuple[start:end],TPs[i]))
			
			count_sbsize = 0
			for sbsize in range(start,end):

				intr_tuple_sub = srcPos_info[sbsize][0]
				pos_tuple_sub = srcPos_info[sbsize][1]
				trgPos_tuple_sub = srcPos_info[sbsize][2]
				trgInd_tuple_sub = srcPos_info[sbsize][3]

				res_mut = attributions_mut[count_sbsize,:,:].squeeze(dim=0).cpu().detach().numpy()
				count_sbsize += 1
				for subsize in range(0,len(intr_tuple_sub)):
					
					#res_mut = normalize(res_mut, norm='l2')
			
					#res_mut_max = np.max(res_mut,axis=0)
					
					trgPos = trgPos_tuple_sub[subsize]
					target_ind = trgInd_tuple_sub[subsize]
					
					C_orig = res[:,trgPos:trgPos+CNNfiltersize]
					C_orig = np.sum([C_orig[target_ind[i],i] for i in range(0,len(target_ind))])
					
					C_mut = res_mut[:,trgPos:trgPos+CNNfiltersize]
					C_mut = np.sum([C_mut[target_ind[i],i] for i in range(0,len(target_ind))])
					
					FIS = C_orig - C_mut
					
					
					intr = intr_tuple_sub[subsize]
					
					row_index = Filter_Intr_Keys[intr]
					
					Filter_Intr_Attn[row_index][col_index] = abs(FIS) #ideally we shouldn't take absolute but to compare it to SATORI, we need the abs
					
					
					
					Filter_Intr_Pos[row_index][col_index] = pos_tuple_sub[subsize]

		
		col_index += 1
	

		print('batch: ',batch_idx,'example: ',i)
		
		if col_index >= numPosExamples:
			break


with open(Intr_dir+'/interaction_keys_dict.pckl','wb') as f:
	pickle.dump(Filter_Intr_Keys,f)
		
	
with open(Intr_dir+'/main_results_raw.pckl','wb') as f:
	pickle.dump([Filter_Intr_Attn,Filter_Intr_Pos],f)




main_time = datetime.now() - startTime







#print(lohahi)

if argSpace.intBackground == 'shuffle':
	bg_prefix = get_shuffled_background_v3(test_loader,argSpace) #get_shuffled_background_v2(test_loader, params['CNN_filters'], params['CNN_filtersize'], argSpace)  the version 2 doesn't work that well (average AUC 0.50 and 0 motifs in background)
	#data_bg = ProcessedDataVersion2(bg_prefix,num_labels)
	if argSpace.deskLoad == False:
		data_bg = ProcessedDataVersion2(bg_prefix,num_labels) #get all data
	else:
		data_bg = ProcessedDataVersion2A(bg_prefix,num_labels)
			
	test_loader_bg = DataLoader(data_bg,batch_size=batchSize,num_workers=argSpace.numWorkers)	
	res_test_bg = evaluateRegularMC(net, test_loader_bg, criterion, out_dirc = output_dir+"/Temp_Data/Stored_Values", getPAttn = False,#genPAttn,
											storePAttn = argSpace.storeInterCNN, getCNN = getCNNout,
											storeCNNout = argSpace.storeInterCNN, getSeqs = getSequences)



#############---------For background--------------##########

startTime = datetime.now()

if num_labels == 2:
	dl = IntegratedGradients(model)
else:
	dl = IntegratedGradients(model_wrapper)


for_background = True

if argSpace.intBackground == 'shuffle':
	test_loader = test_loader_bg

Filter_Intr_Attn_Bg = np.ones((len(Filter_Intr_Keys),numNegExamples))*-1
Filter_Intr_Pos_Bg = np.ones((len(Filter_Intr_Keys),numNegExamples)).astype(int)*-1

col_index = 0
for batch_idx, batch in enumerate(test_loader):
	
	if col_index >= numNegExamples:
			break
	
	if num_labels == 2:
		res_test = evaluateRegularBatch(net,batch,criterion)
	else:
		res_test = evaluateRegularBatchMC(net,batch,criterion)
	
	Seqs = res_test[-1]
	per_batch_labelPreds = res_test[-2]
	
	headers,seqs,datapoints,target = batch
	
	if num_labels == 2:
		if for_background and argSpace.intBackground != None:
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
		else:
			tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
	else:
		if argSpace.useAll==True:
			tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
		else:
			#tp_indices,TPs = get_TP_info(headers,tp_pos_dict)
			tp_indices = []
			TPs = {}
			for h_i in range(0,len(headers)):
				header = headers[h_i]
				if header in tp_pos_dict:
					tp_indices.append(h_i)
					TPs[h_i] = tp_pos_dict[header]
	
	#pos_ex_ind = [(i,res_test[2][i][1],target[i]) for i in range(0,res_test[2].shape[0]) if res_test[2][i][1]>0.5 and target[i]==1] 
	
	Seqs_tp = Seqs[tp_indices]
	
	seq_info_dict = get_filters_in_seq_dict(Seqs_tp,motif_dir_neg,num_filters,CNNfirstpool,numWorkers=argSpace.numWorkers)

	
	if num_labels == 2:
		datapoints, target = datapoints.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
	else:
		datapoints, target = datapoints.to(device,dtype=torch.float), target.to(device,dtype=torch.float)

	for ind in range(0, len(tp_indices)):#i_pos in range(0,len(pos_ex_ind)):
		i = tp_indices[ind]
		#i = pos_ex_ind[i_pos][0]
		#baseline = torch.zeros(1, 4, 600)#datapoints[:100]#torch.zeros(3, 4, 600).to(device)
		test_points = datapoints[i]#i
		
		##this baseline assumes all 0s
		baseline = test_points*0#
		
		##this baseline should be used with deepLIFTSHAP
		#baseline = datapoints[:100]#test_points*0
		
		#-----------------------------------------------------------------------------------#
		#---This can be put outside the loop since baseline won't change for fixed GC (at least this case, simulated)---#
		##Kundaje et al mentioned that they used baseline based on the background GC content
		baseline_seq = generate_reference(test_points.shape[-1], seq_GC = GC_content) #0.46 was for the simulated data
		baseline = one_hot_encode(baseline_seq) 
		
		baseline = torch.Tensor(baseline).to(device, dtype=torch.float)
		#------------------------------------------------------------------------------------#
		
		test_points = test_points.unsqueeze(dim=0)
		baseline = baseline.unsqueeze(dim=0)
		
		#suggestion from captum developer
		
		
		if num_labels == 2:
			attributions = dl.attribute(test_points,baseline,target=target[i])
		else:
			attributions = dl.attribute(test_points,baseline,additional_forward_args=(target[i].unsqueeze(dim=0),TPs[i]))
		
		res = attributions.squeeze(dim=0).cpu().detach().numpy()
		#res = normalize(res, norm='l2') #normalization maybe is causing problems
		
		
		
		#res_max = np.max(res,axis=0) don't need the max value; instead we need to pick the corresponding nucleotide value 
		
		#--to visualize and save the attribution across input--#
		#viz_sequence.plot_weights(res,subticks_frequency=50,figsize=(20,4))
		#plt.savefig('somefile.png')
		#------------------------------------------------------#
		
		
			
		
		#-------after mutating source--------#
		header = headers[i]
		seq = seqs[i]
		
		
		seq_GC = GC(seq)/100.0 #GC is from Bio.Utils #this is used to mutate source motif (GC content of the current input seq)
		
		
		pos_and_filters = seq_info_dict[header]
		
		intr_tuple = []
		tpnt_tuple = torch.tensor([]).to(device)
		bsln_tuple = torch.tensor([]).to(device)

		if num_labels==2:
			trgt_tuple = torch.tensor([]).to(device,dtype=torch.long)
		else:
			trgt_tuple = torch.tensor([]).to(device,dtype=torch.float)
		#intr_tuple = []
		#pos_tuple = []
		#trgPos_tuple = []
		#trgInd_tuple = []
		count_part = 0
		srcPos_info = {} #keeps track of interactions info
		count_srcInd = 0
		covered_intr_tuple = []
		for srcPos in pos_and_filters:

			#moving source part here
			source_mutated, source_seq, source_ind = process_motif(seq,srcPos,CNNfiltersize,seq_GC)
			seq_mutated = seq[:srcPos] + source_mutated + seq[srcPos+CNNfiltersize:]	
			mut_data = one_hot_encode(seq_mutated)
			test_points_mut = torch.Tensor(mut_data).to(device, dtype=torch.float)

			#-----------------------------------------------------------------------------------#
			##Kundaje et al mentioned that they used baseline based on the background GC content
			baseline_mut_seq = generate_reference(test_points_mut.shape[-1], seq_GC = GC_content) #0.46 was for the simulated data
			baseline_mut = one_hot_encode(baseline_mut_seq) 
		
			baseline_mut = torch.Tensor(baseline_mut).to(device, dtype=torch.float)
			#------------------------------------------------------------------------------------#

			test_points_mut = test_points_mut.unsqueeze(dim=0)
			baseline_mut = baseline_mut.unsqueeze(dim=0)

			tpnt_tuple = torch.cat((tpnt_tuple,test_points_mut),dim=0)
			bsln_tuple = torch.cat((bsln_tuple,baseline_mut),dim=0)
			trgt_tuple = torch.cat((trgt_tuple,target[i].unsqueeze(dim=0)))


			#print(pos_and_filters[srcPos])
			intr_tuple = []
			pos_tuple = []
			trgPos_tuple = []
			trgInd_tuple = []
			for trgPos in pos_and_filters:
				if abs(srcPos-trgPos) < CNNfirstpool: #Similar to what I do in self-attention; discard attention values at row == column (this is after pooling, where poolsize should be used)
					continue
				srcFilter = pos_and_filters[srcPos][0][0]
				trgFilter = pos_and_filters[trgPos][0][0]
				
				intr = srcFilter + '<-->' + trgFilter
				if intr not in Filter_Intr_Keys:
					intr = trgFilter + '<-->' + srcFilter
					
				if intr in covered_intr_tuple: #if already covered
					continue
				
				
				if srcFilter == trgFilter: #they can't be the same for interaction
					continue
				
				intr_tuple.append(intr)

                #for tracking
				covered_intr_tuple.append(intr)
				
				pos_diff = abs(srcPos-trgPos)
				pos_tuple.append(pos_diff)
				
				target_mutated, target_seq, target_ind = process_motif(seq,trgPos,CNNfiltersize,seq_GC) ##not needed but we need some of the info later
				
				trgPos_tuple.append(trgPos)
				trgInd_tuple.append(target_ind)

				count_part += 1
			srcPos_info[count_srcInd] = [intr_tuple,pos_tuple,trgPos_tuple,trgInd_tuple]
			count_srcInd += 1
		print('# filter comparisons: ',count_part)
		
		for bsize in range(0,len(srcPos_info),argSpace.attrBatchSize): #trying 96 at a time, otherwise it runs out of CUDA memory (too many tensors to test). 172 also fails. I guess 120 will work
			start = bsize
			end = min([bsize+argSpace.attrBatchSize,len(srcPos_info)])
			
			if num_labels == 2:
				attributions_mut = dl.attribute(tpnt_tuple[start:end],bsln_tuple[start:end],target=trgt_tuple[start:end])
			else:
				attributions_mut = dl.attribute(tpnt_tuple[start:end],bsln_tuple[start:end],additional_forward_args=(trgt_tuple[start:end],TPs[i]))
			
			count_sbsize = 0
			for sbsize in range(start,end):

				intr_tuple_sub = srcPos_info[sbsize][0]
				pos_tuple_sub = srcPos_info[sbsize][1]
				trgPos_tuple_sub = srcPos_info[sbsize][2]
				trgInd_tuple_sub = srcPos_info[sbsize][3]

				res_mut = attributions_mut[count_sbsize,:,:].squeeze(dim=0).cpu().detach().numpy()
				count_sbsize += 1
				for subsize in range(0,len(intr_tuple_sub)):
					
					#res_mut = normalize(res_mut, norm='l2')
			
					#res_mut_max = np.max(res_mut,axis=0)
					
					trgPos = trgPos_tuple_sub[subsize]
					target_ind = trgInd_tuple_sub[subsize]
					
					C_orig = res[:,trgPos:trgPos+CNNfiltersize]
					C_orig = np.sum([C_orig[target_ind[i],i] for i in range(0,len(target_ind))])
					
					C_mut = res_mut[:,trgPos:trgPos+CNNfiltersize]
					C_mut = np.sum([C_mut[target_ind[i],i] for i in range(0,len(target_ind))])
					
					FIS = C_orig - C_mut
					
					
					intr = intr_tuple_sub[subsize]
					
					row_index = Filter_Intr_Keys[intr]

					Filter_Intr_Attn_Bg[row_index][col_index] = abs(FIS) #ideally we shouldn't take absolute but to compare it to SATORI, we need the abs
				
				
				
					Filter_Intr_Pos_Bg[row_index][col_index] = pos_tuple_sub[subsize]
	
				
						
		col_index += 1
	

		print('batch: ',batch_idx,'example: ',i)
		
		if col_index >= numNegExamples:
			break


with open(Intr_dir+'/background_results_raw.pckl','wb') as f:
	pickle.dump([Filter_Intr_Attn_Bg,Filter_Intr_Pos_Bg],f)



bg_time = datetime.now() - startTime




##############################################################################
#--------------------------motif interactions analysis-----------------------#
##############################################################################
tomtom_data = np.loadtxt(motif_dir+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
if argSpace.intBackground != None:
	tomtom_data_neg = np.loadtxt(motif_dir_neg+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')




resMain = Filter_Intr_Attn[Filter_Intr_Attn!=-1]                                                                                                                                               
resBg = Filter_Intr_Attn_Bg[Filter_Intr_Attn_Bg!=-1]
resMainHist = np.histogram(resMain,bins=20)
resBgHist = np.histogram(resBg,bins=20)
plt.plot(resMainHist[1][1:],resMainHist[0]/sum(resMainHist[0]),linestyle='--',marker='o',color='g',label='main')
plt.plot(resBgHist[1][1:],resBgHist[0]/sum(resBgHist[0]),linestyle='--',marker='x',color='r',label='background')

plt.legend(loc='best',fontsize=10)
plt.savefig(Intr_dir+'/normalized_Attn_scores_distributions.pdf')
plt.clf()

plt.hist(resMain,bins=20,color='g',label='main')
plt.hist(resBg,bins=20,color='r',alpha=0.5,label='background')
plt.legend(loc='best',fontsize=10)
plt.savefig(Intr_dir+'/Attn_scores_distributions.pdf')
plt.clf()


Bg_MaxMean = []
Main_MaxMean = []

for entry in Filter_Intr_Attn:
	try:
		Main_MaxMean.append([np.max(entry[entry!=-1]),np.mean(entry[entry!=-1])])
	except:
		continue
	
for entry in Filter_Intr_Attn_Bg:
	try:
		Bg_MaxMean.append([np.max(entry[entry!=-1]),np.mean(entry[entry!=-1])])
	except:
		continue
	
Bg_MaxMean = np.asarray(Bg_MaxMean)
Main_MaxMean = np.asarray(Main_MaxMean)

plt.hist(Main_MaxMean[:,0],bins=20,color='g',label='main')
plt.hist(Bg_MaxMean[:,0],bins=20,color='r',alpha=0.5,label='background')
plt.legend(loc='best',fontsize=10)
plt.savefig(Intr_dir+'/Attn_scores_distributions_MaxPerInteraction.pdf')
plt.clf()

plt.hist(Main_MaxMean[:,1],bins=20,color='g',label='main')
plt.hist(Bg_MaxMean[:,1],bins=20,color='r',alpha=0.5,label='background')
plt.legend(loc='best',fontsize=10)
plt.savefig(Intr_dir+'/Attn_scores_distributions_MeanPerInteraction.pdf')
plt.clf()






#dummy in this case, we are not dropping values based on the attnLimit
attnLimits = [0]#[argSpace.attnCutoff * i for i in range(1,11)] #save results for 10 different attention cutoff values (maximum per interaction) eg. [0.05, 0.10, 0.15, 0.20, 0.25, ...]

for attnLimit in attnLimits:
	pval_info = []#{}
	for i in range(0,Filter_Intr_Attn.shape[0]):                                                                                                                                                   
		pos_attn = Filter_Intr_Attn[i,:]                                                                                                                                                              
		pos_attn = pos_attn[pos_attn!=-1]#pos_attn[pos_attn>0.04] #pos_attn[pos_attn!=-1]                                                                                                                                                                   
		neg_attn = Filter_Intr_Attn_Bg[i,:]                                                                                                                                                          
		neg_attn = neg_attn[neg_attn!=-1]#neg_attn[neg_attn>0.04] #neg_attn[neg_attn!=-1] 
		
		
		num_pos = len(pos_attn)
		num_neg = len(neg_attn)
		
		if len(pos_attn) <= 1:# or len(neg_attn) <= 1:
			continue
		
		if len(neg_attn) <= 1: #if just 1 or 0 values in neg attn, get a vector with all values set to 0 (same length as pos_attn)
			neg_attn = np.asarray([0 for i in range(0,num_pos)])
		
		if np.max(pos_attn) < attnLimit: # 
			continue
		
		pos_posn = Filter_Intr_Pos[i,:]  
		#pos_posn_mean = pos_posn[pos_posn!=-1].mean()
		pos_posn_mean = pos_posn[np.argmax(Filter_Intr_Attn[i,:])] #just pick the max
		
		neg_posn = Filter_Intr_Pos_Bg[i,:]  
		#neg_posn_mean = neg_posn[neg_posn!=-1].mean()
		neg_posn_mean = neg_posn[np.argmax(Filter_Intr_Attn_Bg[i,:])] #just pick the max
		                                                                                                                                                                
                                                                                                                                                                              
		stats,pval = mannwhitneyu(pos_attn,neg_attn,alternative='greater')#ttest_ind(pos_d,neg_d)#mannwhitneyu(pos_d,neg_d,alternative='greater')                                                        
		pval_info.append([i, pos_posn_mean, neg_posn_mean,num_pos,num_neg, stats,pval])#pval_dict[i] = [i,stats,pval]                                                                                                                                                              
		#if i%100==0:                                                                                                                                                                               
		#	print('Done: ',i) 
	
	
	pval_info = np.asarray(pval_info)
	
	res_final = pval_info#[pval_info[:,-1]<0.01] #can be 0.05 or any other threshold #For now, lets take care of this in post processing (jupyter notebook)
	
	res_final_int = []                                                                                                                                                                             
                                                                                                                                                                     
	for i in range(0,res_final.shape[0]):                                                                                                                                                          
		#res_final_int.append([res_final[i][-1],Filter_Intr_Keys[int(res_final[i][0])]])                                                                                                           
		value = int(res_final[i][0])                                                                                                                                                               
		pval = res_final[i][-1]
		pp_mean = res_final[i][1]
		np_mean = res_final[i][2]  
		num_pos = res_final[i][3]
		num_neg = res_final[i][4]         
		stats = res_final[i][-2]                                                                                                                                                         
		for key in Filter_Intr_Keys:                                                                                                                                                               
			if Filter_Intr_Keys[key] == value:                                                                                                                                                     
				res_final_int.append([key,value,pp_mean,np_mean,num_pos,num_neg,stats,pval])  
	
	res_final_int = np.asarray(res_final_int) 
	qvals = multipletests(res_final_int[:,-1].astype(float), method='fdr_bh')[1] #res_final_int[:,1].astype(float)
	res_final_int = np.column_stack((res_final_int,qvals))
	
	final_interactions = [['filter_interaction','example_no','motif1','motif1_qval','motif2','motif2_qval','mean_distance','mean_distance_bg','num_obs','num_obs_bg','pval','adjusted_pval']]
	for entry in res_final_int:                                                                                                                                                                    
		f1,f2 = entry[0].split('<-->')                                                                                                                                                             
                                                                                                                                                                      
		m1_ind = np.argwhere(tomtom_data[:,0]==f1)                                                                                                                                                 
		m2_ind = np.argwhere(tomtom_data[:,0]==f2)                                                                                                                                                 
		#print(m1_ind,m2_ind)
		if len(m1_ind) == 0 or len(m2_ind) == 0:
			continue
		m1 = tomtom_data[m1_ind[0][0]][1]
		m2 = tomtom_data[m2_ind[0][0]][1]
		m1_pval = tomtom_data[m1_ind[0][0]][5]
		m2_pval = tomtom_data[m2_ind[0][0]][5]
		final_interactions.append([entry[0],entry[1],m1,m1_pval,m2,m2_pval,entry[2],entry[3],entry[4],entry[5],entry[-2],entry[-1]])
		#print(entry[-1],m1,m2,entry[0])
	
	
	np.savetxt(Intr_dir+'/interactions_summary_attnLimit-'+str(attnLimit)+'.txt',final_interactions,fmt='%s',delimiter='\t')
	
	with open(Intr_dir+'/processed_results_attnLimit-'+str(attnLimit)+'.pckl','wb') as f:
		pickle.dump([pval_info,res_final_int],f)
	
	print("Done for Attention Cutoff Value: ",str(attnLimit))


time_taken = [['main_loop','bg_loop','total']]
time_taken.append([main_time.total_seconds(),bg_time.total_seconds(),main_time.total_seconds()+bg_time.total_seconds()])

np.savetxt(Intr_dir+'/timing_stats.txt',time_taken,fmt='%s',delimiter='\t')


