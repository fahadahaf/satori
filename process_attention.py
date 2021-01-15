import numpy as np
import os

# get rid of word2vec related stuff for now (or keep it for future work?) #
import gensim
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from fastprogress import progress_bar
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from multiprocessing import Pool
from random import randint
from scipy.stats import mannwhitneyu
from sklearn import metrics
from statsmodels.stats.multitest import multipletests
from torch.backends import cudnn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations

#local imports
from datasets import DatasetLoadAll, DatasetLazyLoad
from extract_motifs import get_motif
from models import AttentionNet
from utils import get_params_dict, get_random_seq

from utils import get_params_dict, get_popsize_for_interactions, get_intr_filter_keys


def get_filters_in_individual_seq(sdata):
	header,num_filters,filter_data_dict,CNNfirstpool = sdata
	s_info_dict = {}
	for j in range(0,num_filters):
		filter_data = filter_data_dict['filter'+str(j)] #np.loadtxt(motif_dir+'/filter'+str(j)+'_logo.fa',dtype=str)
		for k in range(0,len(filter_data),2):
			hdr = filter_data[k].split('_')[0]
			if hdr == header:
				pos = int(filter_data[k].split('_')[-2]) #-2 because the format is header_num_pos_activation
				pooled_pos = int(pos/CNNfirstpool)
				key = pooled_pos#header+'_'+str(pooled_pos)
				if key not in s_info_dict:
					s_info_dict[key] = ['filter'+str(j)]
				else:
					if 'filter'+str(j) not in s_info_dict[key]:
						s_info_dict[key].append('filter'+str(j))
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


def score_individual_head(data):
	count,header,seq_inf_dict,k,ex,params,tomtom_data,attn_cutoff,sequence_len,CNNfirstpool,num_filters,motif_dir,feat_size,storeInterCNN,considerTopHit = data
	#print(k,ex)
	global Prob_Attention_All# = res_test[3]
	global Seqs# = res_test[6]
	global LabelPreds# = res_test[4]
	#global Filter_Intr_Attn
	#global Filter_Intr_Pos
	global Filter_Intr_Keys
	
	filter_Intr_Attn = np.ones(len(Filter_Intr_Keys))*-1
	filter_Intr_Pos = np.ones(len(Filter_Intr_Keys)).astype(int)*-1
	
	y_ind = count#(k*params['batch_size']) + ex
	
	PAttn = Prob_Attention_All[k]
	if storeInterCNN:
		with open(PAttn,'rb') as f:
			PAttn = pickle.load(f)
	
	#filter_intr_dict = {}
	
	attn_mat = PAttn[ex,:,:]
	
	attn_mat = np.asarray([attn_mat[:,feat_size*i:feat_size*(i+1)] for i in range(0,params['num_multiheads'])]) 
	attn_mat = np.max(attn_mat, axis=0) #out of the 8 attn matrices, get the max value at the corresponding positions
	
	for i in range(0, attn_mat.shape[0]):
		if i not in seq_inf_dict:
			continue
		for j in range(0, attn_mat.shape[1]):
			#pdb.set_trace()
			if j not in seq_inf_dict:
				continue
			if i==j:
				continue
			max_loc = [i,j]#attn_mat[i,j]
			
			pos_diff = CNNfirstpool * abs(max_loc[0]-max_loc[1])
			
			KeyA = i #seq_inf_dict already is for the current header and we just need to specify the Pooled position
			KeyB = j 
			
			attn_val = attn_mat[i,j]
			
			all_filters_posA = seq_inf_dict[KeyA]
			all_filters_posB = seq_inf_dict[KeyB]
			
			for keyA in all_filters_posA:
				for keyB in all_filters_posB:
					if keyA == keyB:
						continue
					intr = keyA+'<-->'+keyB
					rev_intr = keyB+'<-->'+keyA
					if intr in Filter_Intr_Keys:
						x_ind = Filter_Intr_Keys[intr]
					elif rev_intr in Filter_Intr_Keys:
						x_ind = Filter_Intr_Keys[rev_intr]
							
					if attn_val > filter_Intr_Attn[x_ind]:#[y_ind]:
						filter_Intr_Attn[x_ind] = attn_val #[y_ind] = attn_val
					filter_Intr_Pos[x_ind] = pos_diff#[y_ind] = pos_diff
						
				
	return y_ind,filter_Intr_Attn,filter_Intr_Pos
			
			

def score_individual_head_bg(data):
	
	count,header,seq_inf_dict,k,ex,params,tomtom_data,attn_cutoff,sequence_len,CNNfirstpool,num_filters,motif_dir,feat_size,storeInterCNN,considerTopHit = data
	
	global Prob_Attention_All_neg# = res_test[3]
	global Seqs_neg# = res_test[6]
	global LabelPreds_neg# = res_test[4]
	#global Filter_Intr_Attn
	#global Filter_Intr_Pos
	global Filter_Intr_Keys
	
	filter_Intr_Attn = np.ones(len(Filter_Intr_Keys))*-1
	filter_Intr_Pos = np.ones(len(Filter_Intr_Keys)).astype(int)*-1
	
	y_ind = count#(k*params['batch_size']) + ex
	
	PAttn = Prob_Attention_All_neg[k]
	if storeInterCNN:
		with open(PAttn,'rb') as f:
			PAttn = pickle.load(f)
	
	attn_mat = PAttn[ex,:,:]
	
	attn_mat = np.asarray([attn_mat[:,feat_size*i:feat_size*(i+1)] for i in range(0,params['num_multiheads'])]) 
	attn_mat = np.max(attn_mat, axis=0) #out of the 8 attn matrices, get the max value at the corresponding positions
	
	for i in range(0, attn_mat.shape[0]):
		if i not in seq_inf_dict:
			continue
		for j in range(0, attn_mat.shape[1]):
			#pdb.set_trace()
			if j not in seq_inf_dict:
				continue
			if i==j:
				continue
			max_loc = [i,j]#attn_mat[i,j]
			
			pos_diff = CNNfirstpool * abs(max_loc[0]-max_loc[1])
			
			KeyA = i #seq_inf_dict already is for the current header and we just need to specify the Pooled position
			KeyB = j 
			
			attn_val = attn_mat[i,j]
			
			all_filters_posA = seq_inf_dict[KeyA]
			all_filters_posB = seq_inf_dict[KeyB]
			
			for keyA in all_filters_posA:
				for keyB in all_filters_posB:
					if keyA == keyB:
						continue
					intr = keyA+'<-->'+keyB
					rev_intr = keyB+'<-->'+keyA
					if intr in Filter_Intr_Keys:
						x_ind = Filter_Intr_Keys[intr]
					elif rev_intr in Filter_Intr_Keys:
						x_ind = Filter_Intr_Keys[rev_intr]
							
					if attn_val > filter_Intr_Attn[x_ind]:#[y_ind]:
						filter_Intr_Attn[x_ind] = attn_val #[y_ind] = attn_val
					filter_Intr_Pos[x_ind] = pos_diff#[y_ind] = pos_diff
						
	return y_ind,filter_Intr_Attn,filter_Intr_Pos


def estimate_interactions(num_filters, params, tomtom_data, motif_dir, verbose = False, CNNfirstpool = 6, sequence_len = 200, pos_score_cutoff = 0.65, seq_limit = -1, attn_cutoff = 0.25, for_background = False, numWorkers=1, storeInterCNN = True, considerTopHit = True, useAll=False, precisionLimit=0.5):
	global Prob_Attention_All# = res_test[3]
	global Seqs# = res_test[6]
	global LabelPreds# = res_test[4]
	global Filter_Intr_Attn
	global Filter_Intr_Pos
	global Filter_Intr_Keys
	global Filter_Intr_Attn_neg
	global Filter_Intr_Pos_neg
	global tp_pos_dict
	
	final_all = [['Batch','ExNo','SeqHeader','SingleHeadNo','PositionA','PositionB','AveragePosDiff','AttnScore','PositionAInfo','PositionBInfo']]
	count = 0		
	for k in range(0,len(Prob_Attention_All)): #going through all batches
		start_time = time.time()
		if count == seq_limit: #break if seq_limit number of sequences tested
			break
		
		PAttn = Prob_Attention_All[k]
		if storeInterCNN:
			with open(PAttn,'rb') as f:
				PAttn = pickle.load(f)

		feat_size = PAttn.shape[1]
		per_batch_labelPreds = LabelPreds[k]
		
		if num_labels == 2:
			if for_background and for_background != None:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
			else:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
		else:
			if useAll == True:
				tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
			else:
				tp_indices=[] 
				TPs = {}                                                                                                                                                                                   
				batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
				batch_preds = per_batch_labelPreds['preds']   
				headers = np.asarray(Seqs[k][:,0])                                                                                                                                                                        
				for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
					ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
					ex_preds = batch_preds[e]                                                                                                                                                                                   
					ex_preds = np.asarray([i>=0.5 for i in ex_preds]).astype(int)    
					prec = metrics.precision_score(ex_labels, ex_preds)         
					if prec >= precisionLimit:   
						TP = [i for i in range(0,ex_labels.shape[0]) if (ex_labels[i]==1 and ex_preds[i]==1)] #these are going to be used in calculating attributes: average accross only those columns which are true positives                                                                                                                               
						tp_indices.append(e)
						tp_pos_dict[headers[e]] = TP
						TPs[e] = TP

		Seqs_tp = Seqs[k][tp_indices]
		print('generating sequence position information...')
		seq_info_dict = get_filters_in_seq_dict(Seqs_tp,motif_dir,num_filters,CNNfirstpool,numWorkers=numWorkers)
		print('Done!')

		fdata = []
		for ex in tp_indices:
			header = np.asarray(Seqs[k])[ex][0]
			fdata.append([count,header,seq_info_dict[header],k,ex,params,tomtom_data,attn_cutoff,sequence_len,CNNfirstpool,num_filters,motif_dir,feat_size,storeInterCNN, considerTopHit])
			count += 1
			if count == seq_limit:
				break

		with Pool(processes = numWorkers) as pool:
			result = pool.map(score_individual_head, fdata, chunksize=1)
		#pdb.set_trace()
		for element in result:
			bid = element[0]
			if for_background == False:
				Filter_Intr_Pos[:,bid] = element[2]
				Filter_Intr_Attn[:,bid] = element[1]
			else:
				Filter_Intr_Pos_neg[:,bid] = element[2]
				Filter_Intr_Attn_neg[:,bid] = element[1]
		
		end_time = time.time()
		if verbose:	
			print("Done for Batch: ",k, "Sequences Done: ",count, "Time Taken: %d seconds"%round(end_time-start_time))
				#print("Done for batch: ",k, "example: ",ex, "count: ",count)
	pop_size = count * params['num_multiheads'] #* int(np.ceil(attn_cutoff)) #total sequences tested x # multi heads x number of top attn scores allowed
	return pop_size


def estimate_interactions_bg(num_filters, params, tomtom_data, motif_dir, verbose = False, CNNfirstpool = 6, sequence_len = 200, pos_score_cutoff = 0.65, seq_limit = -1, attn_cutoff = 0.25, for_background = False, numWorkers=1, storeInterCNN = True, considerTopHit = True, useAll=False):
	global Prob_Attention_All_neg# = res_test[3]
	global Seqs_neg# = res_test[6]
	global LabelPreds_neg# = res_test[4]
	global Filter_Intr_Attn_neg
	global Filter_Intr_Pos_neg
	global Filter_Intr_Keys
	global tp_pos_dict
	
	final_all = [['Batch','ExNo','SeqHeader','SingleHeadNo','PositionA','PositionB','AveragePosDiff','AttnScore','PositionAInfo','PositionBInfo']]
	count = 0		
	for k in range(0,len(Prob_Attention_All_neg)): #going through all batches
		start_time = time.time()
		if count == seq_limit: #break if seq_limit number of sequences tested
			break
		
		PAttn = Prob_Attention_All_neg[k]
		if storeInterCNN:
			with open(PAttn,'rb') as f:
				PAttn = pickle.load(f)

		feat_size = PAttn.shape[1]
		per_batch_labelPreds = LabelPreds_neg[k]
		
		if num_labels == 2:
			if for_background:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0])]
			else:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
		else:
			if useAll==True:
				tp_indices = [i for i in range(0,per_batch_labelPreds['labels'].shape[0])]
			else:
				tp_indices = []
				TPs = {}
				headers = np.asarray(Seqs_neg[k][:,0])
				for h_i in range(0,len(headers)):
					header = headers[h_i]
					if header in tp_pos_dict:
						tp_indices.append(h_i)
						TPs[h_i] = tp_pos_dict[header]
		Seqs_tp = Seqs_neg[k][tp_indices]
		print('Generating sequence position information...')
		seq_info_dict = get_filters_in_seq_dict(Seqs_tp,motif_dir,num_filters,CNNfirstpool,numWorkers=numWorkers)
		print('Done!')

		fdata = []
		for ex in tp_indices:
			header = np.asarray(Seqs_neg[k])[ex][0]
			fdata.append([count,header,seq_info_dict[header],k,ex,params,tomtom_data,attn_cutoff,sequence_len,CNNfirstpool,num_filters,motif_dir,feat_size,storeInterCNN, considerTopHit])
			count += 1
			if count == seq_limit:
				break
			
		with Pool(processes = numWorkers) as pool:
			result = pool.map(score_individual_head_bg, fdata, chunksize=1)
		
		for element in result:
			bid = element[0]
			Filter_Intr_Pos_neg[:,bid] = element[2]
			Filter_Intr_Attn_neg[:,bid] = element[1]

		end_time = time.time()
		if verbose:	
			print("Done for Batch: ",k, "Sequences Done: ",count, "Time Taken: %d seconds"%round(end_time-start_time))
	pop_size = count * params['num_multiheads'] #* int(np.ceil(attn_cutoff)) #total sequences tested x # multi heads x number of top attn scores allowed
	return pop_size


# a function that can be used to process the interactions, generate plots and other stuff
# perhaps can use a function from post process for ploting
def analyze_interactions(argSpace, Interact_dir, tomtom_data, plot_dist=True):
	if plot_dist:
		resMain = Filter_Intr_Attn[Filter_Intr_Attn!=-1]                                                                                                                                               
		resBg = Filter_Intr_Attn_neg[Filter_Intr_Attn_neg!=-1]
		resMainHist = np.histogram(resMain,bins=20)
		resBgHist = np.histogram(resBg,bins=20)
		plt.plot(resMainHist[1][1:],resMainHist[0]/sum(resMainHist[0]),linestyle='--',marker='o',color='g',label='main')
		plt.plot(resBgHist[1][1:],resBgHist[0]/sum(resBgHist[0]),linestyle='--',marker='x',color='r',label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/normalized_Attn_scores_distributions.pdf')
		plt.clf()

		plt.hist(resMain,bins=20,color='g',label='main')
		plt.hist(resBg,bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/Attn_scores_distributions.pdf')
		plt.clf()
		
		Bg_MaxMean = []
		Main_MaxMean = []
		for entry in Filter_Intr_Attn:
			try:
				Main_MaxMean.append([np.max(entry[entry!=-1]),np.mean(entry[entry!=-1])])
			except:
				continue	
		for entry in Filter_Intr_Attn_neg:
			try:
				Bg_MaxMean.append([np.max(entry[entry!=-1]),np.mean(entry[entry!=-1])])
			except:
				continue
			
		Bg_MaxMean = np.asarray(Bg_MaxMean)
		Main_MaxMean = np.asarray(Main_MaxMean)
		
		plt.hist(Main_MaxMean[:,0],bins=20,color='g',label='main')
		plt.hist(Bg_MaxMean[:,0],bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/Attn_scores_distributions_MaxPerInteraction.pdf')
		plt.clf()
		
		plt.hist(Main_MaxMean[:,1],bins=20,color='g',label='main')
		plt.hist(Bg_MaxMean[:,1],bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(Interact_dir+'/Attn_scores_distributions_MeanPerInteraction.pdf')
		plt.clf()
	
	attnLimits = [argSpace.attnCutoff * i for i in range(1,11)] #save results for 10 different attention cutoff values (maximum per interaction) eg. [0.05, 0.10, 0.15, 0.20, 0.25, ...]
	for attnLimit in attnLimits:
		pval_info = []#{}
		for i in range(0,Filter_Intr_Attn.shape[0]):                                                                                                                                                   
			pos_attn = Filter_Intr_Attn[i,:]                                                                                                                                                              
			pos_attn = pos_attn[pos_attn!=-1]#pos_attn[pos_attn>0.04] #pos_attn[pos_attn!=-1]                                                                                                                                                                   
			neg_attn = Filter_Intr_Attn_neg[i,:]                                                                                                                                                          
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
			neg_posn = Filter_Intr_Pos_neg[i,:]  
			#neg_posn_mean = neg_posn[neg_posn!=-1].mean()
			neg_posn_mean = neg_posn[np.argmax(Filter_Intr_Attn_neg[i,:])] #just pick the max
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
		np.savetxt(Interact_dir+'/interactions_summary_attnLimit-'+str(attnLimit)+'.txt',final_interactions,fmt='%s',delimiter='\t')
		with open(Interact_dir+'/processed_results_attnLimit-'+str(attnLimit)+'.pckl','wb') as f:
			pickle.dump([pval_info,res_final_int],f)
		print("Done for Attention Cutoff Value: ",str(attnLimit))


### Global variables ###
Prob_Attention_All = None
Prob_Attention_All_neg = None
Seqs = None
Seqs_neg = None
LabelPreds = None
LabelPreds_neg = None
Filter_Intr_Attn = None
Filter_Intr_Attn_neg = None
Filter_Intr_Pos = None
Filter_Intr_Pos_neg = None
Filter_Intr_Keys = None
tp_pos_dict = None
num_labels = None
#######################


# entry point to this module: will do all the processing (will be called from satori.py)
def infer_intr_attention(experiment_blob, params, argSpace):
	global Prob_Attention_All
	global Prob_Attention_All_neg
	global Seqs
	global Seqs_neg
	global LabelPreds
	global LabelPreds_neg
	global Filter_Intr_Attn
	global Filter_Intr_Attn_neg
	global Filter_Intr_Pos
	global Filter_Intr_Pos_neg
	global Filter_Intr_Keys
	global tp_pos_dict
	global num_labels

	Prob_Attention_All = experiment_blob['res_test'][3]
	LabelPreds = experiment_blob['res_test'][4]
	Seqs = experiment_blob['res_test'][6]
	output_dir = experiment_blob['output_dir']
	motif_dir_pos = experiment_blob['motif_dir_pos']
	motif_dir_neg = experiment_blob['motif_dir_neg']

	Interact_dir = output_dir + '/Interactions_SATORI'
	if not os.path.exists(Interact_dir):
	    os.makedirs(Interact_dir)
	tomtom_data = np.loadtxt(motif_dir_pos+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
	if argSpace.intBackground != None:
		tomtom_data_neg = np.loadtxt(motif_dir_neg+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
	num_filters = params['CNN_filters']
	CNNfirstpool = params['CNN_poolsize']
	batchSize = params['batch_size']
	sequence_len = len(Seqs[0][0][1])
	num_labels = argSpace.numLabels

	Filter_Intr_Keys = get_intr_filter_keys(num_filters)
	numPosExamples,numNegExamples = get_popsize_for_interactions(argSpace, experiment_blob['res_test'][4], batchSize)

	Filter_Intr_Attn = np.ones((len(Filter_Intr_Keys),numPosExamples))*-1
	Filter_Intr_Pos = np.ones((len(Filter_Intr_Keys),numPosExamples)).astype(int)*-1
	tp_pos_dict = {}
	
	_ = estimate_interactions(num_filters, params, tomtom_data, motif_dir_pos, verbose = argSpace.verbose, CNNfirstpool = CNNfirstpool, 
											   sequence_len = sequence_len, pos_score_cutoff = argSpace.scoreCutoff, seq_limit = argSpace.intSeqLimit, attn_cutoff = argSpace.attnCutoff,
											   for_background = False, numWorkers = argSpace.numWorkers, storeInterCNN = argSpace.storeInterCNN, considerTopHit = True, useAll=argSpace.useAll, precisionLimit=argSpace.precisionLimit) #considerTopHit never used but kept for future use
	
	Filter_Intr_Attn_neg = np.ones((len(Filter_Intr_Keys),numNegExamples))*-1
	Filter_Intr_Pos_neg = np.ones((len(Filter_Intr_Keys),numNegExamples)).astype(int)*-1
	
	if argSpace.intBackground == 'negative':
		_ = estimate_interactions(num_filters, params, tomtom_data_neg, motif_dir_neg, verbose = argSpace.verbose, CNNfirstpool = CNNfirstpool, 
											   sequence_len = sequence_len, pos_score_cutoff = argSpace.scoreCutoff, seq_limit = argSpace.intSeqLimit, attn_cutoff = argSpace.attnCutoff,
											   for_background = True, numWorkers = argSpace.numWorkers, storeInterCNN = argSpace.storeInterCNN, considerTopHit = True) 
	elif argSpace.intBackground == 'shuffle':
		Prob_Attention_All_neg = experiment_blob['res_test_bg'][3]
		LabelPreds_neg = experiment_blob['res_test_bg'][4]
		Seqs_neg = experiment_blob['res_test_bg'][6]
		_ = estimate_interactions_bg(num_filters, params, tomtom_data_neg, motif_dir_neg, verbose = argSpace.verbose, CNNfirstpool = CNNfirstpool, 
											   sequence_len = sequence_len, pos_score_cutoff = argSpace.scoreCutoff, seq_limit = argSpace.intSeqLimit, attn_cutoff = argSpace.attnCutoff,
											   for_background = True, numWorkers = argSpace.numWorkers, storeInterCNN = argSpace.storeInterCNN, considerTopHit = True) 
	with open(Interact_dir+'/interaction_keys_dict.pckl','wb') as f:
		pickle.dump(Filter_Intr_Keys,f)
	with open(Interact_dir+'/background_results_raw.pckl','wb') as f:
		pickle.dump([Filter_Intr_Attn_neg,Filter_Intr_Pos_neg],f)	
	with open(Interact_dir+'/main_results_raw.pckl','wb') as f:
		pickle.dump([Filter_Intr_Attn,Filter_Intr_Pos],f)
	
	analyze_interactions(argSpace, Interact_dir, tomtom_data)
	
