import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

from Bio.SeqUtils import GC
from captum.attr import IntegratedGradients
from deeplift.visualization import viz_sequence
from multiprocessing import Pool
from scipy.stats import mannwhitneyu
from sklearn import metrics
from statsmodels.stats.multitest import multipletests
from torch.backends import cudnn
from torch.utils import data

#local imports
from models import AttentionNet
from utils import get_popsize_for_interactions, get_intr_filter_keys


def evaluateRegularBatch(net, batch, criterion, device=None):
	running_loss = 0.0
	valid_auc = []
	net.eval() 
	with torch.no_grad():
		headers, seqs, data, target = batch
		data, target = data.to(device,dtype=torch.float), target.to(device, dtype=torch.long)
		# Model computations
		outputs = net(data)
		loss = criterion(outputs, target)
		softmax = torch.nn.Softmax(dim=1)
		labels=target.cpu().numpy()
		pred = softmax(outputs)
		pred=pred.cpu().detach().numpy()
		label_pred = np.column_stack((labels,pred[:,1]))
		try:
			valid_auc.append(metrics.roc_auc_score(labels, pred[:,1]))
		except:
			valid_auc.append(0.0)
		running_loss += loss.item()
		headers_seqs = np.column_stack((headers,seqs))
	return running_loss,valid_auc,label_pred,headers_seqs
	
	
def evaluateRegularBatchMC(net, batch, criterion, device=None):
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


def get_filters_in_individual_seq(sdata):
	header,num_filters,filter_data_dict,CNNfirstpool = sdata
	s_info_dict = {}
	for j in range(0,num_filters):
		filter_data = filter_data_dict['filter'+str(j)] #np.loadtxt(motif_dir+'/filter'+str(j)+'_logo.fa',dtype=str)
		for k in range(0,len(filter_data),2):
			hdr = filter_data[k].split('_')[0]
			if hdr == header:
				pos = int(filter_data[k].split('_')[-2])
				act_val = float(filter_data[k].split('_')[-1])
				pooled_pos = int(pos/CNNfirstpool)
				key = pos#pooled_pos #we are no longer dealing with attention so lets use the actual position of the filter activation instead
				if key not in s_info_dict:
					s_info_dict[key] = [('filter'+str(j),act_val)]
				else:
					if 'filter'+str(j) not in s_info_dict[key][0]:
						if act_val > s_info_dict[key][0][1]:
							s_info_dict[key] = [('filter'+str(j),act_val)]
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
	with Pool(processes = numWorkers) as pool:
		result = pool.map(get_filters_in_individual_seq,sdata,chunksize=1)
		for subdict in result:
			seq_info_dict.update(subdict)
	return seq_info_dict


#from https://github.com/pytorch/captum/issues/171
def model_wrapper(inputs, model, targets, TPs):
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


def process_FIS(experiment_blob, intr_dir, motif_dir, params, argSpace, Filter_Intr_Keys=None, device=None, tp_pos_dict={}, for_background=False):
	criterion = experiment_blob['criterion']
	train_loader = experiment_blob['train_loader']
	train_indices = experiment_blob['train_indices']
	if for_background:
		test_loader = experiment_blob['test_loader_bg'] if argSpace.intBackground=='shuffle' else experiment_blob['test_loader']
	else:
		test_loader =  experiment_blob['test_loader']
	net = experiment_blob['net']
	saved_model_dir = experiment_blob['saved_model_dir']
	optimizer = experiment_blob['optimizer']
	
	if not os.path.exists(intr_dir):
		os.makedirs(intr_dir)

	num_labels = argSpace.numLabels
	pos_score_cutoff = argSpace.scoreCutoff
	net = AttentionNet(argSpace, params, device=device, genPAttn=False).to(device)
	try:    
	    checkpoint = torch.load(saved_model_dir+'/model')
	    net.load_state_dict(checkpoint['model_state_dict'])
	    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	    epoch = checkpoint['epoch']
	    loss = checkpoint['loss']
	except:
	    raise Exception("No pre-trained model found! Please run with --mode set to train.")

	model = net.to(device)
	model.eval()
	torch.backends.cudnn.enabled=False
	#use the following
	if num_labels == 2:
		dl = IntegratedGradients(model)
	else:
		dl = IntegratedGradients(model_wrapper)

	######-------------------------Some Notes---------------------------############
	#1. For a single position, I am selecting filter with the highest activation
	#   This is different than SATORI since there I considered all filters.
	#   I am doing this to reduce the overhead while calculating all interactions
	################################################################################
	num_filters = params['CNN_filters']
	CNNfirstpool = params['CNN_poolsize'] 
	CNNfiltersize = params['CNN_filtersize']
	batchSize = params['batch_size']
	#GC_content of the train set sequences
	GC_content = GC(''.join(train_loader.dataset.df_seq_final['sequence'][train_indices].values))/100 #0.46 #argSpace.gcContent

	numPosExamples,numNegExamples = get_popsize_for_interactions(argSpace, experiment_blob['res_test'][4], batchSize)			
	numExamples = numNegExamples if for_background else numPosExamples

	Filter_Intr_Attn = np.ones((len(Filter_Intr_Keys),numExamples))*-1
	Filter_Intr_Pos = np.ones((len(Filter_Intr_Keys),numExamples)).astype(int)*-1

	col_index = 0
	for batch_idx, batch in enumerate(test_loader):
		if col_index >= numExamples:
				break

		if num_labels == 2:
			res_test = evaluateRegularBatch(net, batch, criterion, device)
		else:
			res_test = evaluateRegularBatchMC(net, batch, criterion, device)
		
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
				if not for_background:                                                                                                                                                                               
					batch_labels = per_batch_labelPreds['labels']                                                                                                                                                                         
					batch_preds = per_batch_labelPreds['preds']                                                                                                                                                                           
					for e in range(0,batch_labels.shape[0]):                                                                                                                                                                        
						ex_labels = batch_labels[e].astype(int)                                                                                                                                                                     
						ex_preds = batch_preds[e]                                                                                                                                                                                   
						ex_preds = np.asarray([i>=0.5 for i in ex_preds]).astype(int)    
						prec = metrics.precision_score(ex_labels, ex_preds)         
						if prec >= argSpace.precisionLimit:   
							TP = [i for i in range(0,ex_labels.shape[0]) if (ex_labels[i]==1 and ex_preds[i]==1)] #these are going to be used in calculating attributes: average accross only those columns which are true positives                                                                                                                               
							tp_indices.append(e)
							tp_pos_dict[headers[e]] = TP
							TPs[e] = TP
				else:
					for h_i in range(0, len(headers)):
						header = headers[h_i]
						if header in tp_pos_dict:
							tp_indices.append(h_i)
							TPs[h_i] = tp_pos_dict[header]

		#print(len(tp_indices))
		Seqs_tp = Seqs[tp_indices]
		seq_info_dict = get_filters_in_seq_dict(Seqs_tp, motif_dir ,num_filters, CNNfirstpool, numWorkers=argSpace.numWorkers)

		if num_labels == 2:
			datapoints, target = datapoints.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
		else:
			datapoints, target = datapoints.to(device,dtype=torch.float), target.to(device,dtype=torch.float)

		for ind in range(0, len(tp_indices)):#i_pos in range(0,len(pos_ex_ind)):
			i = tp_indices[ind]
			test_points = datapoints[i]#i
			baseline_seq = generate_reference(test_points.shape[-1], seq_GC = GC_content) #0.46 was for the simulated data
			baseline = one_hot_encode(baseline_seq) 
			baseline = torch.Tensor(baseline).to(device, dtype=torch.float)
			test_points = test_points.unsqueeze(dim=0)
			baseline = baseline.unsqueeze(dim=0)
			if num_labels == 2:
				attributions = dl.attribute(test_points, baseline, target=target[i])
			else:
				attributions = dl.attribute(test_points, baseline, additional_forward_args=(model, target[i].unsqueeze(dim=0),TPs[i]))
			res = attributions.squeeze(dim=0).cpu().detach().numpy()
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
					attributions_mut = dl.attribute(tpnt_tuple[start:end],bsln_tuple[start:end],additional_forward_args=(model,trgt_tuple[start:end],TPs[i]))
				count_sbsize = 0
				for sbsize in range(start,end):
					intr_tuple_sub = srcPos_info[sbsize][0]
					pos_tuple_sub = srcPos_info[sbsize][1]
					trgPos_tuple_sub = srcPos_info[sbsize][2]
					trgInd_tuple_sub = srcPos_info[sbsize][3]
					res_mut = attributions_mut[count_sbsize,:,:].squeeze(dim=0).cpu().detach().numpy()
					count_sbsize += 1
					for subsize in range(0,len(intr_tuple_sub)):
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
			if col_index >= numExamples:
				break

	if not for_background:
		with open(intr_dir+'/interaction_keys_dict.pckl','wb') as f:
			pickle.dump(Filter_Intr_Keys, f)
		with open(intr_dir+'/main_results_raw.pckl','wb') as f:
			pickle.dump([Filter_Intr_Attn,Filter_Intr_Pos], f)
		return tp_pos_dict
	else:
		with open(intr_dir+'/background_results_raw.pckl','wb') as f:
			pickle.dump([Filter_Intr_Attn, Filter_Intr_Pos], f)
		return None


def analyze_motif_interactions(argSpace, motif_dir, motif_dir_neg, intr_dir, plot_dist=True):
	tomtom_data = np.loadtxt(motif_dir+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
	if argSpace.intBackground != None:
		tomtom_data_neg = np.loadtxt(motif_dir_neg+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
	
	with open(intr_dir+'/interaction_keys_dict.pckl','rb') as f:
		Filter_Intr_Keys = pickle.load(f)
	with open(intr_dir+'/main_results_raw.pckl','rb') as f:
		Filter_Intr_Attn,Filter_Intr_Pos = pickle.load(f)
	with open(intr_dir+'/background_results_raw.pckl','rb') as f:
		Filter_Intr_Attn_Bg,Filter_Intr_Pos_Bg = pickle.load(f)
	if plot_dist:
		resMain = Filter_Intr_Attn[Filter_Intr_Attn!=-1]                                                                                                                                               
		resBg = Filter_Intr_Attn_Bg[Filter_Intr_Attn_Bg!=-1]
		resMainHist = np.histogram(resMain,bins=20)
		resBgHist = np.histogram(resBg,bins=20)
		plt.plot(resMainHist[1][1:],resMainHist[0]/sum(resMainHist[0]),linestyle='--',marker='o',color='g',label='main')
		plt.plot(resBgHist[1][1:],resBgHist[0]/sum(resBgHist[0]),linestyle='--',marker='x',color='r',label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(intr_dir+'/normalized_Attn_scores_distributions.pdf')
		plt.clf()

		plt.hist(resMain,bins=20,color='g',label='main')
		plt.hist(resBg,bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(intr_dir+'/Attn_scores_distributions.pdf')
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
		plt.savefig(intr_dir+'/Attn_scores_distributions_MaxPerInteraction.pdf')
		plt.clf()

		plt.hist(Main_MaxMean[:,1],bins=20,color='g',label='main')
		plt.hist(Bg_MaxMean[:,1],bins=20,color='r',alpha=0.5,label='background')
		plt.legend(loc='best',fontsize=10)
		plt.savefig(intr_dir+'/Attn_scores_distributions_MeanPerInteraction.pdf')
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
			pos_posn_mean = pos_posn[np.argmax(Filter_Intr_Attn[i,:])] #just pick the max
			neg_posn = Filter_Intr_Pos_Bg[i,:]  
			neg_posn_mean = neg_posn[np.argmax(Filter_Intr_Attn_Bg[i,:])] #just pick the max																																																																															
			stats,pval = mannwhitneyu(pos_attn,neg_attn,alternative='greater')#ttest_ind(pos_d,neg_d)#mannwhitneyu(pos_d,neg_d,alternative='greater')                                                        
			pval_info.append([i, pos_posn_mean, neg_posn_mean,num_pos,num_neg, stats,pval])#pval_dict[i] = [i,stats,pval]                                                                                                                                                              
		
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
		
		np.savetxt(intr_dir+'/interactions_summary_attnLimit-'+str(attnLimit)+'.txt',final_interactions,fmt='%s',delimiter='\t')
		with open(intr_dir+'/processed_results_attnLimit-'+str(attnLimit)+'.pckl','wb') as f:
			pickle.dump([pval_info,res_final_int],f)
		print("Done for Attention Cutoff Value: ",str(attnLimit))

	#time_taken = [['main_loop','bg_loop','total']]
	#time_taken.append([main_time.total_seconds(),bg_time.total_seconds(),main_time.total_seconds()+bg_time.total_seconds()])
	#np.savetxt(intr_dir+'/timing_stats.txt',time_taken,fmt='%s',delimiter='\t')


# entry point to this module: will do all the processing (will be called from satori.py)
def infer_intr_FIS(experiment_blob, params, argSpace, device=None):
	output_dir = experiment_blob['output_dir']
	intr_dir = output_dir + '/Interactions_FIS'

	Filter_Intr_Keys = get_intr_filter_keys(params['CNN_filters']) 

	tp_pos_dict = process_FIS(experiment_blob, intr_dir, experiment_blob['motif_dir_pos'], params, argSpace, Filter_Intr_Keys=Filter_Intr_Keys, device=device)
	_ = process_FIS(experiment_blob, intr_dir, experiment_blob['motif_dir_neg'], params, argSpace, Filter_Intr_Keys=Filter_Intr_Keys, device=device, tp_pos_dict=tp_pos_dict, for_background=True)
	
	analyze_motif_interactions(argSpace, experiment_blob['motif_dir_pos'], experiment_blob['motif_dir_neg'], intr_dir)