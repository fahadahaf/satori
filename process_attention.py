import numpy as np


def get_filters_in_individual_seq(sdata):
	header,num_filters,filter_data_dict,CNNfirstpool = sdata
	s_info_dict = {}
	for j in range(0,num_filters):
		filter_data = filter_data_dict['filter'+str(j)] #np.loadtxt(motif_dir+'/filter'+str(j)+'_logo.fa',dtype=str)
		for k in range(0,len(filter_data),2):
			hdr = filter_data[k].split('_')[0]
			if hdr == header:
				pos = int(filter_data[k].split('_')[-1])
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
	
	attn_mat = np.asarray([attn_mat[:,feat_size*i:feat_size*(i+1)] for i in range(0,params['numMultiHeads'])]) 
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
	
	attn_mat = np.asarray([attn_mat[:,feat_size*i:feat_size*(i+1)] for i in range(0,params['numMultiHeads'])]) 
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


def estimate_interactions(num_filters, params, tomtom_data, motif_dir, verbose = False, CNNfirstpool = 6, sequence_len = 200, pos_score_cutoff = 0.65, seq_limit = -1, attn_cutoff = 0.25, for_background = False, numWorkers=1, storeInterCNN = True, considerTopHit = True):
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
			if for_background and argSpace.intBackground != None:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==0 and per_batch_labelPreds[i][1]<(1-pos_score_cutoff))]
			else:
				tp_indices = [i for i in range(0,per_batch_labelPreds.shape[0]) if (per_batch_labelPreds[i][0]==1 and per_batch_labelPreds[i][1]>pos_score_cutoff)]
		else:
			if argSpace.useAll == True:
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
					prec = metrics.precision_score(ex_labels,ex_preds)         
					if prec >= argSpace.precisionLimit:   
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
		if argSpace.verbose:	
			print("Done for Batch: ",k, "Sequences Done: ",count, "Time Taken: %d seconds"%round(end_time-start_time))
				#print("Done for batch: ",k, "example: ",ex, "count: ",count)
	pop_size = count * params['numMultiHeads'] #* int(np.ceil(attn_cutoff)) #total sequences tested x # multi heads x number of top attn scores allowed
	return pop_size


def estimate_interactions_bg(num_filters, params, tomtom_data, motif_dir, verbose = False, CNNfirstpool = 6, sequence_len = 200, pos_score_cutoff = 0.65, seq_limit = -1, attn_cutoff = 0.25, for_background = False, numWorkers=1, storeInterCNN = True, considerTopHit = True):
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
			if argSpace.useAll==True:
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
		seq_info_dict = get_filters_in_seq_dict(Seqs_tp,motif_dir_neg,num_filters,CNNfirstpool,numWorkers=numWorkers)
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
		if argSpace.verbose:	
			print("Done for Batch: ",k, "Sequences Done: ",count, "Time Taken: %d seconds"%round(end_time-start_time))
	pop_size = count * params['numMultiHeads'] #* int(np.ceil(attn_cutoff)) #total sequences tested x # multi heads x number of top attn scores allowed
	return pop_size


# -------------Later Part---------------- #
# create a single function that can process a head regardlesss of 
# main or background data
def score_individual_head_exp():
	pass

# create a single function that can estimate interactions regardless of 
# main or background data
def estimate_interactions_exp():
	pass
# --------------------------------------- #

# a function that can be used to process the interactions, generate plots and other stuff
# perhaps can use a function from post process for ploting
def to_be_named_helper():
	pass

# entry point to this module: will do all the processing (will be called from satori.py)
def infer_intr_attention():
	pass