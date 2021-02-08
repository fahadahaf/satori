import numpy as np
import os
import random
import torch

from Bio.motifs import minimal
from deeplift_dinuc_shuffle import dinuc_shuffle
from fastprogress import progress_bar
from sklearn import metrics


def get_params_dict(params_path):
    param_data = np.loadtxt(params_path, dtype=str, delimiter='|')
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
    return params
def calculate_padding(inputLength, filterSize):
    padding = inputLength - (inputLength - filterSize + 1)
    return int(padding/2) #appended to both sides the half of what is needed


def annotate_motifs(annotate_arg, motif_dir):
    ###-----------------Adding TF details to TomTom results----------------###
        try:
            tomtom_res = np.loadtxt(motif_dir+'/tomtom/tomtom.tsv',dtype=str,delimiter='\t')
        except: #shouldn't stop the flow of program if this fails
            print("Error! motif file not found. Make sure to do motif analysis first.")
            return
        if annotate_arg == 'default':#this for now makes sure that we don't annotate (will be added in future).
            database = np.loadtxt('../../Basset_Splicing_IR-iDiffIR/Analysis_For_none_network-typeB_lotus_posThresh-0.60/MEME_analysis/Homo_sapiens_2019_01_14_4_17_pm/TF_Information_all_motifs.txt',dtype=str,delimiter='\t')
        else:
            database = np.loadtxt(annotate_arg, dtype=str, delimiter='\t')
        final = []                                     
        for entry in tomtom_res[1:]:
            motifID = entry[1]                         
            res = np.argwhere(database[:,3]==motifID)
            TFs = ','.join(database[res.flatten(),6])
            final.append(entry.tolist()+[TFs])
        np.savetxt(motif_dir+'/tomtom/tomtom_annotated.tsv', final, delimiter='\t', fmt='%s')


def get_popsize_for_interactions(argSpace, per_batch_labelPreds, batchSize):
    pos_score_cutoff = argSpace.scoreCutoff
    num_labels = argSpace.numLabels
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
    if argSpace.verbose:
        print('Positive and Negative Population: ',	numPosExamples, numNegExamples)	
    return numPosExamples,numNegExamples


def get_intr_filter_keys(num_filters=200):
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
    return Filter_Intr_Keys


def get_random_seq(pwm, alphabet=['A','C','G','T']):
    seq = ''
    for k in range(0,pwm.shape[0]):
        nc = np.random.choice(alphabet,1,p=pwm[k,:])
        seq += nc[0]
    return seq


def get_shuffled_background(tst_loader, argSpace): #this function is similar to the first one (get_shuffled_background()) however, instead of using consensus, it generates a random sequences (of same size as the PWM) based on the probability distributions in the matrix
    out_directory = argSpace.directory+'/Temp_Data'
    if argSpace.mode == 'test':
        if os.path.exists(out_directory+'/'+'shuffled_background.txt') and os.path.exists(out_directory+'/'+'shuffled_background.fa'):
            return out_directory+'/'+'shuffled_background' #name of the prefix to use
        else:
            print("Shuffled data missing! Regenerating...")
    labels_array = np.asarray([i for i in range(0,argSpace.numLabels)])
    final_fa = []
    final_bed = []
    for batch_idx, (headers,seqs,_,batch_targets) in enumerate(tst_loader):
        for i in range (0, len(headers)):
            header = headers[i]
            seq = seqs[i]
            targets = batch_targets[i]
            seq = dinuc_shuffle(seq)
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
    try:
        with open(argSpace.directory+'/Motif_Analysis/filters_meme.txt') as f:
            filter_motifs = minimal.read(f)
    except:
        raise Exception("motif file not found! Make sure to extract motifs first from the network output (hint: use --motifanalysis)")
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
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    np.savetxt(out_directory+'/'+'shuffled_background.fa',np.asarray(final_fa).flatten(),fmt='%s')
    np.savetxt(out_directory+'/'+'shuffled_background.txt',np.asarray(final_bed), fmt='%s',delimiter='\t')
    return out_directory+'/'+'shuffled_background' #name of the prefix to use