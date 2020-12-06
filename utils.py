import numpy as np

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
        except:
            print("Error! motif file not found. Make sure to do motif analysis first.")
            return
        if annotate_arg == 'default':
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


#-----------Calculating Positive and Negative population------------#
def get_popsize_for_interactions(argSpace, per_batch_labelPreds, batchSize):
    #per_batch_labelPreds = res_test[4]
    ##print(lohahi)
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