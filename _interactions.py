class Interactions():
    def __init__(self):

    @staticmethod 
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
    
    @staticmethod
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
    


class Interactions_Attn(Interactions):


class Interactions_FIS(Interactions):

