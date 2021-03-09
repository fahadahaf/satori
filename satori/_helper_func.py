import numpy as np
import pickle
import pandas as pd
from multiprocessing import Pool
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

def process_intr(Filter_Intr_Attn, Filter_Intr_Attn_neg, attnLimit=0, alt='greater'):
    pval_info = []#{}
    for i in range(0,Filter_Intr_Attn.shape[0]):                                                                                                                                                   
        pos_attn = Filter_Intr_Attn[i,:]                                                                                                                                                              
        pos_attn = pos_attn[pos_attn!=-1]                                                                                                                                                                 
        neg_attn = Filter_Intr_Attn_neg[i,:]                                                                                                                                                          
        neg_attn = neg_attn[neg_attn!=-1] 
        num_pos = len(pos_attn)
        num_neg = len(neg_attn)
        if len(pos_attn) <= 1:# or len(neg_attn) <= 1:
            continue
        if len(neg_attn) <= 1: #if just 1 or 0 values in neg attn, get a vector with all values set to 0 (same length as pos_attn)
            neg_attn = np.asarray([0 for i in range(0,num_pos)])
        if np.max(pos_attn) < attnLimit: # 
            continue
        stats,pval = mannwhitneyu(pos_attn, neg_attn, alternative=alt)                                                      
        pval_info.append([i, num_pos, num_neg, stats, pval])
    
    res_final_int = np.asarray(pval_info) 
    qvals = multipletests(res_final_int[:,-1].astype(float), method='fdr_bh')[1] #res_final_int[:,1].astype(float)
    res_final_int = np.column_stack((res_final_int,qvals))
    return pd.DataFrame(res_final_int)
