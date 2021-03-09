import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

plt.style.use('seaborn-white')


def filter_data_on_thresholds(dfx, intr_pval_cutoff=0.05, motifA_pval_cutoff=0.01, motifB_pval_cutoff=0.01):
    df = dfx.copy()
    df = df[df['adjusted_pval']<intr_pval_cutoff]
    df = df[(df['motif1_qval'] < motifA_pval_cutoff) & (df['motif2_qval']<motifA_pval_cutoff)]
    return df


def get_annotation(motif, annotation_data=None, single_TF=False):
    TF_A = annotation_data[annotation_data['Motif_ID']==motif]['TF_Name']
    TF_A = list(TF_A)
    if single_TF:
        return TF_A[0] if len(TF_A)!=0 else 'UNKNOWN'
    else:
        return ','.join(TF_A)


def process_for_redundant_interactions(dfx, intr_type='TF'):
    if intr_type == 'TF':
        TF1_name = 'TF1'
        TF2_name = 'TF2'
        TF_intr_name = 'TF_Interaction'
    else:
        TF1_name = 'TF1_Family'
        TF2_name = 'TF2_Family'
        TF_intr_name = 'Family_Interaction'
    df = dfx.copy()
    TF_interactions_list = []
    for i in range(df.shape[0]):
        TF1 = df[TF1_name][i]
        TF2 = df[TF2_name][i]
        TF_intr = TF1+r'$\longleftrightarrow$'+TF2
        TF_intr_rev = TF2+r'$\longleftrightarrow$'+TF1
        if TF_intr not in TF_interactions_list and TF_intr_rev not in TF_interactions_list:
            TF_interactions_list.append(TF_intr)
            
        if TF_intr in TF_interactions_list:
            df.loc[i, TF_intr_name] = TF_intr
        elif TF_intr_rev in TF_interactions_list:
            df.loc[i, TF_intr_name] = TF_intr_rev
    return df


def plot_interaction_distance_distribution(df, nbins=30, fig_size=(12,8), color='slateblue', store_pdf_path=None, title=False):
    ax = df['mean_distance'].plot(kind='hist',bins=nbins, figsize=fig_size, color=color, fontsize=12)
    ax.set_xlabel("interaction distance",fontsize=16)
    ax.set_ylabel("frequency",fontsize=16)
    ax.xaxis.set_tick_params(rotation=0)
    if title:
        ax.set_title('Distribution of motif interaction distances',fontsize=18)
    if store_pdf_path:
        plt.savefig(store_path,bbox_inches='tight')
    plt.plot()


def plot_frequent_interactions(df, intr_level='TF_Interaction', first_n=15, color='steelblue', fig_size=(18,6), store_pdf_path=None):
    ax = df[intr_level].value_counts()[:first_n].plot(kind='bar', color=color, figsize=fig_size, fontsize=14)
    if intr_level == 'TF_Interaction':
        ax.set_xlabel("TF interaction",fontsize=19)
    else:
        ax.set_xlabel("TF family interaction",fontsize=19)
    ax.set_ylabel("# of occurences",fontsize=19)
    ax.xaxis.set_tick_params(rotation=90)
    if store_pdf_path:
        plt.savefig(store_pdf_path, bbox_inches='tight')
    plt.plot()


def plot_interactions_and_distances_boxplot(dfx, first_n=20, sort_distances=False, tick_fontsize=14, label_fontsize=19, add_sub_caption=True, show_median_dist=True, store_pdf_path=None, dist_color='salmon', cap_pos=[0.5,-0.79]):
    df = dfx.copy()
    res = df['TF_Interaction'].value_counts()[:first_n]
    list_distance = df.groupby('TF_Interaction')['mean_distance'].apply(list)[res.index]
    df_distance = pd.DataFrame([list_distance.keys(), list_distance.values]).T
    df_distance.columns = ['TF_Interaction', 'distances']
    df_distance['mean_distance'] = df_distance['distances'].apply(lambda x: np.mean(x))
    median_dist = np.median(df['mean_distance'])
    if sort_distances:
        df_distance.sort_values(by='mean_distance', inplace=True)
    fig, axes = plt.subplots(1, 2)
    ax1 = res.plot(kind='bar', color=dist_color, figsize=(18,5), fontsize=tick_fontsize, ax=axes[0])
    ax1.set_xlabel("motif interaction",fontsize=label_fontsize)
    ax1.set_ylabel("# of occurences",fontsize=label_fontsize)
    ax1.xaxis.set_tick_params(rotation=90)
    ax2 = axes[1]
    ax2.boxplot(df_distance['distances'].values, labels=df_distance['TF_Interaction'])
    ax2.set_xlabel("motif interaction",fontsize=label_fontsize)
    ax2.set_ylabel("distance (bp)",fontsize=label_fontsize)
    ax2.xaxis.set_tick_params(rotation=90)
    ax2.tick_params(axis='x', labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    if add_sub_caption:
        ax1.text(cap_pos[0], cap_pos[1], "(a)", size=23, ha="center", 
                transform=ax1.transAxes)
        ax2.text(cap_pos[0], cap_pos[1], "(b)", size=23, ha="center", 
                transform=ax2.transAxes)
    if show_median_dist:
        ax2.axhline(y=median_dist, xmin=0.0, xmax=1.0, color='steelblue', ls='--', lw=1.5)
    if store_pdf_path:
        plt.savefig(store_pdf_path, bbox_inches='tight')
    plt.show()


def plot_interactions_and_distances_histogram(dfx, first_n=20, dist_nbins=25, tick_fontsize=14, label_fontsize=19, add_sub_caption=True, show_median_dist=True, store_pdf_path=None, dist_colors=['salmon', 'cadetblue'], cap_pos=[0.5,-0.79]):
    df = dfx.copy()
    res = df['TF_Interaction'].value_counts()[:first_n]
    median_dist = np.median(df['mean_distance'])
    fig, axes = plt.subplots(1, 2)
    ax1 = res.plot(kind='bar', color=dist_colors[0], figsize=(18,5), fontsize=tick_fontsize, ax=axes[0])
    ax1.set_xlabel("motif interaction",fontsize=label_fontsize)
    ax1.set_ylabel("# of occurences",fontsize=label_fontsize)
    ax1.xaxis.set_tick_params(rotation=90)
    ax2 = df['mean_distance'].plot(kind='hist',bins=dist_nbins, figsize=(18,5), color=dist_colors[1], fontsize=tick_fontsize, ax=axes[1])
    ax2.set_xlabel("interaction distance",fontsize=label_fontsize)
    ax2.set_ylabel("frequency",fontsize=label_fontsize)
    ax2.xaxis.set_tick_params(rotation=0)
    if add_sub_caption:
        ax1.text(cap_pos[0], cap_pos[1], "(a)", size=23, ha="center", 
                transform=ax1.transAxes)
        ax2.text(cap_pos[0], cap_pos[1], "(b)", size=23, ha="center", 
                transform=ax2.transAxes)
    if show_median_dist:
        ax2.axvline(x=median_dist, ymin=0.0, ymax=1.0, color='indianred', ls='--', lw=1.5)
    if store_pdf_path:
        plt.savefig(store_pdf_path, bbox_inches='tight')
    plt.show()


def db_annotate_interaction(x, intr_dict = None):
    TFs_A,TFs_B = x.split(r'$\longleftrightarrow$')
    TFs_A = TFs_A.split(',')
    TFs_B = TFs_B.split(',')
    for TF_A in TFs_A:
        if TF_A not in intr_dict:
            continue
        for TF_B in TFs_B:
            if TF_B not in intr_dict:
                continue
            TF_A_targets = intr_dict[TF_A][:,0]
            TF_B_targets = intr_dict[TF_B][:,0]
            if len(set.intersection(set(TF_A_targets),set(TF_B_targets))) != 0:
                return 1 #interaction found (the two TFs have shared targets)
    return 0


def preprocess_for_comparison(dfx, annotation_df=None, for_arabidopsis=False, m1_pval=0.05, m2_pval=0.05):
    df = filter_data_on_thresholds(dfx.copy(), motifA_pval_cutoff=m1_pval, motifB_pval_cutoff=m2_pval)
    #--------------- For the TFs ----------------#
    if annotation_df is not None:
        df['TF1'] = df['motif1'].apply(get_annotation, annotation_data=annotation_df, single_TF=True)
        df['TF2'] = df['motif2'].apply(get_annotation, annotation_data=annotation_df, single_TF=True)
    else:
        if for_arabidopsis:
            df['TF1'] = df['motif1'].apply(lambda x: x.split('_')[1].strip('.tnt'))
            df['TF2'] = df['motif2'].apply(lambda x: x.split('_')[1].strip('.tnt'))
        else:
            df['TF1'] = df['motif1']
            df['TF2'] = df['motif2']
    df['TF_Interaction'] = df.apply(lambda x: x['TF1']+r'$\longleftrightarrow$'+x['TF2'], axis=1)
    df = df[df['TF1']!=df['TF2']]
    df = df.reset_index(drop=True)
    df = process_for_redundant_interactions(df, intr_type='TF')
    #--------------- For the families ----------------#
    if annotation_df is not None:
        tf_family_dict = {}
        for TF in annotation_df['TF_Name']:
            tf_family_dict[TF] = annotation_df[annotation_df['TF_Name']==TF]['Family_Name'].iloc[0]
        df['TF1_Family'] = df['TF1'].apply(lambda x: tf_family_dict[x] if x in tf_family_dict else 'UNKNOWN')
        df['TF2_Family'] = df['TF2'].apply(lambda x: tf_family_dict[x] if x in tf_family_dict else 'UNKNOWN')
    else:
        if for_arabidopsis:
            df['TF1_Family'] = df['motif1'].apply(lambda x: x.split('_')[0])
            df['TF2_Family'] = df['motif2'].apply(lambda x: x.split('_')[0])
        else:
            print("Warning! Cannot infer motif families, please provide an annotation reference (see arguments).")
            df['TF1_Family'] = df['TF1'].apply(lambda x: x+'_Family')
            df['TF2_Family'] = df['TF2'].apply(lambda x: x+'_Family')
    df['Family_Interaction'] = df.apply(lambda x: x['TF1_Family']+r'$\longleftrightarrow$'+x['TF2_Family'],axis=1)
    df = process_for_redundant_interactions(df, intr_type='Family')
    return df


def get_comparison_stats(DFIM, ATTN, intr_type='TF_interaction'):
    DFIM_unique = DFIM[intr_type].value_counts()
    ATTN_unique = ATTN[intr_type].value_counts()
    intersected = set.intersection(set(DFIM_unique.keys()),set(ATTN_unique.keys()))
    return ATTN_unique, DFIM_unique, intersected


def common_interaction_stats(df_method1, df_method2):
    final_list = [['interaction','count','in_both']]
    for key in df_method1.keys():
        rev_key = key.split('$\\longleftrightarrow$')[1]+'$\\longleftrightarrow$'+key.split('$\\longleftrightarrow$')[0]
        if key in df_method2 or rev_key in df_method2:
            final_list.append([key,df_method1[key],'b'])
        else:
            final_list.append([key,df_method1[key],'r'])
    final_list = np.asarray(final_list)
    df_res = pd.DataFrame(final_list[1:],columns=final_list[0])   
    df_res['count'] = df_res['count'].apply(lambda x: int(x))
    return df_res


def plot_interaction_comparison(df_comp, first_n=15, xlabel='TF interaction', store_pdf_path=None, fig_size=(9,6), alpha=0.6):
    ax = df_comp[:first_n].plot(kind='bar', x='interaction', y='count', color=df_comp['in_both'], figsize=fig_size, legend=False, alpha=alpha, fontsize=12)
    ax.set_xlabel(xlabel,fontsize=16)
    ax.set_ylabel("# of occurences",fontsize=16)
    NA = mpatches.Patch(color='b', alpha = alpha, label='In both')
    EU = mpatches.Patch(color='r', alpha = alpha, label='FIS only')
    plt.legend(handles=[NA,EU], loc=1,fontsize=14)
    if store_pdf_path:
        plt.savefig(store_pdf_path, bbox_inches='tight') # eg. 'SATORI-vs-FIS_human_indTF.pdf' for filename
    plt.show()