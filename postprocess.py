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
        return TF_A[0]
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


def plot_interactions_and_distances_boxplot(dfx, first_n=20, sort_distances=False, tick_fontsize=14, label_fontsize=19, add_sub_caption=True, show_mean_dist=True, store_pdf_path=None):
    df = dfx.copy()
    res = df['TF_Interaction'].value_counts()[:first_n]
    list_distance = df.groupby('TF_Interaction')['mean_distance'].apply(list)[res.index]
    df_distance = pd.DataFrame([list_distance.keys(), list_distance.values]).T
    df_distance.columns = ['TF_Interaction', 'distances']
    df_distance['mean_distance'] = df_distance['distances'].apply(lambda x: np.mean(x))
    mean_dist = df['mean_distance'].mean()
    if sort_distances:
        df_distance.sort_values(by='mean_distance', inplace=True)
    fig, axes = plt.subplots(1, 2)
    ax1 = res.plot(kind='bar', color='salmon', figsize=(18,5), fontsize=tick_fontsize, ax=axes[0])
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
        ax1.text(0.5,-0.69, "(a)", size=23, ha="center", 
                transform=ax1.transAxes)
        ax2.text(0.5,-0.69, "(b)", size=23, ha="center", 
                transform=ax2.transAxes)
    if show_mean_dist:
        ax2.axhline(y=mean_dist, xmin=0.0, xmax=1.0, color='steelblue', ls='--', lw=1.5)
    if store_pdf_path:
        plt.savefig(store_pdf_path, bbox_inches='tight')
    plt.show()


def plot_interactions_and_distances_histogram(dfx, first_n=20, dist_nbins=25, tick_fontsize=14, label_fontsize=19, add_sub_caption=True, show_mean_dist=True, store_pdf_path=None):
    df = dfx.copy()
    res = df['TF_Interaction'].value_counts()[:first_n]
    mean_dist = df['mean_distance'].mean()
    fig, axes = plt.subplots(1, 2)
    ax1 = res.plot(kind='bar', color='salmon', figsize=(18,5), fontsize=tick_fontsize, ax=axes[0])
    ax1.set_xlabel("motif interaction",fontsize=label_fontsize)
    ax1.set_ylabel("# of occurences",fontsize=label_fontsize)
    ax1.xaxis.set_tick_params(rotation=90)
    ax2 = df['mean_distance'].plot(kind='hist',bins=dist_nbins, figsize=(18,5), color='cadetblue', fontsize=tick_fontsize, ax=axes[1])
    ax2.set_xlabel("interaction distance",fontsize=label_fontsize)
    ax2.set_ylabel("frequency",fontsize=label_fontsize)
    ax2.xaxis.set_tick_params(rotation=0)
    if add_sub_caption:
        ax1.text(0.5,-0.69, "(a)", size=23, ha="center", 
                transform=ax1.transAxes)
        ax2.text(0.5,-0.69, "(b)", size=23, ha="center", 
                transform=ax2.transAxes)
    if show_mean_dist:
        ax2.axvline(x=mean_dist, ymin=0.0, ymax=1.0, color='indianred', ls='--', lw=1.5)
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

