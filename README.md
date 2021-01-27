# SATORI v0.2
**SATORI** is **S**elf-**AT**tenti**O**n based deep learning model to capture **R**egulatory element **I**nteractions in genomic sequences. It can be used to infer a global landscape of interactions in a given genomic dataset, without a computationally-expensive post-processing step.

## Manuscript
SATORI manuscript is [available](https://www.biorxiv.org/content/10.1101/2020.01.31.927996v2) on bioRxiv.

## Dependency
**SATORI** is written in python 3. The following python packages are required:  
[biopython (version 1.75)](https://biopython.org)  
[captum (version 0.2.0)](https://captum.ai)  
[fastprogress (version 0.1.21)](https://github.com/fastai/fastprogress)  
[matplotlib (vresion 3.1.3)](https://matplotlib.org)  
[numpy (version 1.17.2)](www.numpy.org)   
[pandas (version 0.25.1)](www.pandas.pydata.org)  
[pytorch (version 1.2.0)](https://pytorch.org)  
[scikit-learn (vresion 0.24)](https://scikit-learn.org/stable/)  
[scipy (version 1.4.1)](www.scipy.org)  
[seaborn (version 0.9.0)](https://seaborn.pydata.org)  
[statsmodels (version 0.9.0)](http://www.statsmodels.org/stable/index.html)  

and for motif analysis:  
[MEME suite](http://meme-suite.org/doc/download.html)  
[WebLogo](https://weblogo.berkeley.edu)

## Installation
1. Download SATORI (via git clone):
```
git clone git@github.com:fahadahaf/satori_v2.git satori
```
2. Navigate to the cloned directory:
```
cd satori
```
3. Install SATORI:
```
python setup.py install
```
4. Make the main script (satori.py) executable:
```
chmod +x satori.py
```
We recommend adding this directory to the PATH environment variable.

## Usage
```
usage: satori.py [-h] [-v] [-o DIRECTORY] [-m MODE] [--deskload]
                 [-w NUMWORKERS] [--splitperc SPLITPERC] [--motifanalysis]
                 [--scorecutoff SCORECUTOFF] [--tomtompath TOMTOMPATH]
                 [--database TFDATABASE] [--annotate ANNOTATETOMTOM] [-i]
                 [-b INTBACKGROUND] [--attncutoff ATTNCUTOFF]
                 [--intseqlimit INTSEQLIMIT] [-s] [--numlabels NUMLABELS]
                 [--tomtomdist TOMTOMDIST] [--tomtompval TOMTOMPVAL]
                 [--testall] [--useall] [--precisionlimit PRECISIONLIMIT]
                 [--attrbatchsize ATTRBATCHSIZE] [--method METHODTYPE]
                 inputprefix hparamfile

Main SATORI script.

positional arguments:
  inputprefix           Input file prefix for the bed/text file and the
                        corresponding fasta file (sequences).
  hparamfile            Name of the hyperparameters file to be used.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose output [default is quiet running]
  -o DIRECTORY, --outDir DIRECTORY
                        output directory
  -m MODE, --mode MODE  Mode of operation: train or test.
  --deskload            Load dataset from desk. If false, the data is
                        converted into tensors and kept in main memory (not
                        recommended for large datasets).
  -w NUMWORKERS, --numworkers NUMWORKERS
                        Number of workers used in data loader. For loading
                        from the desk, use more than 1 for faster fetching.
  --splitperc SPLITPERC
                        Pecentages of test, and validation data splits, eg. 10
                        for 10 percent data used for testing and validation.
  --motifanalysis       Analyze CNN filters for motifs and search them against
                        known TF database.
  --scorecutoff SCORECUTOFF
                        In case of binary labels, the positive probability
                        cutoff to use.
  --tomtompath TOMTOMPATH
                        Provide path to where TomTom (from MEME suite) is
                        located.
  --database TFDATABASE
                        Search CNN motifs against known TF database. Default
                        is Human CISBP TFs.
  --annotate ANNOTATETOMTOM
                        Annotate tomtom motifs. The value of this variable
                        should be path to the database file used for
                        annotation. Default is None.
  -i, --interactions    Self attention based feature(TF) interactions
                        analysis.
  -b INTBACKGROUND, --background INTBACKGROUND
                        Background used in interaction analysis: shuffle (for
                        di-nucleotide shuffled sequences with embedded
                        motifs.), negative (for negative test set). Default is
                        not to use background (and significance test).
  --attncutoff ATTNCUTOFF
                        Attention (probability) cutoff value to use while
                        searching for maximum interaction. A value (say K)
                        greater than 1.0 will mean using top K interaction
                        values.
  --intseqlimit INTSEQLIMIT
                        A limit on number of input sequences to test. Default
                        is -1 (use all input sequences that qualify).
  -s, --store           Store per batch attention and CNN outpout matrices. If
                        false, the are kept in the main memory.
  --numlabels NUMLABELS
                        Number of labels. 2 for binary (default). For multi-
                        class, multi label problem, can be more than 2.
  --tomtomdist TOMTOMDIST
                        TomTom distance parameter (pearson, kullback, ed etc).
                        Default is euclidean (ed). See TomTom help from MEME
                        suite.
  --tomtompval TOMTOMPVAL
                        Adjusted p-value cutoff from TomTom. Default is 0.05.
  --testall             Test on the entire dataset (default False). Useful for
                        interaction/motif analysis.
  --useall              Use all examples in multi-label problem instead of
                        using precision based example selection. Default is
                        False.
  --precisionlimit PRECISIONLIMIT
                        Precision limit to use for selecting examples in case
                        of multi-label problem.
  --attrbatchsize ATTRBATCHSIZE
                        Batch size used while calculating attributes for FIS
                        scoring. Default is 12.
  --method METHODTYPE   Interaction scoring method to use; options are:
                        SATORI, FIS, or BOTH. Default is SATORI.
```

## Tutorial
### Pre-processing
TO-DO

### Example: binary classification
For the TAL-GATA experiment:  
```
satori.py Data/TAL-GATA/Final_dataset_combined_uniq_neg80k_binaryFeat ModelParam/CNN-RNN-MH-noEmbds_hyperParams.txt -w 8 --outDir Results/TAL-GATA_Analysis --mode train -v -s --background negative --intseqlimit 5000 --numlabels 2 --motifanalysis --interactions --method BOTH --attrbatchsize 18 --deskload --tomtompath PATH-TO-TOMTOM-TOOL --database PATH-TO-MEME-TF-DATABASE --annotate No
```
### Example: multi-label classification
For the arabidopsis genomewide chromatin accessibility dataset:  
```
satori.py Data/Arabidopsis/atAll_m200_s600 ModelParam/CNN-RNN-MH-noEmbds_hyperParams.txt -w 8 --outDir Results/Arabidopsis_GenomeWide_Analysis --mode train -v -s --background shuffle --intseqlimit 5000 --numlabels 36 --motifanalysis --interactions --method BOTH --attrbatchsize 32 --deskload --tomtompath PATH-TO-TOMTOM-TOOL --database PATH-TO-MEME-TF-DATABASE --annotate No
```
**Note:** make sure to specify path to the TomTom tool and the corresponding motif database.  
```PATH-TO-TOMTOM-TOOL``` path to TomTom tool in the MEME suite.  
```PATH-TO-MEME-TF-DATABASE``` path to the TF database to use (MEME suite comes with different databases).

### Post-processing
TO-DO