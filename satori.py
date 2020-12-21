####################################################################################################################
##################################--------------Argument Parsing--------------######################################
def parseArgs():
    """Parse command line arguments
    
    Returns
    -------
    a : argparse.ArgumentParser
    
    """
    parser = ArgumentParser(description='Main deepSAMIREI script.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', 
                        default=False, help="verbose output [default is quiet running]")
    parser.add_argument('-o','--outDir', dest='directory', type=str,
                        action='store', help="output directory", default='')
    parser.add_argument('-m','--mode', dest='mode', type=str,
                        action='store', help="Mode of operation: train or test.", default='train')     
    parser.add_argument('--deskload', dest='deskLoad',
                        action='store_true', default=False,
                        help="Load dataset from desk. If false, the data is converted into tensors and kept in main memory (not recommended for large datasets).")  
    parser.add_argument('-w','--numworkers', dest='numWorkers', type=int,
                        action='store', help="Number of workers used in data loader. For loading from the desk, use more than 1 for faster fetching.", default=1)        
    parser.add_argument('--splitperc', dest='splitperc', type=float, action='store',
                        help="Pecentages of test, and validation data splits, eg. 10 for 10 percent data used for testing and validation.", default=10)
    parser.add_argument('--motifanalysis', dest='motifAnalysis',
                        action='store_true', default=False,
                        help="Analyze CNN filters for motifs and search them against known TF database.")
    parser.add_argument('--scorecutoff', dest='scoreCutoff', type=float,
                        action='store', default=0.65,
                        help="In case of binary labels, the positive probability cutoff to use.")
    parser.add_argument('--tomtompath', dest='tomtomPath',
                        type=str, action='store', default=None,
                        help="Provide path to where TomTom (from MEME suite) is located.") 
    parser.add_argument('--database', dest='tfDatabase', type=str, action='store',
                        help="Search CNN motifs against known TF database. Default is Human CISBP TFs.", default=None)
    parser.add_argument('--annotate',dest='annotateTomTom',type=str,action='store',
                        default=None, help="Annotate tomtom motifs. The value of this variable should be path to the database file used for annotation. Default is None.")                   
    parser.add_argument('-a','--attnfigs', dest='attnFigs',
                        action='store_true', default=False,
                        help="Generate Attention (matrix) figures for every test example.")
    parser.add_argument('-i','--interactions', dest='featInteractions',
                        action='store_true', default=False,
                        help="Self attention based feature(TF) interactions analysis.")
    parser.add_argument('-b','--background', dest='intBackground', type=str,
                        action='store', default=None,
                        help="Background used in interaction analysis: shuffle (for di-nucleotide shuffled sequences with embedded motifs.), negative (for negative test set). Default is not to use background (and significance test).")
    parser.add_argument('--attncutoff', dest='attnCutoff', type=float,
                        action='store', default=0.04,
                        help="Attention (probability) cutoff value to use while searching for maximum interaction. A value (say K) greater than 1.0 will mean using top K interaction values.") #In human promoter DHSs data analysis, lowering the cutoff leads to more TF interactions. 
    parser.add_argument('--intseqlimit', dest='intSeqLimit', type=int,
                        action='store', default = -1,
                        help="A limit on number of input sequences to test. Default is -1 (use all input sequences that qualify).")
    parser.add_argument('-s','--store', dest='storeInterCNN',
                        action='store_true', default=False,
                        help="Store per batch attention and CNN outpout matrices. If false, the are kept in the main memory.")
    parser.add_argument('--numlabels', dest='numLabels', type=int,
                        action='store', default = 2,
                        help="Number of labels. 2 for binary (default). For multi-class, multi label problem, can be more than 2. ")
    parser.add_argument('--tomtomdist', dest='tomtomDist', type=str,
                        action='store', default = 'pearson',
                        help="TomTom distance parameter (pearson, kullback, ed etc). Default is pearson. See TomTom help from MEME suite.")
    parser.add_argument('--tomtompval', dest='tomtomPval', type=float,
                        action='store', default = 0.05,
                        help="Adjusted p-value cutoff from TomTom. Default is 0.05.")
    parser.add_argument('--testall', dest='testAll',
                        action='store_true', default=False,
                        help="Test on the entire dataset (default False). Useful for interaction/motif analysis.")
    parser.add_argument('--useall', dest='useAll',
                        action='store_true', default=False,
                        help="Use all examples in multi-label problem instead of using precision based example selection.  Default is False.")
    parser.add_argument('--precisionlimit', dest='precisionLimit', type=float,
                        action='store', default = 0.50,
                        help="Precision limit to use for selecting examples in case of multi-label problem.")
    parser.add_argument('--attrbatchsize', dest='attrBatchSize', type=int,
                        action='store', default = 12,
                        help="Batch size used while calculating attributes for FIS scoring. Default is 12.")
    parser.add_argument('--method', dest='methodType', type=str,
                        action='store', default='SATORI',
                        help="Interaction scoring method to use; options are: SATORI, DFIM, or BOTH. Default is SATORI.")
    parser.add_argument('inputprefix', type=str,
                        help="Input file prefix for the bed/text file and the corresponding fasta file (sequences).")
    parser.add_argument('hparamfile',type=str,
                        help='Name of the hyperparameters file to be used.')

    args = parser.parse_args()

    return args
####################################################################################################################


def main():
    #CUDA for pytorch
    use_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
    cudnn.benchmark = True
    arg_space = parseArgs()
    #create params dictionary
    params_dict = get_params_dict(arg_space.hparamfile)
    test_resBlob, CNNWeights = run_experiment(device, arg_space, params_dict)
    if arg_space.motifAnalysis:
        motif_dir_pos, num_examples_pos = motif_analysis(test_resBlob, CNNWeights, arg_space)
        motif_dir_neg, num_examples_neg = motif_analysis(test_resBlob, CNNWeights,  arg_space, for_negative=True)
    
    if arg_space.annotateTomTom != None:
        if arg_space.verbose:
            print("Annotating motifs...")
        annotate_motifs(arg_space.annotateTomTom, motif_dir_pos)
        annotate_motifs(arg_space.annotateTomTom, motif_dir_neg)

    if arg_space.featInteractions:
        if arg_space.methodType in ['SATORI','BOTH']:
            _ = infer_intr_attention()
        if arg_space.methodType in ['FIS','BOTH']:
            _ = infer_intr_FIS()


if __name__ == "__main__":
    main()