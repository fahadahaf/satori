## SATORI v0.2 input data format
SATORI requires the input data to be split into two files:  
1. A ``fasta`` file containing all the input sequences. It should follow the standard fasta format with a header followed by the actual sequence. Here's an example:  
```
>chr13:19314260-19314350(+)
TTACACATGTGGATCCTCGTTTTCCAAGCATGGCTTGTTTGTTTTGATTTCTGCTGTGCTTATAAATCACTTTCGGTGGGCAAGGGAGGA
```
The header has the following format:  
```
>SEQID:START-END(STRAND)
```
where the SEQID can be either a unique identifier or the chromosome from which the sequence comes from.  
2. A tab-delimited text file containing the header and label information. This file is very similar to a ``bed`` file except with the label information in the last column. Here's an example:  
```
chr13   19314260    19314350    0,1,2,3
```
In case of binary classification, the last column should have a single value (either 0 or 1). For multi-label classification problems, the labels should be comma-separated.