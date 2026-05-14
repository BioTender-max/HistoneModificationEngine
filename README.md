# HistoneModificationEngine

**ChIP-seq Histone Mark Analysis Pipeline**

A pure-Python computational engine for histone modification analysis from ChIP-seq data.

## Features
- Peak calling (fold-enrichment over input, Poisson p-value, BH FDR)
- Bivalent domain detection (H3K4me3 + H3K27me3 co-occurrence)
- 5-state chromatin segmentation (active/poised/repressed/heterochromatin/quiescent)
- Histone mark correlation matrix
- Enhancer/promoter classification from mark combinations

## Results
- 200 genomic bins × 5 histone marks, 2 cell types
- Total peaks (Cell A): 66, (Cell B): 68
- Bivalent domains: A=1, B=4
- H3K4me3-H3K27ac correlation: 0.503

## Usage
```bash
pip install numpy scipy matplotlib
python histone_modification_engine.py
```

## Tags
`histone-modification` `chip-seq-peak` `h3k27ac` `h3k4me3` `bivalent-chromatin` `chromatin-state`
