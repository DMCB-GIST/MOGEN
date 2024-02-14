# MOGEN
Multi-omics Gene Embedding based feedforward Neural networks

![figure1](https://github.com/DMCB-GIST/MOGEN/assets/104506641/e8eb0a47-b394-4042-a79c-a2354802112f)

(a) Feature selection : Select individual gene sets by considering the variance of each sample (top) and the negative correlation of features across omics data (bottom).  
(b,c) Attention encoder module : In (b), the omics data is concatenated and used as input to the Attention Encoder, and in (c), each omics data and the concatenated data are put independently.  
(d) Workflow of the study.  


### Datasets
-----------------------

https://drive.google.com/drive/folders/1YHCXnurnK75bP_OztbQmXbVwPCjFhTk-?usp=drive_link


### Installation
-----------------------

```bash
git clone https://github.com/DMCB-GIST/MOGEN.git
```

### Requirements
-----------------------

- pytorch >= 1.8.0
- conda install pyg -c pyg
- pip install scipy
- conda install -c anaconda scikit-learn
- conda install hickle
- We have already provided our environment list as environment.yml. You can create your own environment by:
```bash
conda env create -n envname -f environment.yml
```

