# Intro to torch geometric

I've prepared 3 notebooks:

- `pyg_basics.ipynb` introduces the basics of [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/) graph data and hows it's handled
- `graph_tasks` introduces building graph models in torch geometric and some basic graph-level tasks
- `contrastive_learning.ipynb` introduces contrastive representation learning for graphs and shamelessly plugs my own work

## Environment Setup

You can run these commands if you use conda:

```
conda create -n pyg_intro python=3.11

conda activate pyg_intro

conda install pytorch::pytorch torchvision torchaudio -c pytorch

conda install pyg -c pyg

conda install pandas scikit-learn matplotlib networkx  

conda install rdkit -c conda-forge

conda install ipykernel -c conda-forge
```

if you don't use Conda then I'm afraid you are too smart for me to help, but I have included a requirements file `requirements.txt` for my environment on Mac with an M1 chip.