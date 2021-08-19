# Unsupervised Attributed Multiplex Network Embedding (DMGI)

<p align="center">
    <a href="https://aaai.org/Conferences/AAAI-20/" alt="Conference">
        <img src="https://img.shields.io/badge/AAAI'20-brightgreen" /></a>   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>   
</p>

### Overview
Nodes in a multiplex network are connected by multiple types of relations. However, most existing network embedding methods assume that only a single type of relation exists between nodes. Even for those that consider the multiplexity of a network, they overlook node attributes, resort to node labels for training, and fail to model the global properties of a graph. We present a simple yet effective unsupervised network embedding method for attributed multiplex network called DMGI, inspired by Deep Graph Infomax (DGI) that maximizes the mutual information between local patches of a graph, and the global representation of the entire graph. We devise a systematic way to jointly integrate the node embeddings from multiple graphs by introducing 1) the consensus regularization framework that minimizes the disagreements among the relation-type specific node embeddings, and 2) the universal discriminator that discriminates true samples regardless of the relation types. We also show that the attention mechanism infers the importance of each relation type, and thus can be useful for filtering unnecessary relation types as a preprocessing step. Extensive experiments on various downstream tasks demonstrate that DMGI outperforms the state-of-the-art methods, even though DMGI is fully unsupervised.

### Paper
- [ **Unsupervised Attributed Multiplex Network Embedding (*AAAI 2020*)** ](https://arxiv.org/abs/1911.06750)
  - [_**Chanyoung Park**_](http://pcy1302.github.io), Donghyun Kim, Jiawei Han, Hwanjo Yu

### Requirements

- Python version: 3.6.8
- Pytorch version: 1.2.0
- networkx version: 2.3
  

### How to Run

```
git clone https://github.com/pcy1302/DMGI.git
cd DMGI
mkdir saved_model
```
- Download ``IMDB`` data from [ **here** ](https://www.dropbox.com/s/ntutrhk8nr3vveb/imdb.pkl?dl=0) to ``data``
```
python main.py --embedder DMGI --dataset imdb --metapaths MAM,MDM --isAttn
```
- Refer to the directory ``data`` for preprocessing for ``DBLP`` and ``Amazon`` datasets.

### Data format [(ex) IMDB]
- A dictionary containing the following keys
  - ``train_idx``: training index, ``val_idx``: validation index, ``test_idx``: test index, ``feature``: feature matrix, ``label``: labels
  - Relations: ``MDM``, ``MAM``

- <b>NEW (20/10/06): You can download all the preprocessed datasets used in the paper from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0)</b>
### Cite (Bibtex)
- If you find ``DMGI`` useful in your research, please cite the following paper:
  - Park, Chanyoung, Donghyun Kim, Jiawei Han, and Hwanjo Yu. "Unsupervised Attributed Multiplex Network Embedding." AAAI 2020.
  - Bibtex
```
@article{park2019unsupervised,
  title={Unsupervised Attributed Multiplex Network Embedding},
  author={Park, Chanyoung and Kim, Donghyun and Han, Jiawei and Yu, Hwanjo},
  booktitle={AAAI},
  year={2020}
}
```
