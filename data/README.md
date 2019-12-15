# Preprocessing datasets

### How to Run (DBLP dataset)

- Download DBLP-Citation-network V8 dataset from https://www.aminer.cn/citation, and name the downloaded file ``dblp.txt``
- Preprocess DBLP dataset (num_trai: number of labels for training)
```
python preprocess_dblp.py num_train
```
- Note that ACM dataset can be downloaded from https://github.com/Jhy1993/HAN/tree/master/data

### Data format
- A dictionary containing the following keys
  - ``train_idx``: training index, ``val_idx``: validation index, ``test_idx``: test index, ``feature``: feature matrix, ``label``: labels
  - Relations
    - IMDB: ``MDM``, ``MAM`` 
    - DBLP: ``PAP``, ``PPP``, ``PATAP``

#### Please note that the preprocessing code is not well organized.

