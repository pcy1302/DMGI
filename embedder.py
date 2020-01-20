import time
import numpy as np
import torch
from utils import process
import torch.nn as nn
from layers import AvgReadout

class embedder:
    def __init__(self, args):
        args.batch_size = 1
        args.sparse = True
        args.metapaths_list = args.metapaths.split(",")
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        adj, features, labels, idx_train, idx_val, idx_test = process.load_data_dblp(args)
        features = [process.preprocess_features(feature) for feature in features]

        args.nb_nodes = features[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = labels.shape[1]
        args.nb_graphs = len(adj)
        args.adj = adj
        adj = [process.normalize_adj(adj_) for adj_ in adj]
        self.adj = [process.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj]

        self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features]

        self.labels = torch.FloatTensor(labels[np.newaxis]).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
        self.val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
        self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)

        # How to aggregate
        args.readout_func = AvgReadout()

        # Summary aggregation
        args.readout_act_func = nn.Sigmoid()

        self.args = args

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s
