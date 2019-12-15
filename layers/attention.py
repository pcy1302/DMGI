import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hid_units, 1) for _ in range(args.nb_graphs)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.nb_graphs):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos, feat_neg, summary):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        summary, summary_attn = self.attn_summary(summary)

        return feat_pos, feat_neg, summary


    def attn_feature(self, features):
        features_attn = []
        for i in range(self.args.nb_graphs):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features,1).squeeze(0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.nb_graphs, self.args.nb_nodes, self.args.hid_units).sum(0).unsqueeze(0)

        return features, features_attn

    def attn_summary(self, features):
        features_attn = []
        for i in range(self.args.nb_graphs):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1).unsqueeze(1)
        features = torch.cat(features, 0)
        features_attn_expanded = features_attn.expand_as(features)
        features = (features * features_attn_expanded).sum(0).unsqueeze(0)

        return features, features_attn
