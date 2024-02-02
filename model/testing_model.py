from utils.loggers import *
from utils.utils import *
import torch
import torch_geometric
from torch_geometric.data import Data
import os
DEVICE = get_device()
class One_graph(torch.nn.Module):
    def __init__(self, NUM, CAT, LABEL, cat_num):
        super(One_graph, self).__init__()
        '''
        make one graph for all samples
        edge connection is based on feature importance pattern similarity(thresholded by 0.5)
        
        num_cols: number of numerical columns
        cat_cols: number of categorical columns
        label_cols: number of label columns
        cat_num: number of unique value of each categorical columns
        '''
        self.hidden_dim = get_run_config()['hidden_dim']
        # order: num -> cat -> label
        self.num_cols = len(NUM)
        self.cat_cols = len(CAT)
        self.label_cols = len(LABEL)
        self.number_of_columns = self.num_cols + self.cat_cols 
        self.K = round(self.number_of_columns*get_run_config()['K_ratio'])
        
        # numerical feature
        self.num_embeddings = torch.nn.ModuleList([torch.nn.Linear(1, self.hidden_dim) for i in range(self.num_cols)])
        # categorical feature
        self.cat_embeddings = torch.nn.ModuleList([torch.nn.Embedding(cat_num[i], self.hidden_dim) for i in range(self.cat_cols)])
        
        self.prediction = torch.nn.Sequential(
            # torch.nn.Linear(self.hidden_dim *( 1 ) , self.hidden_dim),
            torch.nn.Linear(self.hidden_dim *(self.number_of_columns + 1 ) , self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.label_cols + 1)
        )
        
        # feature importance learning
        self.num_learner = get_run_config()['num_learner']
        self.feature_importance_learners = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.hidden_dim, 1),
        )  for i in range(self.num_learner)])
        
        # # graph convolution layers
        self.conv_GCN_input = torch_geometric.nn.GCNConv(self.number_of_columns*self.hidden_dim, self.hidden_dim)
        # self.conv_GCN_input = torch_geometric.nn.GCNConv(self.number_of_columns*self.hidden_dim, self.number_of_columns*self.hidden_dim)
        # # self.conv_GCN_input = torch_geometric.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        # # self.conv_1_input = torch_geometric.nn.GATConv(self.number_of_columns*self.hidden_dim, self.hidden_dim)
        # self.conv_GCN_2 = torch_geometric.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        
        self.transform = torch.nn.Linear(self.number_of_columns*self.hidden_dim, self.hidden_dim)
        
        self.weight_feature_importance = torch.nn.Sequential(
            torch.nn.Linear(self.number_of_columns*self.num_learner, self.num_learner),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.num_learner),
        )
        
    def forward(self, input_data, epoch = -1):
        
        # make feature embedding
        num_data = input_data[:,:self.num_cols].unsqueeze(-1).unsqueeze(-1) 
        feature_embedding_num = torch.cat([self.num_embeddings[i](num_data[:,i]) for i in range(self.num_cols)], dim=1).reshape(len(input_data), -1) # [batch_size, num_cols * hidden_dim]
        feature_embedding_num = torch.nn.ReLU()(feature_embedding_num)
        feature_embedding_num = torch.layer_norm(feature_embedding_num, feature_embedding_num.shape)
        # categorical feature
        if self.cat_cols != 0:
            feature_embedding_cat = torch.cat([self.cat_embeddings[i](input_data[:,self.num_cols+i].long()) for i in range(self.cat_cols)], dim=1) # [batch_size, cat_cols * hidden_dim]
            feature_embedding_cat = torch.layer_norm(feature_embedding_cat, feature_embedding_cat.shape)
        # concat
        if self.cat_cols != 0:
            feature_embedding = torch.cat((feature_embedding_num, feature_embedding_cat), dim=1) # [batch_size, (num_cols + cat_cols) * hidden_dim]
            del feature_embedding_num, feature_embedding_cat, num_data
        else:
            feature_embedding = feature_embedding_num
            del feature_embedding_num, num_data
        
        # feature importance learning
        feature_importance = []
        for learner_index in range(self.num_learner):
            feature_importance.append(torch.cat([self.feature_importance_learners[learner_index](feature_embedding[:,i*self.hidden_dim:(i+1)*self.hidden_dim]) for i in range(self.number_of_columns)], dim=1)) # [batch_size, num_cols + cat_cols, 1]
        feature_importance = torch.stack(feature_importance, dim=1) # [batch_size, 128, num_cols + cat_cols]
        feature_importance = torch.layer_norm(feature_importance, feature_importance.shape)
        # print(feature_importance.shape)
        
        # weighted feature importance
        feature_importance_weight = self.weight_feature_importance(feature_importance.reshape(len(input_data),-1)) # [batch_size, 128]
        feature_importance_weight = torch.softmax(feature_importance_weight, dim=1) # [batch_size, 128]
        # weighted sum
        feature_importance = torch.sum(feature_importance * feature_importance_weight.unsqueeze(-1), dim=1) # [batch_size, num_cols + cat_cols]
        # print(feature_importance.shape)
        
        # weighted feature embedding 
        feature_embedding = feature_embedding.reshape((len(input_data),self.number_of_columns, -1)) * feature_importance.unsqueeze(-1) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        feature_embedding = feature_embedding.reshape((len(input_data), -1)) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        
        # construct graph
        weighted_adj = torch.matmul(feature_importance, feature_importance.T) # [batch_size, cols] * [cols, batch_size] = [batch_size, batch_size]
        # # prune the diagonal
        # weighted_adj = weighted_adj - torch.diag(weighted_adj.diagonal())
        # threshold
        weighted_adj = torch.where(weighted_adj > 0.5, weighted_adj, torch.zeros_like(weighted_adj, device=DEVICE))

        # construct graph
        edge_index = weighted_adj.nonzero().T  # [2, num_edges]
        edge_wight = weighted_adj[edge_index[0], edge_index[1]] # [num_edges]
        edge_wight = torch.softmax(edge_wight, dim=0)
        
        # construct graph 
        data = Data(x=feature_embedding, edge_index=edge_index, edge_weight=edge_wight) 
        del edge_index, edge_wight, weighted_adj, feature_importance_weight, feature_importance, feature_embedding
        torch.cuda.empty_cache()
        
        # apply GCN
        x = self.conv_GCN_input(data.x, data.edge_index, data.edge_weight)  # [barch_size, hidden_dim]
        x = torch.relu(x)
        x = torch.layer_norm(x, x.shape) # [barch_size, hidden_dim]
        # x = self.conv_GCN_2(x, data.edge_index, data.edge_weight)  # [barch_size, hidden_dim]
        # x = torch.relu(x)
        # x = torch.layer_norm(x, x.shape)

        # residual connection
        # x = x + self.transform(data.x)
        x = torch.cat((x, data.x), dim=1) # [barch_size, (hidden_dim + (num_cols + cat_cols) * hidden_dim)]
        
        # make prediction
        prediction = self.prediction(x)
        
        return prediction
   