from utils.loggers import *
from utils.utils import *
import torch
import torch_geometric
from torch_geometric.data import Data
import os


# CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")
# AVAIL_GPUS = min(1, torch.cuda.device_count())

DEVICE = get_device()

class K_graph(torch.nn.Module):
    def __init__(self, NUM, CAT, LABEL, cat_num):
        super(K_graph, self).__init__()
        '''
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
            torch.nn.Linear(self.hidden_dim *( self.K + self.number_of_columns), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.label_cols + 1)
        )
        
        # feature importance learning
        self.feature_importance_learners = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.hidden_dim, 1),
        ) 
        
        # graph convolution layers
        self.conv_GCN_input = torch_geometric.nn.GCNConv(self.number_of_columns*self.hidden_dim, self.hidden_dim)
        # self.conv_GCN_input = torch_geometric.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        # self.conv_1_input = torch_geometric.nn.GATConv(self.number_of_columns*self.hidden_dim, self.hidden_dim)
        self.conv_GCN_2 = torch_geometric.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        
        # self.transform = torch.nn.Linear(self.number_of_columns*self.hidden_dim, self.hidden_dim)
        
    def forward(self, input_data, epoch = -1):
        
        # make feature embedding
        num_data = input_data[:,:self.num_cols].unsqueeze(-1).unsqueeze(-1) 
        feature_embedding_num = torch.cat([self.num_embeddings[i](num_data[:,i]) for i in range(self.num_cols)], dim=1).reshape(len(input_data), -1) # [batch_size, num_cols * hidden_dim]
        feature_embedding_num = torch.nn.ReLU()(feature_embedding_num)
        feature_embedding_num = torch.layer_norm(feature_embedding_num, feature_embedding_num.shape)
        # categorical feature
        feature_embedding_cat = torch.cat([self.cat_embeddings[i](input_data[:,self.num_cols+i].long()) for i in range(self.cat_cols)], dim=1) # [batch_size, cat_cols * hidden_dim]
        feature_embedding_cat = torch.layer_norm(feature_embedding_cat, feature_embedding_cat.shape)
        # concat
        feature_embedding = torch.cat((feature_embedding_num, feature_embedding_cat), dim=1) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        # feature_embedding = feature_embedding.reshape((len(input_data), self.number_of_columns, -1)) # [batch_size, (num_cols + cat_cols), hidden_dim]
        
        # feature importance learning
        feature_importance = torch.cat([self.feature_importance_learners(feature_embedding[:,i*self.hidden_dim:(i+1)*self.hidden_dim]) for i in range(self.number_of_columns)], dim=1) # [batch_size, num_cols + cat_cols, 1]
        # print(feature_importance)
        feature_importance = torch.layer_norm(feature_importance, feature_importance.shape)
        # feature_importance = torch.softmax(feature_importance, dim=1) # [batch_size, num_cols + cat_cols, 1]
        # print(feature_importance.shape)
        # print(feature_importance.sum(dim=1))
        # print(feature_importance)
        
        # weighted feature embedding 
        feature_embedding = feature_embedding.reshape((len(input_data),self.number_of_columns, -1)) * feature_importance.unsqueeze(-1) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        feature_embedding = feature_embedding.reshape((len(input_data), -1)) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        
        # top K feature importance
        K = self.K
        value, indices = torch.topk(feature_importance, K) # (value: [batch_size, k], indices: [batch_size, k])
        mask = torch.zeros_like(feature_importance, device=DEVICE)
        mask.scatter_(1, indices, 1)
        # importance_topK = torch.where(mask > 0, feature_importance, torch.zeros(feature_importance.shape,device=DEVICE)) # [batch_size, cols]
        importance_topK = torch.where(mask > 0, feature_importance, torch.empty(feature_importance.shape,device=DEVICE).fill_(-1e9)) # [batch_size, cols]
        importance_topK = torch.softmax(importance_topK, dim=1) # [batch_size, cols]
        # importance_topK = torch.stack([importance_topK.clone() for _ in range(self.number_of_columns)], dim=0) # [cols, batch_size, cols]
        
        extractor.update(feature_importance.sum(dim=0)/len(input_data))
        del feature_embedding_num, feature_embedding_cat, num_data
        del mask, feature_importance, value, indices
        
        
        processed_data = []
        processed_indices = []
        for target_col in range(self.number_of_columns):
            importance_topK_current = importance_topK.clone()# [batch_size, cols] 
            indices = importance_topK_current.T[target_col].nonzero().T[0] # selected samples' indices  
            
            if indices.shape[0] == 0:
                continue
            
            importance_topK_current = importance_topK_current[importance_topK_current.T[target_col]>0]# [????, cols]
            
            # for target column, set its importance to 0. so that it will not be fully connected graph
            # copy target column
            tmp = torch.clone(importance_topK_current[:,target_col]) # [????], save for future weighted sum
            importance_topK_current[:,target_col] = 0 # [????, cols]
            # multiply to get weighted adj
            weighted_adj = torch.matmul(importance_topK_current, importance_topK_current.T) # [batch_size, cols] * [cols, batch_size] = [batch_size, batch_size]
            # prune the diagonal
            weighted_adj = weighted_adj - torch.diag(weighted_adj.diagonal())

            # construct graph
            edge_index = weighted_adj.nonzero().T  # [2, num_edges]
            edge_wight = weighted_adj[edge_index[0], edge_index[1]] # [num_edges]
            edge_wight = torch.softmax(edge_wight, dim=0)

            
            if False:
                print('in graph', target_col, 'nodes:', len(indices), 'edges:', len(edge_wight),'ratio', len(edge_wight)/(len(indices)**2+0.000001))
            
            # print(edge_wight)
            # importance_topK_current[:,target_col] = tmp # [????, cols]
            
            features = (feature_embedding[indices]) # [????, cols*hidden_dim]
            # features = (feature_embedding.reshape(len(input_data),self.number_of_columns,-1)[indices][:,target_col,:]) # [????, hidden_dim]
            # print(features.shape)

            # construct graph 
            data = Data(x=features, edge_index=edge_index, edge_weight=edge_wight, indices=indices) 
            
            del features, edge_index, edge_wight, weighted_adj, importance_topK_current, tmp
            
            # apply GCN
            x = self.conv_GCN_input(data.x, data.edge_index, data.edge_weight)  # [???, hidden_dim]
            # x = self.conv_1_input(data.x, data.edge_index)  # [???, hidden_dim]
            x = torch.relu(x)
            x = torch.layer_norm(x, x.shape) # [???, hidden_dim]
            x = torch.nn.Dropout(p=0.5)(x)
            x = self.conv_GCN_2(x, data.edge_index, data.edge_weight)  # [???, hidden_dim]
            x = torch.relu(x)
            x = torch.layer_norm(x, x.shape)

            processed_data.append(x)
            processed_indices.append(indices)
        
        processed_data = torch.cat(processed_data, dim=0) 
        processed_indices = torch.cat(processed_indices, dim=0)

        processed_data = processed_data[processed_indices.argsort()] # ???????
        processed_data = torch.split(processed_data, self.K) # ?????????
        processed_data = torch.stack(list(processed_data), dim=0) # ???????????
        # processed_data = torch.sum(list(processed_data), dim=0) # ???????????

        # cat residual
        processed_data = torch.cat((processed_data, feature_embedding.reshape((len(input_data),self.number_of_columns,-1))), dim=1) # [batch_size, K+cols , hidden_dim]
        
        # make prediction
        prediction = self.prediction(processed_data.reshape(processed_data.shape[0],-1))
        # prediction = self.prediction(feature_embedding)
        
        
        return prediction