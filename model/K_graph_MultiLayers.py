from utils.loggers import *
from utils.utils import *
from data.preprocess import *
import torch
import torch_geometric
from torch_geometric.data import Data
import os

DEVICE = get_device()

class pyrimid_layers_config():
    def __init__(self):
        '''
        check if the pyrimid layers config is valid and caculate the number of columns of each layer
        
        for example:
            total_columns = 20
            layers_prop = [0.5, 0.3, 0.2]
        then:
            IO_of_each_layer = [(20, 10), (10, 6), (6, 4)]
            total_layers = 3
        '''
        # check if the config exists
        if get_run_config()['layers_prop'] is None:
            raise ValueError('layers_prop is not set in the config, please check config file.')
        elif get_run_config()['num_layers'] != len(get_run_config()['layers_prop']):
            raise ValueError('num_layers is not match with layers_prop, please check config file.',get_run_config()['num_layers'],len(get_run_config()['layers_prop']))
        # check if the config is valid and caculate the number of columns of each layer
        NUM, CAT, _, _ = get_colunm_info()
        total_columns = len(NUM) + len(CAT)

        # layers in and out
        IO_of_each_layer = []
        layers_prop = get_run_config()['layers_prop'].copy()
        layers_prop.insert(0, 1)
        for index in range(len(layers_prop)-1):
            IO_of_each_layer.append((int(total_columns * layers_prop[index]), int(total_columns * layers_prop[index+1])))
            if IO_of_each_layer[-1][0] <= 0 or IO_of_each_layer[-1][1] <= 0:
                raise ValueError('in layer '+ str(index) + ' appear zero column input/output, please check layers_prop setting.', IO_of_each_layer)
        self.IO_of_each_layer = IO_of_each_layer
        self.total_layers = len(IO_of_each_layer)
        # print(IO_of_each_layer)
        
    def pyrimid_layers(self):
        return self.columns_of_each_layer
    
        
class feature_embedding(torch.nn.Module):
    def __init__(self, NUM, CAT, LABEL, cat_num):
        super(feature_embedding, self).__init__()
        '''
        transform input feature to embedding
        
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
        
        # numerical feature
        self.num_embeddings = torch.nn.ModuleList([torch.nn.Linear(1, self.hidden_dim) for i in range(self.num_cols)])
        # categorical feature
        self.cat_embeddings = torch.nn.ModuleList([torch.nn.Embedding(cat_num[i], self.hidden_dim) for i in range(self.cat_cols)])
        
        # batch norm
        self.batch_norm = torch.nn.BatchNorm1d(self.number_of_columns * self.hidden_dim)

    def forward(self, input_data):
        # make feature embedding
        num_data = input_data[:,:self.num_cols].unsqueeze(-1).unsqueeze(-1) 
        feature_embedding_num = torch.cat([self.num_embeddings[i](num_data[:,i]) for i in range(self.num_cols)], dim=1).reshape(len(input_data), -1) # [batch_size, num_cols * hidden_dim]
        feature_embedding_num = torch.nn.ReLU()(feature_embedding_num)
        # feature_embedding_num = torch.layer_norm(feature_embedding_num, feature_embedding_num.shape)
        # categorical feature
        if self.cat_cols != 0:
            feature_embedding_cat = torch.cat([self.cat_embeddings[i](input_data[:,self.num_cols+i].long()) for i in range(self.cat_cols)], dim=1) # [batch_size, cat_cols * hidden_dim]
            # feature_embedding_cat = torch.layer_norm(feature_embedding_cat, feature_embedding_cat.shape)
        # concat
        if self.cat_cols != 0:
            feature_embedding = torch.cat((feature_embedding_num, feature_embedding_cat), dim=1) # [batch_size, (num_cols + cat_cols) * hidden_dim]
            # feature_embedding = torch.layer_norm(feature_embedding, feature_embedding.shape)
            feature_embedding = self.batch_norm(feature_embedding)
            feature_embedding = feature_embedding.reshape(len(input_data), self.number_of_columns, -1)
            del feature_embedding_num, feature_embedding_cat, num_data
        else:
            feature_embedding = feature_embedding_num
            # feature_embedding = torch.layer_norm(feature_embedding, feature_embedding.shape)
            feature_embedding = self.batch_norm(feature_embedding)
            feature_embedding = feature_embedding.reshape(len(input_data), self.number_of_columns, -1)
            del feature_embedding_num, num_data
        
        return feature_embedding
    
    
class K_graph_Layer(torch.nn.Module):
    def __init__(self, C_input, C_output, hidden_dim):
        super(K_graph_Layer, self).__init__()
        '''
        include "feature importance learner" and "K graph cell"
        
        C_input: number of input feature (C)
        C_output: number of output feature (C'), that is, K
        hidden_dim: hidden dimension
        '''
        self.C_input = C_input
        self.C_output = C_output
        self.hidden_dim = hidden_dim
        
        
        # feature importance learning
        # self.num_learner = get_run_config()['num_learner']
        self.num_learner = 1
        self.feature_importance_learners = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.hidden_dim, 1),
        )  for i in range(self.num_learner)]).to(DEVICE)
        
        # graph convolution layers
        # self.GNNs = torch.nn.ModuleList([torch_geometric.nn.GCNConv(self.hidden_dim, self.hidden_dim) for i in range(self.C_input)]).to(DEVICE)
        self.SAGEs = torch.nn.ModuleList([torch_geometric.nn.SAGEConv(self.hidden_dim, self.hidden_dim) for i in range(self.C_input)]).to(DEVICE)
        # self.GATs = torch.nn.ModuleList([torch_geometric.nn.GATv2Conv(self.hidden_dim, self.hidden_dim) for i in range(self.C_input)]).to(DEVICE)
        # self.conv_GCN_input = torch_geometric.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        
        # batch norm
        self.batch_norm_GNN = torch.nn.BatchNorm1d(self.hidden_dim)
        
    def forward(self, input_embedding, epoch = -1):
        
        input_embedding = input_embedding.reshape(len(input_embedding), -1) # [batch_size, C_input * hidden_dim]
        # feature importance learning
        feature_importance = []
        for learner_index in range(self.num_learner):
            feature_importance.append(torch.cat([self.feature_importance_learners[learner_index](input_embedding[:,i*self.hidden_dim:(i+1)*self.hidden_dim]) for i in range(self.C_input)], dim=1)) # [batch_size, num_cols + cat_cols, 1]
        feature_importance = torch.stack(feature_importance, dim=1) # [batch_size, num_learner, num_cols + cat_cols]
        # feature_importance = torch.layer_norm(feature_importance, feature_importance.shape)
        feature_importance = torch.softmax(feature_importance, dim=2) # [batch_size, num_learner, num_cols + cat_cols]
        # print(feature_importance.shape)
        feature_importance = feature_importance.squeeze(1) # [batch_size, num_cols + cat_cols]
        # print(feature_importance.shape)
        # print(input_embedding.shape)
        
        # weighted feature embedding 
        feature_embedding = input_embedding.reshape((len(input_embedding),self.C_input, -1)) * feature_importance.unsqueeze(-1) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        feature_embedding = feature_embedding.reshape((len(input_embedding), -1)) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        
        
        # top K feature importance
        value, indices = torch.topk(feature_importance, self.C_output) # (value: [batch_size, k], indices: [batch_size, k])
        mask = torch.zeros_like(feature_importance, device=DEVICE)
        mask.scatter_(1, indices, 1)
        importance_topK = torch.where(mask > 0, feature_importance, torch.empty(feature_importance.shape,device=DEVICE).fill_(-1e9)) # [batch_size, cols]
        importance_topK = torch.softmax(importance_topK, dim=1) # [batch_size, cols]
        
        # extractor.update(feature_importance.sum(dim=0)/len(input_data))
        del mask, feature_importance, value, indices
        torch.cuda.empty_cache()
        processed_data = []
        processed_indices = []
        for target_col in range(self.C_input):
            importance_topK_current = importance_topK.clone()# [batch_size, cols] 
            indices = importance_topK_current.T[target_col].nonzero().T[0] # selected samples' indices  
            
            if indices.shape[0] == 0:
                continue
            
            importance_topK_current = importance_topK_current[importance_topK_current.T[target_col]>0]# [????, cols]
            
            # for target column, set its importance to 0. so that it will not be fully connected graph
            # copy target column
            importance_topK_current[:,target_col] = 0 # [????, cols]
            # multiply to get weighted adj
            weighted_adj = torch.matmul(importance_topK_current, importance_topK_current.T) # [batch_size, cols] * [cols, batch_size] = [batch_size, batch_size]
            # threshold
            # print(weighted_adj.mean(), weighted_adj.max(), weighted_adj.min())
            # weighted_adj = torch.where(weighted_adj > 0.5, weighted_adj, torch.zeros_like(weighted_adj, device=DEVICE))
            # prune the diagonal
            # weighted_adj = weighted_adj - torch.diag(weighted_adj.diagonal())

            # construct graph
            edge_index = weighted_adj.nonzero().T  # [2, num_edges]
            edge_weight = weighted_adj[edge_index[0], edge_index[1]] # [num_edges]
            # normalize to 0~1
            # if len(edge_weight) > 0:
                # edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-9)
                # threshold 
                # edge_weight = torch.where(edge_weight > 0.5, edge_weight, torch.zeros_like(edge_weight, device=DEVICE))
            # print(edge_weight.shape, edge_weight)
            # print('max',max(edge_weight), 'min',min(edge_weight), 'mean',edge_weight.mean())

            del weighted_adj, importance_topK_current
            
            # if True and epoch % 10 == 0:
            #     print('in graph', target_col, 'nodes:', len(indices), 'edges:', len(edge_weight),'ratio', len(edge_weight)/(len(indices)**2+0.000001))
            
            
            # features = (feature_embedding[indices]) # [????, cols*hidden_dim]
            features = (feature_embedding.reshape(len(input_embedding),self.C_input,-1)[indices][:,target_col,:]) # [????, hidden_dim]

            # construct graph 
            data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight) 
            del features, edge_index, edge_weight
            torch.cuda.empty_cache()
            
            # apply GCN
            # x = self.GNNs[target_col](data.x, data.edge_index, data.edge_weight)  # [???, hidden_dim]
            # x = self.GNNs[target_col](data.x, data.edge_index)  # [???, hidden_dim]
            x = self.SAGEs[target_col](data.x, data.edge_index)  # [???, hidden_dim]
            # x = self.GATs[target_col](data.x, data.edge_index)  # [???, hidden_dim]
            x = torch.relu(x)
            x = torch.layer_norm(x, x.shape) # [???, hidden_dim]
            # x = self.batch_norm_GNN(x)
            # x = torch.nn.Dropout(p=0.5)(x)
            # x = self.conv_GCN_2(x, data.edge_index, data.edge_weight)  # [???, hidden_dim]
            # x = torch.relu(x)
            # x = torch.layer_norm(x, x.shape)

            processed_data.append(x)
            processed_indices.append(indices)
            del data, x
        
        processed_data = torch.cat(processed_data, dim=0) 
        processed_indices = torch.cat(processed_indices, dim=0)

        processed_data = processed_data[processed_indices.argsort()] # ???????
        processed_data = torch.split(processed_data, self.C_output) # ?????????
        processed_data = torch.stack(list(processed_data), dim=0) # ???????????
        # processed_data = torch.sum(list(processed_data), dim=0) # ???????????


        
        return processed_data # [batch_size, C_output, hidden_dim]
        # return prediction
    
class K_graph_MultiLayers(torch.nn.Module):
    def __init__(self, NUM, CAT, LABEL, cat_num):
        super(K_graph_MultiLayers, self).__init__()
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
        
        self.pyr_layer = pyrimid_layers_config()
            
        # feature embedding
        self.feature_embedding_learner = feature_embedding(NUM, CAT, LABEL, cat_num)
        
        self.K_graph_layers = []
        for layer_index in range(self.pyr_layer.total_layers):
            self.K_graph_layers.append(K_graph_Layer(self.pyr_layer.IO_of_each_layer[layer_index][0], self.pyr_layer.IO_of_each_layer[layer_index][1], self.hidden_dim))
   
        self.prediction_CAT = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim *(  
                                                int(self.number_of_columns) + 
                                                sum([self.pyr_layer.IO_of_each_layer[i][1] for i in range(self.pyr_layer.total_layers)])
                                              ),
                                            # ex: 20(res) + 10(layer1) + 6(layer2) + 4(layer3)
                            self.hidden_dim * 1),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim * 1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(self.hidden_dim, self.label_cols + 1 )
        )
        
    def forward(self, input_data, epoch = -1):
        
        # feature embedding
        feature_embedding = self.feature_embedding_learner(input_data) # [batch_size, (num_cols + cat_cols) * hidden_dim]
        
        feature_embeddings = []
        feature_embeddings.append(feature_embedding) # [batch_size, C, hidden_dim]
        for layer_index in range(self.pyr_layer.total_layers):
            feature_embeddings.append(self.K_graph_layers[layer_index](feature_embeddings[layer_index], epoch)) # [batch_size, C', hidden_dim]
        
 
        # prediction
        for i in range(len(feature_embeddings)):
            feature_embeddings[i] = feature_embeddings[i].reshape(len(input_data), -1)
        prediction_CAT = self.prediction_CAT(torch.cat(feature_embeddings, dim=1))
        
        return prediction_CAT