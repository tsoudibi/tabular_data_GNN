from utils import *

class MLP(torch.nn.Module):
    def __init__(self, NUM, CAT, LABEL, cat_num):
        super(MLP, self).__init__()
        '''
        num_cols: number of numerical columns
        cat_cols: number of categorical columns
        label_cols: number of label columns
        cat_num: number of unique value of each categorical columns
        '''
        self.hidden_dim = 128
        
        # order: num -> cat -> label
        self.num_cols = len(NUM)
        self.cat_cols = len(CAT)
        self.label_cols = len(LABEL)
        self.number_of_columns = self.num_cols + self.cat_cols 
        
        
        # numerical feature
        self.num_embeddings = torch.nn.ModuleList([torch.nn.Linear(1, self.hidden_dim) for i in range(self.num_cols)])
        # categorical feature
        self.cat_embeddings = torch.nn.ModuleList([torch.nn.Embedding(cat_num[i], self.hidden_dim) for i in range(self.cat_cols)])
        
        self.prediction = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * self.number_of_columns, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.label_cols + 1)
        )
        
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
        feature_embedding = torch.cat((feature_embedding_num, feature_embedding_cat), dim=1)
        
        
        # make prediction
        prediction = self.prediction(feature_embedding)
        
        return prediction