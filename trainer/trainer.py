from utils.utils import *
from utils.loggers import *
from model.model import *
from data.preprocess import *

def test_run():
    x, y, (NUM, CAT, LABEL, cat_num) = get_data()
    the_model = K_graph(NUM, CAT, [LABEL], cat_num).to(DEVICE)
    optimizer = torch.optim.SGD(the_model.parameters(), lr=0.001)

    # optimizer.step()
    data_count = 5
    # random pick data
    indices = torch.randperm(len(x))[:data_count]
    train_data = x[indices]
    train_label = y[indices]

    for i in range(5):
        
        optimizer.zero_grad()
        output = the_model(train_data[:data_count], epoch=200)
        loss = torch.nn.functional.cross_entropy(output, train_label[:data_count])
        loss.backward()
        # print(((the_model.feature_importance_learners.grad).abs().max(dim=1)[0]))
        optimizer.step()
        
        print('-----------------------------------------')