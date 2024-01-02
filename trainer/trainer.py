from utils.utils import *
from utils.loggers import *
from model.model import *
from model.baseline import *
from data.preprocess import *
from tqdm import trange
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import BinaryAUROC
from sklearn.model_selection import KFold

def test_run():
    '''
    run a test run of training, check if the model can be trained
    '''
    set_seed(get_run_config()['random_state'])
    x, y, (NUM, CAT, LABEL, cat_num) = get_data()
    the_model = K_graph(NUM, CAT, [LABEL], cat_num).to(DEVICE)
    # the_model = K_graph_Multi(NUM, CAT, [LABEL], cat_num).to(DEVICE)
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
        print(((the_model.feature_importance_learners[-1].weight.grad).abs().max(dim=1)[0]))
        optimizer.step()
        
        print('epoch',i,'-----------------------------------------')
    print('done')
        

def train_one_epoch(model, optimizer, datas, batch_size, epoch):
    '''
    train one epoch\n
    input:
        - model: model to train
        - optimizer: optimizer to use
        - datas: (train_data, train_label, validation_data, validation_label)
        - batch_size: batch size
        - epoch: current epoch
    '''
    train_data, train_label, validation_data, validation_label = datas
    
    # slice data into batch
    train_data = torch.split(train_data, batch_size)
    train_label = torch.split(train_label, batch_size)
    validation_data = torch.split(validation_data, batch_size)
    validation_label = torch.split(validation_label, batch_size)

    # losses and metrics
    batch_loss = 0
    train_acc = MulticlassAccuracy(num_classes=2).to(DEVICE)
    train_auc = BinaryAUROC().to(DEVICE)
    valid_acc = MulticlassAccuracy(num_classes=2).to(DEVICE)
    valid_auc = BinaryAUROC().to(DEVICE)
    
    # train the model
    stepper = trange(len(train_data))
    for i in stepper:
        stepper.set_description(f'Epoch {epoch}')
        
        optimizer.zero_grad()
        output = model(train_data[i], epoch=epoch)
        loss = torch.nn.functional.cross_entropy(output, train_label[i]) * model.number_of_columns
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        
        # metrics
        preds = output.softmax(dim=1)
        true = torch.nn.functional.one_hot(train_label[i], num_classes=2).to(DEVICE)
        train_acc.update(torch.argmax(preds, 1),true.T[1])
        train_auc.update(preds.T[0],true.T[0])
        
        # at the end of epoch, print result and validate the model
        if i == len(train_data) - 1:
            train_acc = train_acc.compute()
            train_auc = train_auc.compute()
            stepper.set_postfix({'loss': round(batch_loss/(i+1), 3), 'acc': round(train_acc.item(), 3), 'AUC': round(train_auc.item(), 3)})
            stepper.update()
        
            with torch.no_grad():
                for i in range(len(validation_data)):
                    output = model(validation_data[i], epoch=-1)
                    # loss = torch.nn.functional.cross_entropy(output, validation_label[i])
                    preds = output.softmax(dim=1)
                    true = torch.nn.functional.one_hot(validation_label[i], num_classes=2).to(DEVICE)
                    valid_acc.update(torch.argmax(preds,1),true.T[1])
                    valid_auc.update(preds.T[0],true.T[0])
                stepper.set_postfix({'loss': round(batch_loss/(i+1), 3), 'acc': round(train_acc.item(), 3), 'AUC': round(train_auc.item(), 3), 'val_acc': round(valid_acc.compute().item(), 3), 'val_AUC': round(valid_auc.compute().item(), 3)})
                if get_wandb_config()['use_wandb']:
                    get_logger().log({'loss': round(batch_loss/(i+1), 3), 'acc': round(train_acc.item(), 3), 'AUC': round(train_auc.item(), 3), 'val_acc': round(valid_acc.compute().item(), 3), 'val_AUC': round(valid_auc.compute().item(), 3)})

            
def gather_data_package() ->(tuple):
    '''
    make data package for training when not specify data_package\n
    will use configs in 'config.yaml'\n
    return: (train_data, train_label, validation_data, validation_label, test_data, test_label, NUM, CAT, LABEL, cat_num)
    '''
    
    x, y, (NUM, CAT, LABEL, cat_num) = get_data()
    
    train_ratio = get_run_config()['train_ratio']
    validation_ratio = get_run_config()['validation_ratio']
    
    # shuffle data
    indices = torch.randperm(len(x))
    x = x[indices]
    y = y[indices]
    # slice data into train and test and validation
    train_data = x[:int(len(x)*train_ratio)]
    train_label = y[:int(len(x)*train_ratio)]
    validation_data = x[int(len(x)*train_ratio):int(len(x)*(train_ratio+validation_ratio))]
    validation_label = y[int(len(x)*train_ratio):int(len(x)*(train_ratio+validation_ratio))]
    test_data = x[int(len(x)*(train_ratio+validation_ratio)):]
    test_label = y[int(len(x)*(train_ratio+validation_ratio)):]
    return (train_data, train_label, validation_data, validation_label, test_data, test_label, NUM, CAT, LABEL, cat_num)

def train_one_run(configs, data_package = None):
    '''
    run one run of training, including train, validation and test\n
    Noted: if data_package is None, will automatically generate data_package from configs (shouldn't be used when using K fold cross validation)
    inputs:
        - configs: dict, run config of this run, use get_run_config() to get it
        - data_package: (train_data, train_label, validation_data, validation_label, test_data, test_label, NUM, CAT, LABEL, cat_num)
    '''
    if get_wandb_config()['use_wandb']:
        set_logger(wandb_logger(get_wandb_config()))
    
    # load configs hyperparameter
    max_epoch = configs['max_epoch']
    learning_rate = configs['learning_rate']
    batch_size = configs['batch_size']
    
    if data_package == None:
        data_package = gather_data_package()
    train_data, train_label, validation_data, validation_label, test_data, test_label, NUM, CAT, LABEL, cat_num  = data_package
    
    # build model and optimizer
    # the_model = K_graph(NUM, CAT, [LABEL], cat_num).to(DEVICE)
    the_model = K_graph_Multi(NUM, CAT, [LABEL], cat_num).to(DEVICE)
    # the_model = MLP(NUM, CAT, [LABEL], cat_num).to(DEVICE)
    optimizer = torch.optim.SGD(the_model.parameters(), lr = learning_rate)
    
    # train the model
    datas = (train_data, train_label, validation_data, validation_label)
    for i in range(max_epoch):
        train_one_epoch(the_model, optimizer, datas, batch_size, epoch=i+1)
        print(extractor.get())
        extractor.reset()
    
    # test the model
    with torch.no_grad():
        test_data = torch.split(test_data, batch_size)
        test_label = torch.split(test_label, batch_size)
        for i in range(len(test_data)):
            output = the_model(test_data[i], epoch=-1)
            preds = output.softmax(dim=1)
            true = torch.nn.functional.one_hot(test_label[i], num_classes=2).to(DEVICE)
            test_acc = MulticlassAccuracy(num_classes=2).to(DEVICE)
            test_auc = BinaryAUROC().to(DEVICE)
            test_acc.update(torch.argmax(preds,1),true.T[1])
            test_auc.update(preds.T[0],true.T[0])

        print('test_acc:', test_acc.compute().item())
        print('test_auc:', test_auc.compute().item())
        print('-----------------------------------------')
        if get_wandb_config()['use_wandb']:
            get_logger().log({'test_acc': test_acc.compute().item(), 'test_auc': test_auc.compute().item()})
            get_logger().finish()
            
            
def train_K_fold(config: dict):
    '''
    run K fold cross validation
    input: 
        - config: dict, run config of this run, use get_run_config() to get it
    '''
    set_seed(get_run_config()['random_state'])
    # prepare K fold data
    kf = KFold(n_splits=3, shuffle=True)
    
    x, y, (NUM, CAT, LABEL, cat_num) = get_data()
    
    for index, (train_index, test_index) in enumerate(kf.split(x)):
        print('=================[', index+1,'Fold ]=================')
        # split train and test data
        train_data, train_label = x[train_index], y[train_index]
        test_data, test_label = x[test_index], y[test_index]
        # shuffle and split validation data from train data
        train_ratio = 1 - config['validation_ratio']
        indices = torch.randperm(len(train_data))
        val_indeices, train_indices = indices[:int(len(train_data)*(config['validation_ratio']))], indices[int(len(train_data)*(config['validation_ratio'])):]
        validation_data, validation_label = train_data[val_indeices], train_label[val_indeices]
        train_data, train_label = train_data[train_indices], train_label[train_indices]
        # shuffle test
        test_indices = torch.randperm(len(test_data))
        test_data, test_label = test_data[test_indices], test_label[test_indices]
        
        print('train_data:', (train_data).shape)
        print('train_label:', (train_label).shape)
        print('validation_data:', (validation_data).shape)
        print('validation_label:', (validation_label).shape)
        print('test_data:', (test_data).shape)
        print('test_label:', (test_label).shape)
        print('-----------------------------------------')  
        
        data_package = (train_data, train_label, validation_data, validation_label, test_data, test_label, NUM, CAT, LABEL, cat_num)
        train_one_run(config, data_package)
    print('==================[done]==================')
    