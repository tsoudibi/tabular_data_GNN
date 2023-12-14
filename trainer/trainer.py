from utils.utils import *
from utils.loggers import *
from model.model import *
from data.preprocess import *
from tqdm import trange
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import BinaryAUROC

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
        

def train_one_epoch(model, optimizer, datas, batch_size, epoch):
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
                    output = model(validation_data[i], epoch=200)
                    # loss = torch.nn.functional.cross_entropy(output, validation_label[i])
                    preds = output.softmax(dim=1)
                    true = torch.nn.functional.one_hot(validation_label[i], num_classes=2).to(DEVICE)
                    valid_acc.update(torch.argmax(preds,1),true.T[1])
                    valid_auc.update(preds.T[0],true.T[0])
                stepper.set_postfix({'loss': round(batch_loss/(i+1), 3), 'acc': round(train_acc.item(), 3), 'AUC': round(train_auc.item(), 3), 'val_acc': round(valid_acc.compute().item(), 3), 'val_AUC': round(valid_auc.compute().item(), 3)})

def train_one_run(configs):
    # load configs hyperparameter
    max_epoch = configs['max_epoch']
    learning_rate = configs['learning_rate']
    batch_size = configs['batch_size']
    train_ratio = configs['train_ratio']
    validation_ratio = configs['validation_ratio']
    
    x, y, (NUM, CAT, LABEL, cat_num) = get_data()
    
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

    # build model and optimizer
    the_model = K_graph(NUM, CAT, [LABEL], cat_num).to(DEVICE)
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
            output = the_model(test_data[i], epoch=200)
            preds = output.softmax(dim=1)
            true = torch.nn.functional.one_hot(test_label[i], num_classes=2).to(DEVICE)
            test_acc = MulticlassAccuracy(num_classes=2).to(DEVICE)
            test_auc = BinaryAUROC().to(DEVICE)
            test_acc.update(torch.argmax(preds,1),true.T[1])
            test_auc.update(preds.T[0],true.T[0])

        print('test_acc:', test_acc.compute().item())
        print('test_auc:', test_auc.compute().item())
        print('-----------------------------------------')