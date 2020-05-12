
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os,sys
from shutil import copyfile
import shutil
import time
import numpy as np
from sklearn.decomposition import PCA
from preprosses import preprocess
from mood_dataset import MOOD
from models import MLP
from transform import Transform
import xgboost as xgb
def accat(out,trg,thresh=0.5,preprocess_instance=None, transform_targets=False ):
        out = out.detach().cpu().numpy().squeeze() *9
        trg = trg.detach().cpu().numpy().squeeze() *9
        if transform_targets:
            out = preprocess_instance.decode(out)
            trg = preprocess_instance.decode(trg)
        
        diff = np.abs(out - trg)
        # print(diff)
        diff[diff > thresh] = 1
        diff[diff <= thresh] = 0
        diff = (diff * -1) + 1
        correct = np.sum(diff)
        # print(correct)
        return correct
def train_mlp(options, X_train, X_test, y_train, y_test):

    exp_name = options['exp_name']
    batch_size = options['batch_size']
    use_pca = options['use_pca']
    model_type = options['model_type']
    loss_fn = options['loss_fn']
    optim = options['optim']
    use_scheduler = options['use_scheduler']
    lr = options['lr']
    epochs = options['epochs']
    pca_var_hold = options['pca_var_hold']
    debug_mode = options['debug_mode']
    win_size = options['win_size']
    if exp_name is None:

        exp_name = 'runs/Raw_' +str(model_type)+'_pca_'+str(use_pca)+str(round(pca_var_hold))+'_'+str(batch_size)+'_'+str(round(lr,2))+'_win'+str(win_size)+'_transf'+str(options['transform_targets'])
    if os.path.exists(exp_name):
        shutil.rmtree(exp_name)

    # time.sleep(1)
    writer = SummaryWriter(exp_name,flush_secs=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRANSF = Transform(y_train)

    if options['transform_targets']:
        y_train = TRANSF.fit(y_train)
        y_test = TRANSF.fit(y_test)
    if use_pca and 'Raw' in exp_name:
        scaler = PCA(pca_var_hold)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    needed_dim = X_train.shape[1]

    dataset_train = MOOD(X_train, y_train, model_type=model_type,data_type='train',debug_mode=debug_mode)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_val = MOOD(X_test, y_test, model_type=model_type,data_type='val')
    valid_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    
    model = MLP(needed_dim=needed_dim,model_type=model_type,n_classes=None)
    model.to(device)
    if optim == None:
        print('you need to specify an optimizer')
        exit()
    elif optim == 'adam':
        optimizer = torch.optim.Adam(   model.parameters(), lr=lr)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(   model.parameters(), lr=lr,momentum=0.9)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,threshold=0.0001,patience = 10)
    if loss_fn == None:
        print('you need to specify an optimizer')
        exit()
    else:

        if loss_fn == 'mse':

            loss_fn = torch.nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            loss_fn = torch.nn.CrossEntropyLoss()
    
    
    
    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    best = 0  #small number for acc big number for loss to save a model
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # print(images.shape)
            outputs = model(images)

            loss =loss_fn(outputs,labels)
            # print('loss: ',loss.item())
            writer.add_scalar('Loss/train', loss.item(), len(train_loader)*epoch+i)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            del outputs
            # if (i * batch_size) % (batch_size * 100) == 0:
            #     print(f'{i * batch_size} / 50000')
                
        model.eval()
        correct_5_2 = 0
        correct_5_1 = 0
        
        total_loss = 0
        total = 0
        accsat =[1,0.5,0.05]
        accs = np.zeros(len(accsat))
        # corrs = np.zeros(len(accsat))
        correct_array = np.zeros(len(accsat))
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss =  loss_fn(outputs, labels)

                """
                Correct if:
                preprocess.decode_target(output) == proprocess.decode(target)
                
                """
                for i in range(len(accsat)):

                    correct_array[i] += accat(outputs,labels,thresh=accsat[i],preprocess_instance=TRANSF, transform_targets=options['transform_targets'])

                # total_loss += loss.item()
                total += labels.size(0)
                
                
                valid_losses.append(loss.item())


                
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
        # scheduler.step(np.mean(valid_losses))
        for i in range(len(accsat)):
            accs[i] = 100*correct_array[i]/total
            writer.add_scalar('Acc/val_@'+str(accsat[i]), accs[i], epoch)
        
        if float(accs[1] /100) > best:
            best = float(accs[1] /100)
            torch.save(model.state_dict(),os.path.join(os.getcwd(),'models','meh.pth'))
        
        writer.add_scalar('Loss/val', np.mean(valid_losses), epoch)
        # valid_acc_list.append(accuracy)
    return best, np.mean(valid_losses)



def train_xgb(model_options, X_train, X_test, y_train, y_test):
    if model_options['use_pca'] :
        scaler = PCA(model_options['pca_var_hold'])
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if model_options['transform_targets']:
        TRANSF = Transform(y_train)
        y_train = TRANSF.fit(y_train)
        y_test = TRANSF.fit(y_test)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =model_options['colsample_bytree'], learning_rate = model_options['lr'],
                    max_depth =model_options['max_depth'], alpha = model_options['aplha'], n_estimators = model_options['n_estimators'],verbosity=1,gamma = model_options['gamma'],max_delta_step =model_options['max_delta_step'])
        
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_test)
    
    # print(np.round(preds[:5],2),y_test[:5])
    if model_options['transform_targets'] :
        y_test = TRANSF.decode(y_test)
        preds = TRANSF.decode(preds)
    # print(np.round(preds[:5],2),y_test[:5])
    diff = abs(preds- y_test )
    # print(diff[:5])
    accuracy = (len(diff[diff<0.5]) )/preds.shape[0]
    xgb.plot_tree(xg_reg, num_trees=2)
    plt.figure(1)
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    fig.savefig('results/best_xgb_wtransf_tree'+str(round(accuracy,2))+'.png')
    plt.figure(2)
    xgb.plot_importance(xg_reg,show_values=True)
    
    plt.savefig('results/best_xgb_wtransf_impotrance'+str(round(accuracy,2))+'.png')
    return accuracy    , diff.mean()
    # print(' accuracy, ',accuracy)