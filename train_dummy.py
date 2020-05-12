import numpy as np

import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import dcg_score
from utils import  *
# from sklearn.metrics import  ndcg_score
device ='cpu'# torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# df = pd.read_pickle('dummy_data.pkl')
def gen_random_data(n_srchs=50,n_ranks=10,n_props=100, n_ft_q=30, n_ft_p=10, seed=42, type = 'pointwise'):
    #generate random data in the shapes that are expected after pre-prosessing.
    props = torch.rand((n_props,n_ft_p))
    # print(props.shape)
    queries = torch.rand((n_srchs,n_ft_q))
    # ranks = np.arange(1,n_ranks+1)
    _props = np.arange(n_props)
    X = []
    y= []
    for q in queries:

        _props = np.random.permutation(_props)
        for prop in range(n_ranks):
            X.append([q, props[_props[prop]]])
            y.append(torch.from_numpy(np.array([prop])).float())

    # X = np.array(X)
    # y = np.array(y)
    
    return X,y
def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result
def eval_at_k(model,X,y):

    predictions = defaultdict(list)
    for ((q,p),rank) in zip(X,y):
        
        out = model(q,p).item()
        predictions[str(q)].append([rank,out])

    # sub_li.sort(key = lambda x: x[1]) 
    # print(predictions[str(X[0][0])])
    # predictions[str(X[0][0])].sort(key = lambda x: x[0])    
    # print(predictions[str(X[0][0])])    
    # print(np.array(predictions[str(X[0][0])])[:,1])
    scores = []
    for i in range(len(list(predictions.keys()))):

        preds = np.array(predictions[str(X[i][0])])[:,1]
        sorted_preds = [0] * len(preds)
        for j, x in enumerate(sorted(range(len(preds)), key=lambda y: preds[y])):
            sorted_preds[x] = j
    # print(sorted_preds)
        actual_ranks =  np.array(predictions[str(X[i][0])])[:,0].astype(int)
        print(actual_ranks)
        print(preds)
    # ndcg_score(np.array(predictions[str(X[0][0])])[:,0],np.array(predictions[str(X[0][0])])[:,1])
    # print(len(actual_ranks))

    # print(recall(actual_ranks,preds,7))
    # print(precision(actual_ranks,preds,7))
        scores.append( ndcg(sorted_preds,5,  alternate=False))
        # print(score)
    return scores
    # average_precision_score(actual_ranks,sorted_preds)
    # exit()
class Net_point(nn.Module):
    def __init__(self, in_sizes = [30,10],l_sizes = [100,100]  ):
        super(Net_point, self).__init__()
        self.l_sizes = l_sizes
        self.in_sizes = in_sizes
        self.input_layer_p =  nn.Linear(self.in_sizes[1], int(self.l_sizes[0]/2))
        self.input_layer_q =  nn.Linear(self.in_sizes[0], int(self.l_sizes[0]/2))
        self.fcs = []
        
        for i in range(0,len(l_sizes)-1):
 
            self.fcs.append( nn.Linear(l_sizes[i] , l_sizes[i+1]))
        self.fcs.append( nn.Linear(l_sizes[-1], 1))
        
    def forward(self, query,prop):
        q = F.relu(self.input_layer_q(query))
        p = F.relu(self.input_layer_p(prop))
        # print(q.shape)
        # print(p.shape)
        x = torch.cat((q, p), 0)
        # print(x.shape)
        for i in range( len(self.fcs)):
            # fc = getattr(self, 'fc' + str(i))
            x = F.relu(self.fcs[i](x))
        x = F.sigmoid(x)
        # fc = getattr(self, 'fc' + str(self.fc_layers))
        return x

if __name__ == "__main__":
    X,y = gen_random_data()
    # print(X.shape,y.shape)
    X_train = X[:int(X.__len__()*0.8)]
    X_test = X[int(X.__len__()*0.8):]
    y_train = y[:int(y.__len__()*0.8)]
    y_test = y[int(y.__len__()*0.8):]
    
    model = Net_point().to(device)
    print(X[0][0].shape)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for ((q,p),r) in zip (X_train,y_train):
        out = model(q.to(device),p.to(device))
        loss = loss_fn(out*10, r ) 
        print(loss)
        loss.backward()
        optimizer.step()
        # exit()


    print(model(X[0][0].to(device),X[0][1].to(device)))

    scores = eval_at_k(model,X_test,y_test)
    print(scores)




    exit()











