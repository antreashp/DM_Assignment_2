import numpy as np
import pandas as pd
from random import random,randint
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import xgboost as xgb
import pickle
def data_to_str(d):
    str_data = []
    for i in range(len(d)):
        rec = d[i]
        foo = str(rec[0]) +' '
        foo += 'qid:'+str(rec[1]) +' '
        for j in range(0,len(rec)-2):
            foo += str(j)+':'+str(rec[j+2]) +' '

        str_data.append(foo)
    return str_data
def str_to_data(str_data):
    d = []
    for i in range(len(str_data)):
        v = []
        rec = str_data[i].split(' ')
        v.append(int(rec[0]))
        v.append(int(rec[1].split(':')[1]))

        for j in range(2,len(rec)-1):

            v.append(float(rec[j].split(':')[1]))
        d.append(v)
    return d
def gen_dummy_data(n_ft,max_recs,n_qs):
    d = []
    
    for q in range(n_qs):
        props_in_q = randint(10,max_recs)

        for i in range(props_in_q):
            v  = []
            random_vec = np.random.random(n_ft)
            random_y = randint(0,10)
            v.append(random_y)
            v.append(q)
            v.extend(random_vec)
            # print(np.array(v).shape)
            # exit()
        d.append(v)
    return np.array(d)
    
def train(train_data,model_type = 'LR'):
    output = {'trained_model': None,
              'score':None}
    if model_type == 'LR':
        X_train = train_data[:,2:]
        y_train = train_data[:,0]
        reg = LinearRegression().fit(X_train, y_train)
        r_score = reg.score(X_train, y_train)
        output['trained_model'] = reg
        output['score'] = r_score
    elif model_type == 'RankNet':
        pass
    elif model_type == 'LambdaRank':
        # train_data = data_to_str(train_data)
        
        X_train = train_data[:,2:]
        y_train = train_data[:,0]
        train_data = xgb.DMatrix(X_train,y_train)
        params_lm2 = [('objective','rank:ndcg'),
                      ('max_depth',2),
                      ('eta',0.1), 
                      ('num_boost_round',4), 
                      ('seed',404),
                      ('verbose',0)]
        model_lm2 = xgb.train(params_lm2, train_data)
        output['trained_model'] = model_lm2
        output['score'] = None
    elif model_type == 'LambdaMART':
        pass
    return output

def eval(test_data,trained_model=None,model_type = 'LR'):        
    output = {'inferenceDf': None,
              'Loss':None}
        

        
    if model_type == 'LR':
        
        X_test = test_data[:,2:]
        y_test = test_data[:,0]

        preds = trained_model.predict(X_test)
        mse = mean_squared_error(preds, y_test)
        plot_data = pd.concat([pd.DataFrame(y_test), pd.DataFrame(preds)], axis=1)
        plot_data.columns = plot_data.columns.astype(str)
        plot_data = pd.DataFrame(data= {"true score": y_test, "predicted score": preds})
        output['inferenceDf'] = plot_data
        output['Loss'] = mse
    elif model_type == 'RankNet':
        pass
    elif model_type == 'LambdaRank':
        # test_data = data_to_str(test_data)
                
        X_test = test_data[:,2:]
        y_test = test_data[:,0]
        test_data = xgb.DMatrix(X_test,y_test)
        preds = trained_model.predict(test_data)
        mse = mean_squared_error(preds, y_test)
        plot_data = pd.concat([pd.DataFrame(y_test), pd.DataFrame(preds)], axis=1)
        plot_data.columns = plot_data.columns.astype(str)
        plot_data = pd.DataFrame(data= {"true score": y_test, "predicted score": preds})
        output['inferenceDf'] = plot_data
        output['Loss'] = mse
    elif model_type == 'LambdaMART':
        pass

    return output


def plot(plot_data,show=False,save = False,model_type = None):
    my_plot = sns.lmplot('true score', 'predicted score', data=plot_data, fit_reg=False)
    if save:
        my_plot.savefig(os.path.join('plots',model_type+'inference.png'))
    if show:
        plt.show()
if __name__ == "__main__":

    # model_type = 'LR'
    model_type = 'LambdaRank'
    # model_type = 'LambdaMART'
    # train_data = gen_dummy_data(54,25,500)
    # test_data = gen_dummy_data(54,25,25)
    train_data = pickle.load(open('DATA/train_after_gmm.pkl','rb'))
    print(train_data)
    train_data = train_data.to_numpy()
    # def to_xgmatrix(train_data):
    #     def change_row():

    # exit()
    print('Training ',model_type,' model...')
    train_output = train(train_data,model_type = model_type)
    trained_model = train_output['trained_model']
    print('Training Results:')
    print('model: ',model_type,' score: ',train_output['score'])

    # print('Evaluating ',model_type,' model...')
    # test_output = eval(test_data,trained_model=trained_model,model_type = model_type)
    # inference_df = test_output['inferenceDf'] 
    # print('Evaluation Results:')
    # print('model: ',model_type,' Loss: ',test_output['Loss'])


    
    # print('Plotting Inference of ',model_type,' model...')
    # plot(inference_df,show=True,save = False,model_type = model_type)


