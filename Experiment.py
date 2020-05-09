import numpy as np
import pandas as pd
from random import random,randint
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import xgboost as xgb
import os
class Experiment():
    def __init__(self,train_data,test_data,model_type='LR',exp_opts=None,verbose=True):
        self.train_data = train_data
        self.test_data = test_data
        self.model_type = model_type
        if exp_opts is not None:
            self.opts = exp_opts
        else:
            self.opts = {'max_depth':2,
                      'eta':0.1,
                      'num_boost_round':4,
                      'seed':42}
            
        self.verbose = verbose
        self.train_results ={'trained_model': None, 'score':None}
        self.trained_model = None
        self.eval_results = {'inferenceDf': None, 'Loss':None}

    def train(self):
        if self.verbose:
            print('Training ',self.model_type,' model...')
        if self.model_type == 'LR':
            X_train = self.train_data[:,2:]
            y_train = self.train_data[:,0]
            reg = LinearRegression().fit(X_train, y_train)
            r_score = reg.score(X_train, y_train)
            self.train_results['trained_model'] = reg
            self.trained_model = reg
            
            self.train_results['score'] = r_score
        elif self.model_type == 'RankNet':
            pass
        elif self.model_type == 'LambdaRank':
            
            X_train = self.train_data[:,2:]
            y_train = self.train_data[:,0]
            training_data = xgb.DMatrix(X_train,y_train)
            params_lm2 = [('objective','rank:ndcg'),
                        ('max_depth',self.opts['max_depth'] ),
                        ('eta',self.opts['eta']), 
                        ('num_boost_round',self.opts['num_boost_round'] ), 
                        ('seed',self.opts['seed'] ),
                        ('verbose',0)]
            model_lm2 = xgb.train(params_lm2, training_data)
            self.train_results['trained_model'] = model_lm2
            self.trained_model = model_lm2
            self.train_results['score'] = None
        elif model_type == 'LambdaMART':
            
            X_train = self.train_data[:,2:]
            y_train = self.train_data[:,0]
            training_data = xgb.DMatrix(X_train,y_train)
            params_lm2 = [('objective','rank:pairwise'),
                        ('max_depth',self.opts['max_depth'] ),
                        ('eta',self.opts['eta']), 
                        ('num_boost_round',self.opts['num_boost_round'] ), 
                        ('seed',self.opts['seed'] ),
                        ('verbose',0)]
            model_lm2 = xgb.train(params_lm2, training_data)
            self.train_results['trained_model'] = model_lm2
            self.trained_model = model_lm2
            self.train_results['score'] = None
        if self.verbose:
            print('Training Results: model: ',self.model_type,' score: ',self.train_results['score'])
        return self.train_results
        
    
    def eval(self):        
        if self.verbose:
            print('Evaluating ',self.model_type,' model...')
        if model_type == 'LR':
            
            X_test = self.test_data[:,2:]
            y_test = self.test_data[:,0]

            self.preds = self.trained_model.predict(X_test)
            mse = mean_squared_error(self.preds, y_test)
            plot_data = pd.concat([pd.DataFrame(y_test), pd.DataFrame(self.preds)], axis=1)
            plot_data.columns = plot_data.columns.astype(str)
            plot_data = pd.DataFrame(data= {"true score": y_test, "predicted score": self.preds})
            self.eval_results['inferenceDf'] = plot_data
            self.eval_results['Loss'] = mse
        elif model_type == 'RankNet':
            pass
        elif model_type == 'LambdaRank':
            # test_data = data_to_str(test_data)
                    
            X_test = self.test_data[:,2:]
            y_test = self.test_data[:,0]
            testing_data = xgb.DMatrix(X_test,y_test)
            self.preds = self.trained_model.predict(testing_data)
            mse = mean_squared_error(self.preds, y_test)
            plot_data = pd.concat([pd.DataFrame(y_test), pd.DataFrame(self.preds)], axis=1)
            plot_data.columns = plot_data.columns.astype(str)
            plot_data = pd.DataFrame(data= {"true score": y_test, "predicted score": self.preds})
            self.eval_results['inferenceDf'] = plot_data
            self.eval_results['Loss'] = mse
        elif model_type == 'LambdaMART':
                    
            X_test = self.test_data[:,2:]
            y_test = self.test_data[:,0]
            testing_data = xgb.DMatrix(X_test,y_test)
            self.preds = self.trained_model.predict(testing_data)
            mse = mean_squared_error(self.preds, y_test)
            plot_data = pd.concat([pd.DataFrame(y_test), pd.DataFrame(self.preds)], axis=1)
            plot_data.columns = plot_data.columns.astype(str)
            plot_data = pd.DataFrame(data= {"true score": y_test, "predicted score": self.preds})
            self.eval_results['inferenceDf'] = plot_data
            self.eval_results['Loss'] = mse
        if self.verbose:
            print('Evaluation Results: model: ',self.model_type,' Loss: ',self.eval_results['Loss'])
        
        return self.eval_results
    def plot(self,show=True,save = True):
        if self.verbose:
            
            print('Plotting Inference of ',self.model_type,' model...')
        my_plot = sns.lmplot('true score', 'predicted score', data=self.eval_results['inferenceDf'], fit_reg=False)
        if save:
            my_plot.savefig(os.path.join('plots',self.model_type+'inference.png'))
        if show:
            plt.show()

if __name__ == "__main__":

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

    train_data = gen_dummy_data(54,25,500)
    test_data = gen_dummy_data(54,25,25)
    model_type = 'LambdaMART'
    model_type = 'LambdaRank'
    # model_type = 'LR'
    
    
    exp = Experiment(train_data,test_data,model_type=model_type)
    outs = exp.train()
    outs = exp.eval()
    exp.plot()



































