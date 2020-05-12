#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'learning-to-rank'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# In this notebook we train and compare three different learning to rank models. 
# 
# Each of the models are built using the [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html) package, which is optimised for gradient boosting. 
# 
# We models we build are: RankNet, LambdaNet and LambdaRank. 

#%%
import xgboost as xgb

#%% [markdown]
# ## Data processing
# 
# In order to implement the XGBoost algorithms the data must be preprocessed in a particular way - XGBoost requires an file that indicates which query each entry in the data frame comes from. If your training file is `train.txt`, the same folder should contain a file named `train.txt.group`, holding the group information. `train.txt.group` will contain a list of numbers indicating the number of feature vectors for each query, in the same order as the appear in `train.txt`. For example, if `train.txt.group` contained [7,8,5] we would know that there are 20 feature vectors in the training set and that the first 7 correspond to one query, the next 8 correspond to a different query and the final five are from a third query. 
# 
# There's a bash script in the repo which should be run to compute the group information file for your `train.txt` file. 
#%% [markdown]
# We load in the data as a DMatrix, an xgboost data store which is optimised for efficiency and speed when implementing xgboost models. 

#%%
training_data = xgb.DMatrix('data/train_dat.txt')
testing_data = xgb.DMatrix('data/test_dat.txt')

#%% [markdown]
# When XGBoost loads in the data, it also checks the folder containing the data for a file of the same name with file extension .txt.group. The .txt.group file contains query id information for the data in the .txt data set. We can see from the logging info printed when we load in the data that the train_dat file contains data from 6000 different queries, and the test_dat.txt contains data from 2000 queries. XGBoost uses this query information both for building models and for evaluating the goodness of the models built.
# We can also see from the logging that the data loaded in is of length 137 - that is the 136-long feature vector and the one "score" which is given by the human in the preprocessing stage. In this data set, the score is an integer from 0 to 4, with 0 denoting "this article is not relevant" and 4 denoting "this article is extrememly relevent".
#%% [markdown]
# ## LambdaRank

#%%
params_lm2 = [('objective','rank:ndcg'),('max_depth',2), ('eta',0.1), ('num_boost_round',4), ('seed',404)]

start_lm2 = time.time()           
model_lm2 = xgb.train(params_lm2, training_data)
end_lm2 = time.time()
print(end_lm2-start_lm2)


#%%
start_lm2p = time.time()
pred_lm2 = model_lm2.predict(testing_data)
end_lm2p = time.time()
print(end_lm2p-start_lm2p)


#%%
pred_lm2


#%%
params_lm6 = [('objective','rank:ndcg'),('max_depth',6), ('eta',0.1), ('num_boost_round',4), ('seed',404)]

start_lm6 = time.time()           
model_lm6 = xgb.train(params_lm6, training_data)
end_lm6 = time.time()
print(end_lm6-start_lm6)


#%%
start_lm6p = time.time()
pred_lm6 = model_lm6.predict(testing_data)
end_lm6p = time.time()
print(end_lm6p-start_lm6p)

#%% [markdown]
# We write the predictions to file. Later we will read them in and compare them across all the models. 

#%%
import pandas as pd
pd.DataFrame(pred_lm6).to_csv("data/pred_lm6.txt", header=None, sep=" ")
pd.DataFrame(pred_lm2).to_csv("data/pred_lm2.txt", header=None, sep=" ")


#%%
params = [('objective','rank:ndcg'),
          ('max_depth',2), ('eta',0.1), ('num_boost_round',4)]

model = xgb.train(params, training_data)

predictions = model.predict(testing_data)


