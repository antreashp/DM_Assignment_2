from DataTables import DataTables
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns
import timeit
from kmodes.kmodes import KModes
import math
from sklearn.mixture import     GaussianMixture as GMM
from kmodes.kprototypes import KPrototypes
from kmodes.util.dissim import ng_dissim

import pickle


cluster_type = 'gmm'
data_type = 'prop'
debug = False 
all_data_prop_path = 'all_property.pkl'
all_data_srch_path = 'all_search.pkl'
dummy_data_prop_path = 'dummy_property.pkl'
dummy_data_srch_path = 'dummy_search.pkl'
prop_path = all_data_prop_path if not debug else dummy_data_prop_path
srch_path = all_data_srch_path if not debug else dummy_data_srch_path
data_path =          prop_path if 'prop' in data_type else srch_path 
categorical=[0,3,6,8] if data_type == 'prop' else [0,2,3,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

model_path = 'models/'+ 'Clustering_'+cluster_type+'_model_'+data_type+'.pkl'


model = pickle.load(open(model_path,'rb'))
print(model)



data = pickle.load(open(data_path,'rb'))
print(data.shape)
if cluster_type == 'gmm':

    labels = model.predict(data.to_numpy())
elif cluster_type == 'kmodes':
    
    labels = model.predict(data.to_numpy())
elif cluster_type == 'kproto' :
    
    labels = model.predict(data.to_numpy(),categorical=categorical)
else:
    print('meh')
    exit()
# if data_type == 'prop':
data = np.hstack((data,np.expand_dims(np.array(labels), axis=1)))
print(data.shape)
# print(meh.shape)
if debug:
    foo = '_dummy'
else:
    foo = ''
pickle.dump( data , open('DATA/after_'+cluster_type +'_' +data_type+ foo+'.pkl', "wb" ) )


