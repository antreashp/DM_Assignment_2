# from sklearn.decomposition import PCA, IncrementalPCA,  KernelPCA
import pandas as  pd
import numpy as np
# from DataTables import DataTables


import matplotlib.pyplot as plt
import pickle
import os

pca_type = 'IPCA'
data_type = 'prop'
whiten = True
for pca_type in ['IPCA']:
    for data_type in ['prop','srch']:
        for whiten in [True,False]:
            for cluster_type in ['gmm','kmodes','kproto']:
                debug = False 
                # foo = '_dummy' if debug else ''
                foo = '_dummy' if debug else ''
                # cluster_type = 'gmm'
                data_path = 'DATA/after_'+cluster_type+'_'+data_type+foo+'.pkl'

                # foo = '_whitened_' if whiten else '_' 
                # model_path = 'models/' + pca_type +'_' +data_type+foo+ 'model.pkl'
                if whiten:
                    model_path = 'models/'+pca_type+'_whitened_after_'+cluster_type+'_'+data_type+foo
                    dest_path = 'DATA/'+pca_type+'_whitened_after_'+cluster_type+'_'+data_type+foo+'.pkl'
                else:
                    model_path = 'models/'+pca_type+'_after_'+cluster_type+'_'+data_type+foo
                    dest_path = 'DATA/'+pca_type+'_after_'+cluster_type+'_'+data_type+foo+'.pkl'
                model = pickle.load(open(model_path,'rb'))
                print(model)



                data = pickle.load(open(data_path,'rb'))
                print(data.shape)
                data = model.transform(data)
                print(data.shape)

                pickle.dump( data , open('DATA/after_'+pca_type+'_after_'+cluster_type+'_'+data_type+foo+ '.pkl', "wb" ) )

