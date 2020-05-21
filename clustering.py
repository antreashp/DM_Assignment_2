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
from tqdm import tqdm
import pickle
class Clustering():
    def __init__(self,matrix = None,cl_type = 'prop'):
        if matrix is None:
            print('Please enter data, Exiting...')
            exit()
        self.input_variables =  matrix.columns.values
        self.matrix = matrix
        self.cl_type = cl_type
    def find_kmeans_silhouette(self,kmin=2,kmax=15,save = True):
        df_id_vs_variable = self.matrix.fillna(0)
        sil_scores =[]
        for k in tqdm(range(kmin,kmax)):
            km = KMeans(n_clusters=k, n_init=20).fit(df_id_vs_variable)
            sil_scores.append(silhouette_score(df_id_vs_variable, km.labels_))

        #Plot
        plt.plot(range(kmin,kmax), sil_scores)
        plt.title('KMeans Results')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        
    def to_numpy(self):
        return np.array(self.matrix.fillna(np.nan).values.tolist())
        
    def kmodes(self,K=20,N=int(1e5),T=50,type='huang',save = True):
        # data
        
        data = self.to_numpy()
        
        # data.fillna(0)
        # missing = ~np.isfinite(data)
        # mu = np.nanmean(data, 0, keepdims=1)
        # data = np.where(missing, mu, data)
        if type == 'huang':
            model = KModes(n_clusters=K,init='Huang',n_init=1,verbose=2)
        elif type == 'huang_ng':
            model = KModes(n_clusters=K,init='Huang',cat_dissim=ng_dissim,n_init=1,verbose=1)
        if type == 'cao':
            model = KModes(n_clusters=K,init='Cao',verbose=2)
        preds = model.fit_predict(data)
        centroids = model.cluster_centroids_
        labels = model.labels_
        if save:
            self.save(model,'Clustering_kmodes_model')
        return centroids, labels

    def kproto(self,K=20,N=int(1e5),MN=4,T=10,type='cao',save = True):
        data = self.to_numpy()

        M = data.shape[1]
        # MN = 22
        if type == 'huang':
            model = KPrototypes(n_clusters=K, init='Huang', n_init=1, verbose=1)
        if type == 'cao':
            model = KPrototypes(n_clusters=K, init='Cao', verbose=2,max_iter=10000)
        
        clusters = model.fit_predict(data, categorical=[0,3,6,8]if self.cl_type == 'prop' else [0,2,3,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
        if save:
            self.save(model,'Clustering_kproto_model')
        return np.array(model.cluster_centroids_[0]),np.array(model.cluster_centroids_[1]),np.array(clusters)

    def GMM(self,k = 10,covariance_type='diag', init_params='kmeans', min_covar=0.001, n_init=1, n_iter=10, params='wmc', random_state=None,tol=0.0001, verbose=1,save = True):
        data = self.to_numpy()
        
        # missing = ~np.isfinite(data)
        # mu = np.nanmean(data, 0, keepdims=1)
        # data = np.where(missing, mu, data)

        gmm = GMM(n_components=k,  covariance_type=covariance_type, tol=tol, reg_covar=1e-06, max_iter=100000, n_init=n_init, init_params=init_params, warm_start=True, verbose=1, verbose_interval=10)
        # gmm = GMM(n_components=k,covariance_type=covariance_type, init_params=init_params, min_covar=min_covar, n_init=n_init, n_iter=n_iter, params=params, random_state=random_state,tol=tol, verbose=verbose)
        gmm.fit(data)
        labels = gmm.predict(data)
        probs = gmm.predict_proba(data)
        if save:
            self.save(gmm,'Clustering_gmm_model')


        return labels, probs
    def save(self,model,name = 'cluster_model'):
        pickle.dump( model, open( 'models/'+name+'_'+self.cl_type+'.pkl', "wb" ) )
if __name__ == "__main__":
    debug = True
    cl_type = 'srch' 
    all_data_prop_path = 'all_property.pkl'
    all_data_srch_path = 'all_search.pkl'
    dummy_data_prop_path = 'dummy_property.pkl'
    dummy_data_srch_path = 'dummy_search.pkl'
    prop_path = all_data_prop_path if not debug else dummy_data_prop_path
    srch_path = all_data_srch_path if not debug else dummy_data_srch_path

    data_path =          prop_path if 'prop' in cl_type else srch_path 

    
    data = pickle.load(open(data_path,'rb'))
    print (data.columns)
    print (len(list(data.columns)))
    data = data.head(7500)
    c = Clustering(data,cl_type=cl_type)
    
    # c.find_kmeans_silhouette(kmin=3,kmax=20)

    k = 7

    # clusters, probs = c.GMM(k = k)
    # centroids, labels = c.kmodes(K=k,N=int(30000),T=5,type='huang')
    # data = data.head(15000)
    num, cat,labels = c.kproto(K=k,N=int(30000),MN=29,T=2,type='cao')
    
    # centroids, labels = c.kmodes(K=8,N=int(1e5),T=1,type='huang')
    # num, cat,labels = c.kproto(K=8,N=int(100000),MN=10,T=1,type='huang')
    
    # print(clusters)
    # print(clusters, probs)
    # print(centroids,labels)
    # print(num.shape, cat.shape, labels.shape)