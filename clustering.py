from DataTables import DataTables
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns
import timeit
from kmodes.kmodes import KModes
import math
from sklearn.mixture import GMM
from kmodes.kprototypes import KPrototypes
from kmodes.util.dissim import ng_dissim


class Clustering():
    def __init__(self,matrix = None):
        if matrix is None:
            print('Please enter data, Exiting...')
            exit()
        self.input_variables =  matrix.property[data.property_attributes].columns.values
        self.matrix = matrix.property[ ['prop_id']+matrix.property_attributes]
    
    def vanilla(self):
        df_id_vs_variable = self.matrix.fillna(0) # fill nan with 0
        # input_variables = ['prop_id']+self.matrix.property_attributes
        def makeBinary(x):
            if abs(x) > 0.00000:
                return 1
            else:
                return 0
        df_id_vs_variable = df_id_vs_variable.groupby('prop_id').agg('sum').applymap(makeBinary)
        print(df_id_vs_variable.head())
        df_unique_set_variables = df_id_vs_variable.drop_duplicates(keep="first")
        print("Number of securities: {0}".format(df_id_vs_variable.shape[0]))
        print("Number of unique lines: {0}".format(df_unique_set_variables.shape[0]))
        df_no_cluster = df_id_vs_variable.loc[~df_id_vs_variable.duplicated(self.input_variables,keep=False)]

        #These lines are duplicated so they can be "clustered"
        df_cluster = df_id_vs_variable.loc[df_id_vs_variable.duplicated(self.input_variables,keep=False)]
        df_cluster = df_cluster.groupby(df_cluster.index).size()

        array_cluster = df_cluster.values

        print("Number of securities that do not belong to a cluster:{0}".format(df_no_cluster.shape[0]))
        print("Number of clusters: {0}".format(len(array_cluster)))
        print("Number of securities that belong to a cluster: {0}".format(sum(array_cluster)))
        print("##########################")
        print("   Clusters Statistics")
        print("##########################")
        print(df_cluster.describe())
        n, bins, patches = plt.hist(array_cluster, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('Cluster size')
        plt.ylabel('Distribution')
        plt.title(r'Distribution of Cluster Size')
        sns.clustermap(df_id_vs_variable.transpose())
        plt.show()
        return array_cluster
    
    def find_kmeans_silhouette(self,kmin=2,kmax=15,save = True):
        df_id_vs_variable = self.matrix.fillna(0)
        sil_scores =[]
        for k in range(kmin,kmax):
            km = KMeans(n_clusters=k, n_init=20).fit(df_id_vs_variable)
            sil_scores.append(silhouette_score(df_id_vs_variable, km.labels_))

        #Plot
        plt.plot(range(kmin,kmax), sil_scores)
        plt.title('KMeans Results')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        
    def to_numpy(self):
        return np.array(self.matrix.drop(columns=['prop_id']).dropna(axis='columns',thresh=50).values.tolist())
        
    def kmodes(self,K=20,N=int(1e5),M=10,T=3,type='huang',save = True):
        data = self.to_numpy()
        
        missing = ~np.isfinite(data)
        mu = np.nanmean(data, 0, keepdims=1)
        data = np.where(missing, mu, data)
        if type == 'huang':
            model = KModes(n_clusters=K,init='Huang',n_init=1,verbose=2)
        elif type == 'huang_ng':
            model = KModes(n_clusters=K,init='Huang',cat_dissim=ng_dissim,n_init=1,verbose=1)
        if type == 'cao':
            model = KModes(n_clusters=K,init='Cao',verbose=2)
        preds = model.fit_predict(data)
        centroids = model.cluster_centroids_
        labels = model.labels_
        return centroids, labels

    def kproto(self,K=20,N=int(1e5),MN=22,T=3,type='cao',save = True):
        data = np.array(self.matrix.drop(columns=['prop_id']).values.tolist())
        meh = []
        for i in range(len(data)):
            meh.append([])
            for j in range(len(data[i])):
                meh[i].append(1 if not math.isnan(data[i][j]) else 0 )
            meh[i] = np.array(meh[i])
        data = np.array(meh)
        M = data.shape[1]
        # MN = 22
        if type == 'huang':
            model = KPrototypes(n_clusters=K, init='Huang', n_init=2, verbose=1)
        if type == 'cao':
            model = KPrototypes(n_clusters=K, init='Cao', verbose=2)
        clusters = model.fit_predict(data, categorical=list(range(M - MN, M)))
        return np.array(model.cluster_centroids_[0]),np.array(model.cluster_centroids_[1]),np.array(clusters)

    def GMM(self,k = 10,covariance_type='diag', init_params='wmc', min_covar=0.001, n_init=1, n_iter=100, params='wmc', random_state=None,tol=0.001, verbose=1,save = True):
        data = self.to_numpy()
        
        missing = ~np.isfinite(data)
        mu = np.nanmean(data, 0, keepdims=1)
        data = np.where(missing, mu, data)


        gmm = GMM(n_components=k,covariance_type=covariance_type, init_params=init_params, min_covar=min_covar, n_init=n_init, n_iter=n_iter, params=params, random_state=random_state,tol=tol, verbose=verbose)
        gmm.fit(data)
        labels = gmm.predict(data)
        probs = gmm.predict_proba(data)
        if save:
            self.save(gmm,'Clustering_gmm_model.pkl')


        return labels, probs
    def save(self,model,name = 'cluster_model'):
        pickle.dump( data, open( 'models/'+name+'.pkl', "wb" ) )
if __name__ == "__main__":
    # data = DataTables()
    # data = data.property[ [data.property_pk]+data.property_attributes]
    # pickle.dump( data, open( 'datatables.pkl', "wb" ) )

    # exit()
    data = pickle.load( open(  'datatables.pkl', "rb" ) )
    c = Clustering(data)
    
    # c.find_kmeans_silhouette(kmin=2,kmax=15)
    # clusters = c.vanilla()
    clusters, probs = c.GMM(k = 10,covariance_type='diag', init_params='wmc', min_covar=0.001, n_init=1, n_iter=100, params='wmc', verbose=1)
    # centroids, labels = c.kmodes(K=10,N=int(1e5),M=10,T=3,type='huang')
    # num, cat,labels = c.kproto(K=10,N=int(700),MN=22,T=3,type='cao')
    
    
    # print(clusters)
    # print(clusters, probs)
    # print(centroids,labels)
    print(num.shape, cat.shape, labels.shape)