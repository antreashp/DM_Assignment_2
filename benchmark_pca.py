import pandas as  pd
import numpy as np
from DataTables import DataTables
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import pickle
    # from sklearn.decomposition import PCA, IncrementalPCA
class PCA_Benchmark():
    def __init__(self,matrix = None,chunksize = 200,batch_size =50):
        if matrix is None:
            print('Please enter data, Exiting...')
            exit()
        else:
            self.matrix = matrix
        self.chunksize = chunksize
        self.batch_size = batch_size
        # self.input_variables =  matrix.property[data.property_attributes].columns.values
        self.prop_data = matrix.property[ matrix.property_attributes]#.drop(columns=[])
        self.srch_data = matrix.search[ matrix.search_attributes].drop(columns=['date_time'])
        print(self.srch_data)
    def to_numpy(self,matrix):
        return np.array(matrix.dropna(axis='columns',thresh=30).values.tolist())


    def run_props(self):
        # data = self.prop_data.to_numpy()
        data = self.to_numpy(self.prop_data)
        
        missing = ~np.isfinite(data)
        mu = np.nanmean(data, 0, keepdims=1)
        data = np.where(missing, mu, data)
        self.prop_pca = IncrementalPCA(n_components=data.shape[1]-1, batch_size=self.batch_size)
        # chunk_size = 300
        # for i in range(0, data.shape[1]//self.chunksize):
        #     self.prop_pca.partial_fit(data[i*self.chunksize : (i+1)*self.chunksize])
        # for i in range(0, num_rows//chunk_size):
        #     data[i*self.chunksize:(i+1) * self.chunksize] = ipca.transform(features[i*self.chunksize : (i+1)*self.chunksize])
        kpca_transform = self.prop_pca.fit_transform(data)
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        return explained_variance_ratio

    def run_srchs(self):
        data = self.to_numpy(self.srch_data)
        
        missing = ~np.isfinite(data)
        mu = np.nanmean(data, 0, keepdims=1)
        data = np.where(missing, mu, data)
        self.srch_pca = IncrementalPCA(n_components=data.shape[1]-1, batch_size=self.batch_size)
        # chunk_size = 300
        # for i in range(0, data.shape[1]//self.chunksize):
        #     self.srch_pca.partial_fit(data[i*self.chunksize : (i+1)*self.chunksize])
        kpca_transform = self.srch_pca.fit_transform(data)
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        return explained_variance_ratio

    def run_all(self,show=False,save=False):
        exp_var_ratio_props = self.run_props()
        exp_var_ratio_srchs = self.run_srchs()
        
        plt.figure(1)
        plt.plot(np.cumsum(exp_var_ratio_props))
        plt.title('IPCA Principal Components Cumulative Explained Variance For Properties')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        
        plt.figure(2)
        plt.plot(np.cumsum(exp_var_ratio_srchs))
        plt.title('IPCA Principal Components Cumulative Explained Variance For Queries')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        
        if show:
            plt.show()
        if save:
            plt.savefig('plots/IPCA_cum_explained_var_props.png')
            plt.savefig('plots/IPCA_cum_explained_var_srchs.png')
        return exp_var_ratio_props, exp_var_ratio_srchs
if __name__ == "__main__":
    # data = DataTables()

    # pickle.dump( data, open( 'datatables_dummy.pkl', "wb" ) )


    data = pickle.load( open(  'datatables_dummy.pkl', "rb" ) )

    print(__doc__)

    pcab = PCA_Benchmark(data)

    # pcab.run_props()
    # pcab.run_srchs()
    exp_var_ratio_props, exp_var_ratio_srchs = pcab.run_all(show=True,save=True)