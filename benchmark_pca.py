import pandas as  pd
import numpy as np
from DataTables import DataTables
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import pickle
    # from sklearn.decomposition import PCA, IncrementalPCA
class PCA_Benchmark():
    def __init__(self,matrix = None,chunksize = 1000,batch_size =100):
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
    def to_numpy(self,matrix):
        return np.array(matrix.dropna(axis='columns',thresh=50).values.tolist())


    def run_props(self):
        # data = self.prop_data.to_numpy()
        data = self.to_numpy(self.prop_data)
        
        missing = ~np.isfinite(data)
        mu = np.nanmean(data, 0, keepdims=1)
        data = np.where(missing, mu, data)
        self.prop_pca = IncrementalPCA(n_components=data.shape[0]-1, batch_size=self.batch_size)
        # chunk_size = 300
        for i in range(0, data.shape[1]//self.chunksize):
            self.prop_pca.partial_fit(data[i*self.chunksize : (i+1)*self.chunksize])
        # for i in range(0, num_rows//chunk_size):
        #     data[i*self.chunksize:(i+1) * self.chunksize] = ipca.transform(features[i*self.chunksize : (i+1)*self.chunksize])
        kpca_transform = self.prop_pca.transform(data)
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        print(explained_variance_ratio)

        
        print(np.cumsum(explained_variance_ratio))
        # self.prop_pca = self.prop_pca.fit_transform(data)
        # self.prop_pca = PCA()
        # self.prop_pca.fit(data)
        plt.figure(1)
        plt.plot(np.cumsum(explained_variance_ratio))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def run_srchs(self):
        data = self.to_numpy(self.srch_data)
        
        missing = ~np.isfinite(data)
        mu = np.nanmean(data, 0, keepdims=1)
        data = np.where(missing, mu, data)
        self.srch_pca = IncrementalPCA(n_components=data.shape[0]-1, batch_size=self.batch_size)
        # chunk_size = 300
        for i in range(0, data.shape[1]//self.chunksize):
            self.srch_pca.partial_fit(data[i*self.chunksize : (i+1)*self.chunksize])
        kpca_transform = self.srch_pca.transform(data)
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        print(explained_variance_ratio)

        
        print(np.cumsum(explained_variance_ratio))
        # self.prop_pca = self.prop_pca.fit_transform(data)
        # self.prop_pca = PCA()
        # self.prop_pca.fit(data)
        plt.figure(1)
        plt.plot(np.cumsum(explained_variance_ratio))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
if __name__ == "__main__":
    # data = DataTables()

    # pickle.dump( data, open( 'datatables_dummy.pkl', "wb" ) )


    data = pickle.load( open(  'datatables_dummy.pkl', "rb" ) )

    print(__doc__)

    pcab = PCA_Benchmark(data)

    pcab.run_props()
    # pcab.run_srchs()