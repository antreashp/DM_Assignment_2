import pandas as  pd
import numpy as np
from DataTables import DataTables
from sklearn.decomposition import PCA, IncrementalPCA,  KernelPCA
import matplotlib.pyplot as plt
import pickle
    # from sklearn.decomposition import PCA, IncrementalPCA
class PCA_Benchmark():
    def __init__(self,prop_path=None,srch_path=None,chunksize = 1000,batch_size =300,only=None,pca_type='IPCA',kernel_type='linear',whiten=False,model_name=''):
        self.model_name = model_name
        self.kernel_type = kernel_type
        self.type = pca_type
        self.chunksize = chunksize
        self.batch_size = batch_size
        self.only = only
        self.whiten = whiten
        kernel_limit = 3000
        # self.input_variables =  matrix.property[data.property_attributes].columns.values
        if self.only == 'prop':

            self.prop_data = pd.DataFrame(pickle.load( open(  prop_path, "rb" )) )
            self.srch_data = None
            # self.prop_data = self.prop_data[self.prop_data.columns.drop(list(self.prop_data.filter(regex='visitor_hist_adr_usd_')))]
            # self.prop_data = self.prop_data.fillna(0).head(50000)#.drop(columns=[])
            if pca_type == 'Kernel':
                self.prop_data = self.prop_data.head(kernel_limit)
        elif self.only == 'srch':

            self.srch_data = pd.DataFrame(pickle.load( open(  srch_path, "rb" ) ))
            self.prop_data = None
            if pca_type == 'Kernel':
                self.srch_data = self.srch_data.head(kernel_limit)
        elif self.only == None:
            self.prop_data = pickle.load( open(  prop_path, "rb" ) )
            self.srch_data = pickle.load( open(  srch_path, "rb" ) )
            if pca_type == 'Kernel':
                self.prop_data = self.prop_data.head(kernel_limit)
                self.srch_data = self.srch_data.head(kernel_limit)
                 
            # self.prop_data = self.prop_data[self.prop_data.columns.drop(list(self.prop_data.filte))]
            # self.prop_data = self.prop_data#.drop(columns=[])
        
        # print('props')
        # print(len(list(self.prop_data.columns)))
        # print(list(self.prop_data.columns))
        # print('srchs')
        
        # print(len(list(self.srch_data.columns)))
        # print(list(self.srch_data.columns))
        if only == 'prop' or only is None:
            self.prop_data = self.to_numpy(self.prop_data) if  self.only == 'prop' or only == None else self.prop_data
        

        if only == 'srch' or only is None:
            # self.srch_data = self.to_numpy(self.srch_data) if  self.only == 'srch' or only == None else self.srch_data
        
            self.srch_data = self.to_numpy(self.srch_data)  if  self.only == 'srch' or only == None else self.srch_data

    def to_numpy(self,matrix):
        return matrix.to_numpy()


    def run_props(self):
        # data = self.prop_data.to_numpy()
        
        if self.type == 'IPCA':
            self.prop_pca = IncrementalPCA(n_components=self.prop_data.shape[1]-1, batch_size=self.batch_size,whiten=self.whiten)
        elif self.type == 'Kernel':
            self.prop_pca = KernelPCA(n_components=self.prop_data.shape[1]-1,kernel=self.kernel_type)
            
        # chunk_size = 300

        # for i in range(0, data.shape[1]//self.chunksize):
        #     self.prop_pca.partial_fit(data[i*self.chunksize : (i+1)*self.chunksize])
        # for i in range(0, num_rows//chunk_size):
        #     data[i*self.chunksize:(i+1) * self.chunksize] = ipca.transform(features[i*self.chunksize : (i+1)*self.chunksize])
        print(self.prop_data.shape)
        self.prop_pca = self.prop_pca.fit(self.prop_data)
        kpca_transform = self.prop_pca.transform(self.prop_data) 
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        # foo = '_whitened_' if self.whiten else '_'
        # model_name = self.type + '_prop' + foo +'model.pkl'
        pickle.dump( self.prop_pca , open(self.model_name , "wb" ) )
        return explained_variance_ratio

    def run_srchs(self):
        
        if self.type == 'IPCA':
            self.srch_pca = IncrementalPCA(n_components=self.srch_data.shape[1]-1, batch_size=self.batch_size,whiten=self.whiten)
        elif self.type == 'Kernel':
            self.srch_pca = KernelPCA(n_components=self.srch_data.shape[1]-1,kernel=self.kernel_type)
        # chunk_size = 300
        # for i in range(0, data.shape[1]//self.chunksize):
        #     self.srch_pca.partial_fit(data[i*self.chunksize : (i+1)*self.chunksize])

        self.srch_pca = self.srch_pca.fit(self.srch_data)
        kpca_transform = self.srch_pca.transform(self.srch_data) 
        
        
        
        # foo = '_whitened_' if self.whiten else '_'
        # model_name = self.type + '_srch' + foo +'model.pkl'
        pickle.dump( self.srch_pca , open(self.model_name, "wb" ) )
        # pickle.dump(self.srch_pca, open( self.type+'_srch_model.pkl', "wb" ) )
                
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        return explained_variance_ratio
    
    def run_all(self,show=True,save=True):
        exp_var_ratio_props = None
        exp_var_ratio_srchs = None
        if self.only == 'prop' or self.only == None:

            exp_var_ratio_props = self.run_props()
            plt.figure(1)
            plt.plot(np.cumsum(exp_var_ratio_props))
            foo = 'Whitened' if self.whiten else ''
            plt.title(self.type+' ' +foo +  ' Principal Components Cumulative Explained Variance For Properties')
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
                
            # foo = 'whitened_' if self.whiten else '_'
            # plot_name = self.type + '_prop_' + foo +'model'
            plt.savefig('plots/'+self.type+'_'+foo+'_cum_explained_var_prop.png')
        if self.only== 'srch'  or self.only == None:

        
        
            
            exp_var_ratio_srchs = self.run_srchs()
            
            
            plt.figure(2)
            plt.plot(np.cumsum(exp_var_ratio_srchs))
            foo = 'Whitened' if self.whiten else ''
            plt.title(self.type+' ' + foo +  ' Principal Components Cumulative Explained Variance For Queries')
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            
            plt.savefig('plots/'+self.type+'_'+foo+'_cum_explained_var_srch.png')
        
        if show:
            plt.show()
        # if save:
        #     if self.only == 'prop' or self.only is None:
        #         plt.savefig('plots/IPCA_cum_explained_var_props.png')
        #         return exp_var_ratio_props,None
        #     if self.only == 'srch' or self.only is None:
        #         plt.savefig('plots/IPCA_cum_explained_var_srchs.png')
        #         return None,exp_var_ratio_srchs
        return exp_var_ratio_props, exp_var_ratio_srchs
if __name__ == "__main__":
    # data = DataTables()

    # pickle.dump( data, open( 'datatables_dummy.pkl', "wb" ) )


    # data = pickle.load( open(  'datatables_dummy.pkl', "rb" ) )

    # print(__doc__)



    # pca_type = 'search'


    only = 'srch'
    debug = False
    pca_type = 'Kernel' 
    # whiten = False
    whiten = True
    cluster_type = 'gmm'
    # pca_type = 'Kernel' 
    kernel_type = 'rbf'
    # data_type = 'prop'

    foo = '_dummy' if debug else ''

    data_path = 'DATA/after_'+cluster_type+'_'+only+foo+'.pkl' 
    if whiten:
        model_name = 'models/'+pca_type+'_whitened_after_'+cluster_type+'_'+only+foo
        plot_name = 'plots/'+pca_type+'_whitened_after_'+cluster_type+'_'+only+foo 
    
    else:

        model_name = 'models/'+pca_type+'_after_'+cluster_type+'_'+only+foo 
        plot_name = 'plots/'+pca_type+'_after_'+cluster_type+'_'+only+foo 
    # print(prop_data.head())
    # print(srchs_data.head())
    # print(prop_data.isnull().astype(int).sum())

    pcab = PCA_Benchmark(data_path,data_path,only=only,pca_type=pca_type,kernel_type = kernel_type,whiten=whiten,model_name=model_name)

    # pcab.run_props()
    # pcab.run_srchs()


    exp_var_ratio_props, exp_var_ratio_srchs = pcab.run_all(show=False,save=True)