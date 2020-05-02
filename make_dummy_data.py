# Import pandas 
import pandas as pd 



filename = 'training_set_VU_DM.csv'
# reading csv file  
df = pd.read_csv(filename, nrows=1000)
print(df)


df.to_pickle('dummy_data.pkl') 