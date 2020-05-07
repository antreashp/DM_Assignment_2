

# Import pandas 
import pandas as pd 

from sklearn import preprocessing
df = pd.read_pickle('dummy_data.pkl')
# print(df.head(28))
# print(df.columns)

# print(df[df['srch_id']==4]['prop_id'] )

# print(len(df['srch_id'].unique()))
# print(df[df['srch_id']==4].max().dropna())
srchs_id = df['srch_id'].unique()
for s_id in srchs_id:
    s_id = 4
    temp_df = df[df['srch_id']==s_id].max().dropna().drop(['srch_id','date_time','prop_id','srch_destination_id'])
    y = temp_df['position']
    print(y)
    # print(temp_df)
    # temp_df = temp_df
    # temp_df = temp_df.max().dropna()
    # temp_df = temp_df.drop(['date_time'])
    print(temp_df)
    exit()
records = []

print(df[df['srch_id']==4])

print(df[df['srch_id']==4].max())
print(df[df['srch_id']==4].max().dropna())


# print(df[df['srch_id']==4].max().dropna())
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(df)
# df = pd.DataFrame(x_scaled)
# print(df[df['srch_id']==4].max())




