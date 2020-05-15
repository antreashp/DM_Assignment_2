
# from Experiment import Experiment
from DataTables import DataTables
import pandas as pd



data = DataTables().data
print(data)


exit()
print(data.search.head(5))
print(data.property.head(5))
def preprocess_datetime(self):

    meh = data.search.merge(data.property, how ='outer')
    meh = meh.dropna(subset=['date_time'])
    # print(meh)
    meh['date_time'] =  pd.to_datetime(meh['date_time'])
    # print(meh['date'][0].month)
    # print(meh.head())
    meh['year'] = pd.DatetimeIndex(meh['date_time']).year
    meh['month'] = pd.DatetimeIndex(meh['date_time']).month
    meh['day'] = pd.DatetimeIndex(meh['date_time']).day
    meh['weekday'] = meh.date_time.dt.weekday_name

    meh['hours'] = pd.DatetimeIndex(meh['date_time']).hour
    meh['seconds'] = pd.DatetimeIndex(meh['date_time']).second
print(meh.head())
# date_time = pd.to_datetime(meh['date_time'])
# print(date_time.head(5))
# print(date_time.year)

exit()
# train_data = 
# test_data = 
# model_type = 'LambdaMART'
# model_type = 'LambdaRank'
# # model_type = 'LR'

# # from DataTables
# exp = Experiment(train_data,test_data,model_type=model_type)
# outs = exp.train()
# outs = exp.eval()
# exp.plot()