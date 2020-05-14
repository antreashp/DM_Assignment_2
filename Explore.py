from DataTables import *
import pandas as pd
import datetime

data = DataTables()
date_time = data.search['date_time'].dt.date
print(date_time)