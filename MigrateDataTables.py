from DataTables import *
import pickle

new_data = DataTables('dummy.pkl', negative_data=0)

old_data = pickle.load(open('old_data', 'rb'))
objects = {
    'search', 'property', 'keys', 'non_random_keys', 'random_keys'
}

new_data.search = old_data.search

for object in objects:
    eval('new_data.' + object + ' = old_data.' + object)

pickle.dump(new_data, 'data.pkl')
