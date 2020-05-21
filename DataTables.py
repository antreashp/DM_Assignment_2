from DATA import DATA
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from utils import *
import pickle
import sys

class DataTables(DATA):
    search_pk = 'srch_id'
    search_attributes = ['date_time', 'visitor_location_country_id', 'visitor_hist_starrating',
                         'visitor_hist_adr_usd', 'random_bool', 'srch_destination_id', 'srch_length_of_stay',
                         'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                         'srch_saturday_night_bool', ]
    property_pk = 'prop_id'
    property_attributes = ['prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
                           'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
                           'promotion_flag', 'srch_query_affinity_score', 'has_prop_starrating']
    search_property_attributes = ['price_usd', 'click_bool', 'gross_bookings_usd',
       'booking_bool', 'orig_destination_distance']
    average_attributes = ['price_usd', 'orig_destination_distance']
    test_attributes = ['position', 'click_bool', 'gross_bookings_usd',
       'booking_bool']
    features = search_attributes + property_attributes + search_property_attributes
    target = 'position'

    destination = 'srch_destination_id'
    country = 'prop_country_id'

    def __init__(self, filename='dummy_data.pkl', negative_data=100, test=False):
        super().__init__(filename=filename)
        self.negative_data = negative_data
        if test:
            self.data.columns = list(self.data.columns) + self.test_attributes
        else:
            self.relevance()

        self.build_relations()

        self.data['srch_query_affinity_score'] = self.data['srch_query_affinity_score'].fillna(0)
        self.data['prop_starrating'] = self.data['prop_starrating'].fillna(0)
        self.data['has_prop_starrating'] = self.data['prop_starrating'].apply(lambda x: 1 if x else 0)
        self.search_table()
        self.property_table()

        self.property_price()
        self.preprocess()
        self.preprocess_datetime()

        self.property = self.property.fillna(np.nan)
        self.property['prop_review_score'] = self.property['prop_review_score'].fillna(0)
        self.search['visitor_hist_starrating'] = self.search['visitor_hist_starrating'].fillna(2.5)
        self.search = self.search.fillna(np.nan)
        # self.destination_keys = self.data[[self.destination, self.country]]

        self.data = self.data[[self.target, self.search_pk, self.property_pk, 'random_bool']]
        self.features = []

        if not test:
            self.add_negative_data()
            self.random_keys = self.data[(self.data['random_bool'] == True)][[self.search_pk, self.property_pk]]
            self.non_random_keys = self.data[(self.data['random_bool'] == False)][[self.search_pk, self.property_pk]]

        self.keys = self.data[[self.search_pk, self.property_pk]]

    def check_uniqueness(self, pk, attributes, verbose=True):
        if verbose:
            for group in self.data.groupby(pk):
                counts = []
                for x in attributes:
                    counts.append(group[1][x].nunique())
                print(group[1].iloc[0][pk], counts)

        return all(group[1][x].nunique() <= 1 for group in self.data.groupby(pk) for x in attributes)

    def add_negative_data(self):
        for i in tqdm(range(self.negative_data)):
            search_key = self.data.iloc[random.randint(0, self.data.shape[0])][self.search_pk]
            negative_property_keys = list(set(list(self.data[self.property_pk])) - self.relations[search_key])
            negative_property_key = random.choice(negative_property_keys)
            new_row = {}
            new_row[self.search_pk] = search_key
            new_row[self.property_pk] = negative_property_key
            new_row['random_bool'] = 0
            # for search_feature in self.search_attributes:
            #     new_row[search_feature] = self.data[(
            #         self.data[self.search_pk] == search_key
            #     )].iloc[0][search_feature]
            # for property_feature in self.property_attributes:
            #     new_row[property_feature] = self.data[(
            #         self.data[self.property_pk] == negative_property_key
            #     )].iloc[0][property_feature]
            # for feature in self.search_property_attributes:
            #     new_row[feature] = np.nan
            new_row[self.target] = -1
            self.data = self.data.append(new_row, ignore_index=True)

    def search_table(self):
        search = []
        self.search_groups = self.data.groupby(self.search_pk)
        for search_group in tqdm(self.data.groupby(self.search_pk)):
            search_id = search_group[1].iloc[0][self.search_pk]
            count = search_group[1][self.search_pk].count()
            single_search = [search_id] + [search_group[1].iloc[0][x] for x in self.search_attributes]+ \
                            [count]
            search.append(single_search)
        self.search = pd.DataFrame(search, columns=[self.search_pk] + self.search_attributes +
                                                   ['properties_count'])
        self.search_keys = self.search[self.search_pk]

    def property_table(self):
        property = []
        for property_group in tqdm(self.data.groupby(self.property_pk)):
            property_id = property_group[1].iloc[0][self.property_pk]
            count = property_group[1][self.property_pk].count()
            ave_price = np.log(property_group[1]['price_usd'].mean() + 1)
            single_property = [property_id] + [property_group[1].iloc[0][x] for x in self.property_attributes]+ \
                            [count] + [ave_price]
            property.append(single_property)
        self.property = pd.DataFrame(
            property, columns=
            [self.property_pk] +
            self.property_attributes +
            ['search_count', 'ave_price']
        )
        self.property_keys = self.property[self.property_pk]

    def build_relations(self):
        self.relations = {}
        for index, pair in tqdm(self.data[[self.search_pk, self.property_pk]].iterrows()):
            search_id = pair[self.search_pk]
            property_id = pair[self.property_pk]
            if search_id not in self.relations.keys():
                self.relations[search_id] = set()
            self.relations[search_id].add(property_id)

    def random_features(self):
        return self.data[(self.data['random_bool'] == True)][self.features]

    def X(self):
        return self.data[(self.data['random_bool'] == False)][self.features]

    def y(self):
        return self.data[(self.data['random_bool'] == False)][self.target]
    
    def preprocess_datetime(self):
        # self.search = self.search.dropna(subset=['date_time'])
        self.search['date_time'] =  pd.to_datetime(self.search['date_time'])
        self.search['year'] = pd.DatetimeIndex(self.search['date_time']).year
        self.search['month'] = pd.DatetimeIndex(self.search['date_time']).month
        self.search['day'] = pd.DatetimeIndex(self.search['date_time']).day
        self.search['weekday'] = self.search.date_time.dt.day_name
        self.search['hours'] = pd.DatetimeIndex(self.search['date_time']).hour
        self.search = self.search.drop('date_time', axis=1)

        self.search = one_hot(self.search, 'year')
        self.search['last_week_of_nov'] = np.where(
            (self.search['month'] == 11) & (self.search['day'] > 22), 1, 0
        )
        self.search['week_before_christmas'] = np.where(
            (self.search['month'] == 12) & (self.search['day'] > 17), 1, 0
        )
        self.search['hours'] = self.search['hours'].apply(lambda x: int(((x + 2) % 24)/ 8))
        self.search = one_hot(self.search, 'hours')

        self.search = one_hot(self.search, 'month')
        self.search = one_hot(self.search, 'weekday')



    def save(self, data_name='data.pkl', search_name='search.pkl', property_name='property.pkl'):
        self.data.to_pickle(data_name)
        self.search.to_pickle(search_name)
        self.property.to_pickle(property_name)

    def merge(self, search, property):
        search = pd.concat([self.search_keys, search], axis=1)
        property = pd.concat([self.property_keys, property], axis=1)
        merged_data = pd.merge(
            self.keys, self.data[[self.property_pk, self.search_pk, self.target]],
            on=[self.search_pk, self.property_pk]
        )
        merged_data = pd.merge(merged_data, search, on=self.search_pk)
        merged_data = pd.merge(merged_data, property, on=self.property_pk)

        self.data = merged_data
        self.features = list(merged_data.columns)
        return merged_data

    def non_random(self):
        return self.data[~(
            self.data[self.search_pk].isin(self.non_random_keys[self.search_pk]) &
            self.data[self.property_pk].isin(self.non_random_keys[self.property_pk])
        )]

    def random(self):
        return self.data[~(
            self.data[self.search_pk].isin(self.random_keys[self.search_pk]) &
            self.data[self.property_pk].isin(self.random_keys[self.property_pk])
        )]

    def property_price(self):
        search_price_groups = self.data[[
            self.search_pk, self.property_pk, 'srch_adults_count', 'srch_children_count', 'srch_room_count',
            'price_usd'
        ]].copy()
        search_price_groups.loc[:, 'price_score'] = 1 - search_price_groups.groupby(
            ['srch_adults_count', 'srch_children_count', 'srch_room_count']
        )['price_usd'].rank(pct=True)
        property_price_scores = search_price_groups.groupby(self.property_pk)[
            'price_score'].mean().reset_index()

        self.data = self.data.drop('price_usd', axis=1)
        self.property = pd.merge(self.property, property_price_scores, on=self.property_pk)

    def normalize(self):
        # methods = {
        #     'prop_starrating': lambda x: x / 1,
        #     'prop_review_score': lambda x: x / 1,
        #     'prop_location_score1': lambda x: x / 1,
        # }

        # for key, method in methods.items():
        #     self.data[key] = self.data[key].apply(method)

        price_ranges = {
            100: 1,
            200: 2,
            300: 3,
            400: 4,
        }
        select_price_range = lambda x: [price_ranges[i] for i in price_ranges.keys() if x < i][-1] if x < 400 else \
            0 if math.isnan(x) else 5
        self.search['visitor_hist_adr_usd'] = self.search['visitor_hist_adr_usd'].apply(select_price_range)

    def preprocess(self):
        self.normalize()
        # self.property = one_hot(self.property, 'prop_country_id')
        self.property['prop_location_score2'] = self.property['prop_location_score2'].fillna(0)
        self.property['prop_location_score1'] += self.property['prop_location_score2']
        self.property = self.property.drop(['prop_location_score2'], axis=1)
        # self.search = one_hot(self.search, 'visitor_location_country_id')
        self.search = one_hot(self.search, 'visitor_hist_adr_usd')


    def relevance(self):
        method = lambda x: 5 if x['click_bool'] == 1 and x['booking_bool'] == 1 else 1 if x['click_bool'] else 0
        # print('here')

        # self.data['relevance'] = np.nan
        # for i, row in tqdm(self.data.iterrows()):
        #     # print(5 if row['click_bool'] == 1 and row['booking_bool'] == 1 else 1 if row['click_bool'] else 0)
        #     # print(i)
        #     row['relevance'] =  5 if row['click_bool'] == 1 and row['booking_bool'] == 1 else 1 if row['click_bool'] else 0
        




        # self.data['relevance'].map(lambda x: 5 if self.data['click_bool'] == 1 and self.data['booking_bool'] == 1 else 1 if self.data['click_bool'] else 0)
        # print('here')
        
        self.data['relevance'] = self.data[['click_bool','booking_bool']].apply(method, axis=1)
        # print('here')
        self.target = 'relevance'
        # print(self.data.apply(method, axis=1))
        
    def output_data(self, filename, discard_random_data=False):
        columns = list(self.data.columns)
        columns.remove(self.target)
        columns.remove(self.property_pk)
        columns = [self.target] + columns

        if discard_random_data:
            output_data = self.non_random()[columns]
        else:
            output_data = self.data[columns]
        output_data.to_pickle(filename)

    def save_search_property(self, search_path, property_path):
        self.property = self.property.drop(self.property_pk, axis=1)
        self.search = self.search.drop(self.search_pk, axis=1)
        self.property.to_pickle(property_path)
        self.search.to_pickle(search_path)
        del self.search
        del self.property

    def post_processing(self, y_predict, save_path):
        y_predict = pd.Series(y_predict, name='y_predict')
        prediction = pd.concat([self.keys, y_predict], axis=1)
        prediction = prediction.groupby(self.search_pk, as_index=False).apply(pd.DataFrame.sort_values, 'y_predict', ascending=False)
        prediction = prediction.reset_index()[[self.search_pk, self.property_pk]]
        # print(prediction)
        prediction.to_csv(save_path, index=False)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    filename = 'all_data.pkl'
    data = DataTables(filename = filename,negative_data=10000)


    # exit()
    if 'dummy' in filename:
        data.save_search_property('dummy_search.pkl', 'dummy_property.pkl')
        pickle.dump(data, open('dummy_datatables.pkl', 'wb'))
    else:
        data.save_search_property('all_search.pkl', 'all_property.pkl')
        pickle.dump(data, open('all_datatables.pkl', 'wb'))
    exit()
    del data

    data = pickle.load(open('datatables.pkl', 'rb'))
    search = pickle.load(open('search.pkl', 'rb'))
    property = pickle.load(open('property.pkl', 'rb'))
    data.merge(search, property)
    data.output_data('output.pkl')

    print(pickle.load(open('output.pkl', 'rb')))

    data.save_search_property('', '')
    del data
    '''After Imputation, cluster, PCA...'''
    data = pickle.load(open('', 'rb'))

    search_data_path = 'search.pkl'
    property_data_path = 'property.pkl'
    search = pickle.load(open(search_data_path, 'rb'))
    property = pickle.load(open(property_data_path, 'rb'))


    data.merge(search, property)


    output_data_path = ''
    data.output_data(output_data_path)

    '''Train on this output'''
    data_test = DataTables(filename = 'test_set_VU_DM.csv', negative_data=0, test=True)
    data_test.save_search_property('', '')
    '''Test PCA'''
    data_test = pickle.load(open('', 'rb'))

    search_data_path = 'search.pkl'
    property_data_path = 'property.pkl'
    search = pickle.load(open(search_data_path, 'rb'))
    property = pickle.load(open(property_data_path, 'rb'))
    data_test.merge(search, property)


    '''Running the model, prediction -> y_predict'''
    y_predict = [[0, 1, 5][random.randint(0,2)] for x in range(700)] # dummy

    data_test.post_processing(y_predict, 'test_result.csv')

