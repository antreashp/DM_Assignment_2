from DATA import DATA
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from utils import *

class DataTables(DATA):
    search_pk = 'srch_id'
    search_attributes = ['date_time', 'visitor_location_country_id', 'visitor_hist_starrating',
                         'visitor_hist_adr_usd', 'random_bool', 'srch_destination_id', 'srch_length_of_stay',
                         'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                         'srch_saturday_night_bool', ]
    property_pk = 'prop_id'
    property_attributes = ['prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
                           'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
                           'promotion_flag', 'srch_query_affinity_score',]
    search_property_attributes = ['price_usd', 'click_bool', 'gross_bookings_usd',
       'booking_bool', 'orig_destination_distance']
    average_attributes = ['price_usd', 'gross_bookings_usd', 'orig_destination_distance']
    features = [search_pk] + search_attributes + [property_pk] + property_attributes + search_property_attributes
    target = 'position'

    destination = 'srch_destination_id'
    country = 'prop_country_id'

    def __init__(self, negative_data=100):
        super().__init__(filename='dummy_data.pkl')
        self.negative_data = negative_data
        self.search_table()
        self.property_table()
        self.build_relations()
        # self.preprocess_datetime()
        self.preprocess()
        self.preprocess_datetime()
        self.property_price()

        self.random_keys = self.data[(self.data['random_bool'] == True)][[self.search_pk, self.property_pk]]
        self.non_random_keys = self.data[(self.data['random_bool'] == False)][[self.search_pk, self.property_pk]]
        self.keys = self.data[[self.search_pk, self.property_pk]]
        self.destination_keys = self.data[[self.destination, self.country]]

    def check_uniqueness(self, pk, attributes, verbose=True):
        if verbose:
            for group in self.data.groupby(pk):
                counts = []
                for x in attributes:
                    counts.append(group[1][x].nunique())
                print(group[1].iloc[0][pk], counts)

        return all(group[1][x].nunique() <= 1 for group in self.data.groupby(pk) for x in attributes)

    def add_negative_data(self):
        for i in range(self.negative_data):
            search_key = self.data.iloc[random.randint(0, self.data.shape[0])][self.search_pk]
            negative_property_keys = list(set(list(self.data[self.property_pk])) - self.relations[search_key])
            negative_property_key = random.choice(negative_property_keys)
            new_row = {}
            new_row[self.search_pk] = max(list(self.data[self.search_pk])) + 1
            new_row[self.property_pk] = max(list(self.data[self.property_pk])) + 1
            for search_feature in self.search_attributes:
                new_row[search_feature] = self.data[(
                    self.data[self.search_pk] == search_key
                )].iloc[0][search_feature]
            for property_feature in self.property_attributes:
                new_row[property_feature] = self.data[(
                    self.data[self.property_pk] == negative_property_key
                )].iloc[0][property_feature]
            for feature in self.search_property_attributes:
                new_row[feature] = np.nan
            new_row[self.target] = 0
            print(new_row)
            self.data = self.data.append(new_row, ignore_index=True)
        print(self.data)

    def search_table(self):
        search = []
        self.search_groups = self.data.groupby(self.search_pk)
        for search_group in tqdm(self.data.groupby(self.search_pk)):
            search_id = search_group[1].iloc[0][self.search_pk]
            count = search_group[1][self.search_pk].count()
            num_click = (search_group[1]['click_bool'] == 1).sum()
            num_booking = (search_group[1]['booking_bool'] == 1).sum()
            average_attributes = [search_group[1][x].mean() for x in self.average_attributes]
            single_search = [search_id] + [search_group[1].iloc[0][x] for x in self.search_attributes]+ \
                            average_attributes + [count, num_click, num_booking]
            search.append(single_search)
        self.search = pd.DataFrame(search, columns=[self.search_pk] + self.search_attributes +
                                                   ['ave_' + x for x in self.average_attributes] +
                                                   ['properties_count', 'num_clicks', 'num_bookings'])

    def property_table(self):
        property = []
        for property_group in tqdm(self.data.groupby(self.property_pk)):
            property_id = property_group[1].iloc[0][self.property_pk]
            count = property_group[1][self.property_pk].count()
            num_click = (property_group[1]['click_bool'] == 1).sum()
            num_booking = (property_group[1]['booking_bool'] == 1).sum()
            average_attributes = [property_group[1][x].mean() for x in self.average_attributes]
            single_property = [property_id] + [property_group[1].iloc[0][x] for x in self.property_attributes]+ \
                            average_attributes + [count, num_click, num_booking]
            property.append(single_property)
        self.property = pd.DataFrame(
            property, columns=
            [self.property_pk] +
            self.property_attributes +
            ['ave_' + x for x in self.average_attributes] +
            ['search_count', 'num_clicks', 'num_bookings']
        )

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
        self.search['weekday'] = self.search.date_time.dt.weekday_name
        self.search['hours'] = pd.DatetimeIndex(self.search['date_time']).hour
        self.search['seconds'] = pd.DatetimeIndex(self.search['date_time']).second
        self.search.drop('date_time', axis=1)
        self.features.extend(['year', 'month', 'day', 'weekday', 'hours', 'seconds'])
        self.features.remove('date_time')

        self.search['year'] = one_hot(self.search, 'year')
        self.search['last_week_of_nov'] = np.where(
            (self.search['month'] == 11) & (self.search['day'] > 22), 1, 0
        )
        self.search['week_before_christmas'] = np.where(
            (self.search['month'] == 12) & (self.search['day'] > 17), 1, 0
        )

        self.search['month'] = one_hot(self.search, 'month')
        self.search['weekday'] = one_hot(self.search, 'weekday')


    def save(self, data_name='data.pkl', search_name='search.pkl', property_name='property.pkl'):
        self.data.to_pickle(data_name)
        self.search.to_pickle(search_name)
        self.property.to_pickle(property_name)

    def merge(self, search, property, search_property):
        search = pd.concat([self.search[self.search_pk], search], axis=1)
        property = pd.concat([self.property[self.property_pk], property], axis=1)
        # search_property = pd.concat([self.keys, search_property])

        merged_data = pd.merge(self.keys, search, on=self.search_pk)
        merged_data = pd.merge(merged_data, property, on=self.property_pk)
        # search_property_table discarded
        # merged_data = pd.merge(merged_data, search_property, on=[self.search_pk, self.property_pk])

        # here are some code that we need if we keep the data after merging in this class:
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
        self.property = pd.merge(self.property, property_price_scores, on=self.property_pk)

    def normalize(self):
        methods = {
            'prop_starrating': lambda x: x / 5,
            'prop_review_score': lambda x: x / 5,
            'prop_location_score1': lambda x: x / 10,
        }

        for key, method in methods.items():
            self.data[key] = self.data[key].apply(method)

        price_ranges = {
            100: 1,
            200: 2,
            300: 3,
            400: 4,
        }
        select_price_range = lambda x: [price_ranges[i] for i in price_ranges.keys() if x < i][-1] if x < 400 else \
            np.nan if math.isnan(x) else 5
        self.data['visitor_hist_adr_usd'] = self.data['visitor_hist_adr_usd'].apply(select_price_range)


    def preprocess(self):
        self.property = one_hot(self.property, 'prop_country_id')
        self.search = one_hot(self.search, 'visitor_location_country_id')
        self.normalize()

if __name__ == '__main__':
    data = DataTables()
    data.negative_data = 1
    # data.add_negative_data()
    # data.one_hot(data.property, 'prop_country_id')
    # data.preprocess()