from DATA import DATA
import pandas as pd
from tqdm import tqdm

class DataTables(DATA):
    search_pk = 'srch_id'
    search_attributes = ['date_time', 'visitor_location_country_id', 'visitor_hist_starrating',
                         'visitor_hist_adr_usd', 'random_bool', 'srch_destination_id', 'srch_length_of_stay',
                         'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                         'srch_saturday_night_bool', ]
    property_pk = 'prop_id'
    property_attributes = ['prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
                           'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
                           'promotion_flag', 'srch_query_affinity_score', 'comp1_rate', 'comp1_inv',
                           'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
                           'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
                           'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
                           'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
                           'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
                           'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
                           'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
                           'comp8_rate_percent_diff', ]
    search_property_attributes = ['price_usd', 'click_bool', 'gross_bookings_usd',
       'booking_bool', 'orig_destination_distance']
    average_attributes = ['price_usd', 'gross_bookings_usd', 'orig_destination_distance']
    features = [search_pk] + search_attributes + [property_pk] + property_attributes + search_property_attributes
    target = 'position'

    def __init__(self):
        super().__init__(filename='dummy_data.pkl')
        self.search_table()
        # self.preprocess_datetime()
        self.property_table()
        self.build_relations()

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
        self.property = pd.DataFrame(property, columns=[self.property_pk] + self.property_attributes +
                                                   ['ave_' + x for x in self.average_attributes] +
                                                   ['search_count', 'num_clicks', 'num_bookings'])

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
        self.features.extend(['year', 'month', 'day', 'weekday', 'hours', 'seconds'])

    def save(self, data_name='data.pkl', search_name='search.pkl', property_name='property.pkl'):
        self.data.to_pickle(data_name)
        self.search.to_pickle(search_name)
        self.property.to_pickle(property_name)

    def merge(self, search, property, search_property):
        search = pd.concat([self.search[self.search_pk], search], axis=1)
        property = pd.concat([self.property[self.property_pk], property], axis=1)
        search_property = pd.concat([self.keys, search_property])

        merged_data = pd.merge(self.keys, search, on=self.search_pk)
        merged_data = pd.merge(merged_data, property, on=self.property_pk)
        merged_data = pd.merge(merged_data, search_property, on=[self.search_pk, self.property_pk])
        return merged_data

        # here are some code that we need if we keep the data after merging in this class:
        self.data = merged_data
        self.features = list(merged_data.columns)

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


if __name__ == '__main__':
    data = DataTables()
    data.merge(data.search[data.search_attributes], data.property[data.property_attributes], data.data[data.search_property_attributes])
