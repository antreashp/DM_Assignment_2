from DATA import DATA
import pandas as pd


class DataTables(DATA):
    search_pk = 'srch_id'
    search_attributes = ['date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating',
                         'visitor_hist_adr_usd', 'random_bool', 'srch_destination_id', 'srch_length_of_stay',
                         'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
                         'srch_saturday_night_bool', ]
    property_pk = 'prop_id'
    property_attributes = ['prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
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
    search_property_attributes = ['position', 'price_usd', 'click_bool', 'gross_bookings_usd',
       'booking_bool', 'orig_destination_distance']
    average_attributes = ['price_usd', 'gross_bookings_usd', 'orig_destination_distance']

    def __init__(self):
        super().__init__(filename='dummy_data.pkl')
        self.search_table()
        self.property_table()
        self.build_relations()

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
        for search_group in self.data.groupby(self.search_pk):
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
        for property_group in self.data.groupby(self.property_pk):
            property_id = property_group[1].iloc[0][self.property_pk]
            count = property_group[1][self.property_pk].count()
            num_click = (property_group[1]['click_bool'] == 1).sum()
            num_booking = (property_group[1]['booking_bool'] == 1).sum()
            average_attributes = [property_group[1][x].mean() for x in self.average_attributes]
            single_property = [property_id] + [property_group[1].iloc[0][x] for x in self.property_attributes]+ \
                            average_attributes + [count, num_click, num_booking]
            property.append(single_property)
        self.property = pd.DataFrame(property, columns=[self.search_pk] + self.property_attributes +
                                                   ['ave_' + x for x in self.average_attributes] +
                                                   ['search_count', 'num_clicks', 'num_bookings'])

    def build_relations(self):
        self.relations = {}
        for index, pair in self.data[[self.search_pk, self.property_pk]].iterrows():
            search_id = pair[self.search_pk]
            property_id = pair[self.property_pk]
            if search_id not in self.relations.keys():
                self.relations[search_id] = set()
            self.relations[search_id].add(property_id)

    def find_random_keys(self):
        # random_search_keys = set([x for x in self.data[(self.data['random_bool'] == False)][self.search_pk]])
        self.X_random = self.data[(self.data['random_bool'] == False)]
        print(self.X_random)


if __name__ == '__main__':
    data = DataTables()
    # data.search_table()
    # print(data.check_uniqueness(data.search_pk, data.search_attributes, verbose=False))
    # print(data.check_uniqueness(data.property_pk, data.property_attributes,  verbose=True))
    # data.search_table()
    # data.property_table()
    # print(data.data[data.data['prop_id'] == 7880])
    # print(data.property)
    # data.build_relations()
    data.find_random_keys()
