from DataTables import *



if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    filename = 'dummy_test_data.pkl'
    data = DataTables('dummy_test_data.pkl', test=True)

    # exit()
    if 'dummy' in filename:
        data.save_search_property('dummy_test_search.pkl', 'dummy_test_property.pkl')
        pickle.dump(data, open('dummy_test_datatables.pkl', 'wb'))
    else:
        data.save_search_property('all_test_search.pkl', 'all_test_property.pkl')
        pickle.dump(data, open('all_test_datatables.pkl', 'wb'))
    exit()
    del data

    data = pickle.load(open('dummy_test_datatables.pkl', 'rb'))
    search = pickle.load(open('dummy_test_search.pkl', 'rb'))
    property = pickle.load(open('dummy_test_property.pkl', 'rb'))
    data.merge(search, property)
    data.output_data('output_test.pkl')




    '''Test PCA'''


    '''Running the model, prediction -> y_predict'''
    y_predict = [[0, 1, 5][random.randint(0,2)] for x in range(1000)] # dummy

    data.post_processing(y_predict, 'test_result.csv')
