import pandas as pd 


class DATA():
    def __init__(self,filename = 'dummy_data.pkl'):
        self.filename = filename
        self.data = pd.read_pickle(self.filename)


    def print_max_min_features(self):
        maxs = self.data.max()
        mins = self.data.min()
        
        for ((maxcolumnName, maxcolumnData),(mincolumnName, mincolumnData)) in     zip(self.data.max().iteritems(),self.data.min().iteritems()):
            print(maxcolumnName,': ')
            print('max: ',maxcolumnData,'min: ',mincolumnData)
            print()


if __name__ == "__main__":
    data = DATA()
    data.print_max_min_features()

