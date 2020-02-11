from sklearn.model_selection import train_test_split

class TrainTestSplitter:
    def __init__(self, train_column, test_column):
        self.train_column = train_column
        self.test_column = test_column

    def split(self):
        return train_test_split(self.train_column, self.test_column, test_size=0.2, random_state=7)