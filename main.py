from managers.data import DataManager
from classes.fakenews import FakeNews
from helpers.splitter import TrainTestSplitter

path = 'news.csv'

def main():
    # Download and parse to pandas dataframe
    data_frame = DataManager(path).getFrameFromCSV()

    label_column = data_frame.label
    text_column = data_frame['text']

    x_train, x_test, y_train, y_test = TrainTestSplitter(text_column, label_column).split()

    # Create fake news prediction model
    FakeNews(x_train, x_test, y_train, y_test).createModel()

if __name__ == '__main__':
    main()