from managers.data import DataManager
from classes.fakenews import FakeNews
from helpers.splitter import TrainTestSplitter

path = 'news.csv'

def main():
    data_frame = DataManager(path).getFrameFromCSV()

    label_column = data_frame.label
    text_column = data_frame['text']

    x_train, x_test, y_train, y_test = TrainTestSplitter(text_column, label_column).split()

    FakeNews(x_train, x_test, y_train, y_test).predictFakeNews()

if __name__ == '__main__':
    main()