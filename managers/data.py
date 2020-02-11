import pandas as pd

class DataManager:
    def __init__(self, path):
        self.path = path

    def getFrameFromCSV(self):
        data_frame = pd.read_csv(self.path)
        print(data_frame.head())
        return data_frame
