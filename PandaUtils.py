from typing import Tuple
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class PandaUtils:
    def __init__(self, file_path: str):
        self.dataset: pd.DataFrame = pd.read_csv(file_path)
        self.columns: list = []
        self.__getColumns()

    def __getColumns(self):
        self.columns = self.dataset.iloc[:, :-1].columns

    def checkMissingData(self):
        print("Check for missing data")
        print(self.dataset.isna().sum())

    def transformNonNumericData(self):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        non_numeric_dataset = self.dataset.select_dtypes(exclude=numerics)
        columns = non_numeric_dataset.columns

        le = preprocessing.LabelEncoder()

        for i in columns:
            self.dataset[i] = le.fit_transform(self.dataset[i])

    def splitDataset(self) -> Tuple:
        x = self.dataset.iloc[:, :-1].values
        y = self.dataset.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=25)

        return x, y, x_train, x_test, y_train, y_test
