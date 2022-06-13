from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class TrainData:
    def __init__(self, x, y, x_train, y_train):
        self.random_classifier = RandomForestClassifier()
        self.x = x
        self.y = y
        self.x_train = x_train
        self.y_train = y_train

    def gridSearch(self, splits=5):
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 300, 450, 500]
        }
        # Instantiate the grid search model
        cv = StratifiedKFold(n_splits=splits)
        grid_search = GridSearchCV(estimator=self.random_classifier, param_grid=param_grid, scoring='accuracy',
                                   cv=cv, n_jobs=-1, verbose=2)
        grid_search.fit(self.x, self.y)

        return grid_search.best_params_

    def fitData(self, params):
        classifier = RandomForestClassifier(**params)
        classifier.fit(self.x_train, self.y_train)

        return classifier
