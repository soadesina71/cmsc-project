from PandaUtils import PandaUtils
from Visualize import Visualize
from TrainData import TrainData
from TestData import TestData

# Invoking the PandaUtils class and importing the dataset as a dataframe
pandaUtils = PandaUtils("dataset.csv")

dataset = pandaUtils.dataset

# Invoking the Visualize class with the dataset
visualize = Visualize(dataset)

# Calling the visualizeDataset method to plot various graphs
visualize.visualizeDataset()

# Transforming the categorical columns to numeric forms
pandaUtils.transformNonNumericData()

# Splitting the dataset into training and testing
x, y, x_train, x_test, y_train, y_test = pandaUtils.splitDataset()

# Calling the TrainData class
trainData = TrainData(x, y, x_train, y_train)

# Getting the best hyper-parameters with the Grid Search
best_params = trainData.gridSearch(2)

# Fitting and training the model with the best hyper-parameters
fitted_classifier = trainData.fitData(best_params)

# Invoking the TestData class with the trained model and test data
testData = TestData(fitted_classifier, x_test, y_test)

# Predicting the test data
y_pred = testData.predictTestData()

# Saving the predicted test data result into a CSV file
testData.savePredictedData(y_pred)

# Getting the model performance
testData.getModelPerformance(y_pred)

# Plotting the important features in the dataset
testData.plotImportantFeaturesGraph(pandaUtils.columns)

# Plotting the performance graphs for the model
testData.plotPerformanceGraphs(visualize.plt)

visualize.plt.show()
