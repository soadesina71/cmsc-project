from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve, plot_precision_recall_curve
from numpy import round, savetxt
from pandas import DataFrame

class TestData:
    def __init__(self, classifier, x_test, y_test):
        self.classifier = classifier
        self.x_test = x_test
        self.y_test = y_test

    def predictTestData(self):
        y_pred = self.classifier.predict(self.x_test)

        return y_pred

    def savePredictedData(self, y_pred):
        file = open("prediction.csv", "w")
        savetxt(file, y_pred, fmt='%1.2f')
        file.close()

        return

    def getModelPerformance(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion metrics: ", cm)

        acc_sc = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: ", acc_sc)

        return

    def plotImportantFeaturesGraph(self, columns):
        score = round(self.classifier.feature_importances_,3)
        features = DataFrame({'feature': columns,'importance': score})
        features = features.sort_values('importance',ascending=False).set_index('feature')

        features.plot.bar()
        return

    def plotPerformanceGraphs(self, plt):
        plot_roc_curve(self.classifier, self.x_test, self.y_test) 
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='Random Guessing')
        plt.legend()

        plot_precision_recall_curve(self.classifier, self.x_test, self.y_test)

        plt.show(block = False)
        return 