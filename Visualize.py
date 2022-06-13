import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

class Visualize:
    def __init__(self, dataset: DataFrame):
        self.dataset = dataset
        self.plt = plt

    def label_function(self, val: float):
        return f'{val / 100 * len(self.dataset):.0f}\n{val:.0f}%'

    def visualizeDataset(self):
        fig, axs = plt.subplots(1, 3, figsize=(32, 6))
        axs = axs.ravel()

        self.dataset.groupby("protocol_type").size().plot(kind="pie", autopct=self.label_function,
                                                          textprops={'fontsize': 12}, colors=['gold', 'skyblue'],
                                                          ax=axs[0])
        self.dataset.groupby("class").size().plot(kind="pie", autopct=self.label_function, textprops={'fontsize': 12},
                                                  ax=axs[1])

        axs[0].set_title("Plot showing the Protocol Types")
        axs[1].set_title("Plot showing the Classes")

        plt.show(block=False)

        plt.figure()
        sns.set(rc={'figure.figsize':(18,8.27)})
        service_plot = sns.histplot(data=self.dataset, x="service")
        service_plot.tick_params(axis="x", rotation=90)
        service_plot.set_title("Plot showing the services")
