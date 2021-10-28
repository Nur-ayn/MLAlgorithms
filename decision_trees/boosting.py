import numpy as np
import matplotlib.pyplot as plt
from decision_tree_test import Tree, Node, DataSet

class ZeroClassifier:
    def predict(self, inputs):
        return np.zeros(len(inputs))

inputs = np.array([[x] for x in range(-50,50)])
labels = np.array([x**2 + 0.5*x**3 + 7*x for x in range(-50, 50)])
sample_dataset = DataSet(inputs, labels)

x = [ point[0] for point in inputs ]
y = labels

plt.plot(x,y, zorder=0)

# start with a tree that predicts 0 for every label


class GradientBoostedRegressionTrees:
    def __init__(self, dataset, alpha=0.1):
        self.inputs = dataset.inputs
        self.H = [ZeroClassifier()]
        self.alpha = alpha
        self.data = dataset

    def train(self, iterations=100):
        for iteration in range(iterations):
            new_labels = self.negative_dL_dH(self.data)
            new_dataset = DataSet(self.inputs, new_labels) #optimise this, maybe not using DataPoint but rather Dataset
            new_tree = Tree(Node(new_dataset), depth=5)
            new_tree.train()
            self.H.append(new_tree)

    def predict(self, inputs):
        separate_predictions = self.alpha*np.array([ weak_learner.predict(inputs) for weak_learner in self.H ])
        #print('separate predictions', separate_predictions, separate_predictions)
       #print('sum', np.sum(separate_predictions, axis=0))
        return np.sum(separate_predictions, axis=0)

    def loss(self, datapoints):
        actual_labels = np.array([ datapoint.label for datapoint in datapoints ])
        predicted_labels = self.predict([datapoint.input for datapoint in datapoints])
        return 1/len(datapoints)*np.sum( (actual_labels - predicted_labels)**2 )

    def negative_dL_dH(self, dataset):
        actual_labels = np.array(dataset.labels)
        predicted_labels = self.predict(dataset.inputs)
        return actual_labels - predicted_labels

if __name__ == '__main__':
    gbrt = GradientBoostedRegressionTrees(sample_dataset)
    gbrt.train()
    x = np.linspace(-50, 50, 1000)
    y = gbrt.predict([[i] for i in x])
    print(y)

    plt.plot(x,y)
    plt.show()
    