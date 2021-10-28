import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, List


def split_data(dataset, value):
    left, right = dataset.inputs < value, dataset.inputs >= value
    i_left, i_right = np.where(left)[0], np.where(right)[0]
    left_inputs, right_inputs = dataset.inputs[i_left], dataset.inputs[i_right]
    left_labels, right_labels = dataset.labels[i_left], dataset.labels[i_right]

    return DataSet(left_inputs, left_labels), DataSet(right_inputs, right_labels)


@dataclass
class DataSet:
    inputs: Any
    labels: Any
    weights: Any = None




class Node:
    def __init__(self, data):
        self.dimension = None
        self.split_value = None
        self.data = data
        self.mean = np.mean(data.labels) #adjust
        self.left = None
        self.right = None 

    def next_node(self, test_point):
        if self.split_value is None:
            return self.mean

        if test_point[self.dimension] < self.split_value:
            return self.left.next_node(test_point)
        if test_point[self.dimension] >= self.split_value:
            return self.right.next_node(test_point)

    def best_split(self):
        '''loops through each dimension of the inputs and each split of the data and returns the one with
            minimal square loss -- needs to be optimised with some analysis'''
        splits = []
        for dimension in range(len(self.data.inputs[0])):
            values = [ input[dimension] for input in self.data.inputs] # optimise double for
            for value in values:
                split = [dimension, value]
                left, right = split_data(self.data, value)
            
                total_loss = self.loss(left) + self.loss(right)
                split.append(total_loss)
                splits.append(split)

        splits = np.array(splits)
        optimal_split = splits[np.argmin(splits[:,2])]  #getting split with smallest loss

        return int(optimal_split[0]), optimal_split[1]

    def split_node(self, levels=6):
        if levels == 0:
            print("reached max depth")
            return

        if len(self.data.inputs) == 1 or len(self.data.inputs) == 0: #why is zero necessary
            return 

        dimension, value = self.best_split()
        self.dimension = dimension

        left, right = split_data(self.data, value)

        self.split_value = value

        self.left = Node(left)
        self.right = Node(right)
        self.left.split_node(levels-1)
        self.right.split_node(levels-1)

    def loss(self, data):
        if len(data.inputs) == 0:
            return 0
            
        mean_label = np.mean(data.labels)

        return 1/len(data.labels)*np.sum((data.labels - mean_label)**2)



class Tree:
    def __init__(self, root, depth=6):
        self.root = root
        self.depth = depth
    
    def train(self):
        self.root.split_node(levels=self.depth)

    def predict(self, inputs):
        '''takes a data point and runs through the splits'''
        return [ self.root.next_node(input) for input in inputs ]


if __name__ == '__main__':
    print("--------------------------START------------------------------------------")

    inputs = np.array([[x] for x in range(-50, 50)])
    labels = np.array([x**2 + 0.5*x**3 + 7*x for x in range(-50, 50)])
    sample_dataset = DataSet(inputs, labels)

    x = [ point[0] for point in inputs ]
    y = labels

    plt.plot(x,y, zorder=0)

    root_node = Node(sample_dataset)
    decision_tree = Tree(root_node, depth=5)
    decision_tree.train()
    inputs = [[pt] for pt in np.linspace(-50, 50, 1000)]

    outputs = decision_tree.predict(inputs)
    plt.plot(inputs, outputs, 'k')
    plt.show()