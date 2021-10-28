import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

#plt.ion()

def split_data(arr, cond):
      return [arr[cond], arr[~cond]]


@dataclass
class DataPoint:
    input: list
    label: float
    weight: float = None 


dataset = [ [[x], 0.2*x**2 + x  + 0.2*x**3] for x in range(-15, 15) ]

datapoints = np.array([ DataPoint(point[0], point[1]) for point in dataset ])

x = [ point.input[0] for point in datapoints ]
y = [ point.label for point in datapoints ]

plt.plot(x,y, zorder=0)


class Node:
    def __init__(self, data):
        self.dimension = None
        self.split_value = None
        self.data = data
        self.mean = np.mean([ point.label for point in data]) #adjust
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
        for dimension in range(len(self.data[0].input)):
            values = [ datapoint.input[dimension] for datapoint in self.data] # optimise double for
            for value in values:
                split = [dimension, value]
                left, right = split_data(self.data, np.array([ datapoint.input[dimension] for datapoint in self.data ]) < value)
            
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

        if len(self.data) == 1:
            return 

        dimension, value = self.best_split()
        self.dimension = dimension

        left, right = split_data(self.data, np.array([ datapoint.input[self.dimension] for datapoint in self.data ]) < value)

        # if len(left) == 0 or len(right) == 0:
        #     return

        self.split_value = value

        self.left = Node(left)
        self.right = Node(right)
        self.left.split_node(levels-1)
        self.right.split_node(levels-1)

    @staticmethod
    def loss(datapoints):
        if len(datapoints) == 0:
            return 0
            
        mean_label = np.mean( [datapoint.label for datapoint in datapoints])

        return 1/len(datapoints)*np.sum([ (datapoint.label - mean_label)**2 for datapoint in datapoints ])


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

    root_node = Node(datapoints)
    decision_tree = Tree(root_node, depth=5)
    decision_tree.train()
    inputs = [[pt] for pt in np.linspace(-15, 15, 200)]

    outputs = decision_tree.predict(inputs)
    plt.plot(inputs, outputs, 'k')
    plt.show()