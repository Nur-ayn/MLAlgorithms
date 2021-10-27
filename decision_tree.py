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


dataset = [
           [[1], 1],
           [[2], 2],
           [[3], 3],
           [[4], 4],
           [[0], 0],
           [[-1], -1],
           [[-2], -2],
           [[-3], -3],
           [[-4], -4]
           ]

dataset = [ [[x], x] for x in range(-15, 15) ]

datapoints = [ DataPoint(point[0], point[1]) for point in dataset ]
datapoints = np.array(datapoints)


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
            return f"the mean is {self.mean}"
        print('split value', self.split_value)
        print('left and right', self.left, self.right)
        if test_point[self.dimension] < self.split_value:
            return self.left.next_node(test_point)
        if test_point[self.dimension] >= self.split_value:
            return self.right.next_node(test_point)

    def best_split(self):
        '''loops through each dimension of the inputs and each split of the data and returns the one with
            minimal square loss -- needs to be optimised with some analysis'''
        splits = []
        for dimension in range(len(self.data[0].input)):
            #print('DATADATADATa', self.data)
            values = [ datapoint.input[dimension] for datapoint in self.data]
            #print("ALL THE ValUEs", values)
            for value in values:
                split = [dimension, value]
                left, right = split_data(self.data, np.array([ datapoint.input[dimension] for datapoint in self.data ]) < value)
                # print('left points', left)
                # print('right_points', right)
                total_loss = self.loss(left, split) + self.loss(right, split)
                split.append(total_loss)
                splits.append(split)

        splits = np.array(splits)
        #print('considered splits', splits)
        
        #print('mean', self.mean)
        optimal_split = splits[np.argmin(splits[:,2])]  #getting split with smallest loss
        # print('optimal split', optimal_split)

        return optimal_split[0], optimal_split[1]

    def split_node(self, levels=6):
        if levels == 0:
            self.plot_node()
            print("done training")
            return
        dimension, value = self.best_split()
        print('best split', dimension, value)
        self.dimension = int(dimension)

        left, right = split_data(self.data, np.array([ datapoint.input[self.dimension] for datapoint in self.data ]) < value)

        if len(left) == 0 or len(right) == 0:
            print('data no split', self.data)
            return

        self.split_value = value

        if len(left) != 0:
            self.left = Node(left)
            self.left.split_node(levels-1)
            
        if len(right) != 0:
            self.right = Node(right)
            self.right.split_node(levels-1)

    @staticmethod
    def loss(datapoints, split):
        if len(datapoints) == 0:
            return 0
            
        mean_label = np.mean( [datapoint.label for datapoint in datapoints])
        print([ (datapoint.label - mean_label)**2 for datapoint in datapoints ])
        return 1/len(datapoints)*np.sum([ (datapoint.label - mean_label)**2 for datapoint in datapoints ])

    def plot_node(self):
        datapoints = self.data
        x = np.array([ datapoint.input[0] for datapoint in datapoints ])
        y = [self.mean for _ in range(len(x)) ]
        print("TRYING TO PLOT ", x, y)
        plt.scatter(x, y, zorder=5)
        plt.draw()




class Tree:
    def __init__(self, root, depth=6):
        self.root = root
        self.depth = depth
    
    def train(self):
        self.root.split_node(levels=self.depth)

    def predict(self, inputs):
        '''takes a data point and runs through the splits'''
        return [ self.root.next_node(input) for input in inputs ]


print("--------------------------START------------------------------------------")

root_node = Node(datapoints)
decision_tree = Tree(root_node, depth=10)
decision_tree.train()
inputs = [[0],
          [-10],
          [3.5],
          [1.2],
          [10.5]
         ]
print(decision_tree.predict(inputs))


plt.show()