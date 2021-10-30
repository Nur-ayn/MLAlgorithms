import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_blobs
from itertools import cycle

# DATA
k = 3
centers = [(-1, -1), (3, 0), (2, 2)]
cluster_std = [1.9, 1.5, 2.1]

X, y = make_blobs(n_samples=300, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

x_1 = np.concatenate((X[y == 0, 0], X[y == 1, 0], X[y == 2, 0]))
x_2 = np.concatenate((X[y == 0, 1], X[y == 1, 1], X[y == 2, 1]))

datapoints = np.array([ (x1, x2) for x1, x2 in zip(x_1, x_2)])

# PLOTS
colours = cycle('bgrcmy')

fig = plt.figure()
ax = plt.axes()

lines = [ax.plot(x_1,x_2, 'o', lw=2,color='k')[0]] + [ ax.plot([],[], 'o', lw=2,color=next(colours))[0] for _ in range(k)]


# FUNCS
def initialise_centroids(data, k):
    '''return k different randomly-chosen points from the dataset'''
    random_indices = np.arange(0, data.shape[0])   
    np.random.shuffle(random_indices)       
    return data[random_indices[:k]]  

def compute_centroid(cluster):
    return np.mean(cluster, axis=0)

def compute_distance(point, centroid):
    point, centroid = np.array(point), np.array(centroid)
    return np.sqrt(np.sum((point - centroid)**2))

def initial_cluster(data, k): 
    centroids = initialise_centroids(data, k)
    clusters = [ [] for _ in range(len(centroids)) ]

    for point in data:
        distances = np.array([ compute_distance(point, centroid) for centroid in centroids ])
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
    
    return clusters, centroids


def re_cluster(clusters, data):
    new_centroids = [ compute_centroid(cluster) for cluster in clusters]
    new_clusters = [ [] for _ in range(len(clusters)) ]

    for point in data:
        distances = np.array([ compute_distance(point, centroid) for centroid in new_centroids ])
        cluster_index = np.argmin(distances)
        new_clusters[cluster_index].append(point)

    return new_clusters, new_centroids

clusters, centroids = None, None

# ANIMATION

def init():
    lines[0].set_data(x_1, x_2)
    return lines

def update(i):
    print(i)
    global clusters, centroids
    if i == 0:
        clusters, centroids = initial_cluster(datapoints, k)
        print('centroids', centroids)

    for cluster, line, centroid in zip(clusters, lines[1:], centroids):
        x1_data = [ pt[0] for pt in cluster ]
        x2_data = [ pt[1] for pt in cluster ]
        line.set_data(x1_data, x2_data)
        print('centroid ', centroid)
        ax.plot(centroid[0], centroid[1], 'P', mec='black', color=line.get_color(), markersize=10)

    

    clusters, centroids = re_cluster(clusters, datapoints)

    print('centroids', centroids)

    return lines

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=10, interval=500, blit=False, repeat=False)


plt.show()