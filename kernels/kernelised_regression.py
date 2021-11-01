import numpy as np
import matplotlib.pyplot as plt

SIGMA_2 = 9
x = np.linspace(-5, 5, 11)
y = x**2
data = [ [xpt, ypt] for xpt, ypt in zip(x,y) ]
plt.scatter(x,y)

def exponential_cov(x_1, x_2):
    return np.exp(-0.5*(1/SIGMA_2)*np.subtract.outer(x_1, x_2)**2)

class KernelisedRegressor:
    def __init__(self, data):
        self.data = data
        self.x, self.y = np.array([ pt[0] for pt in data]), np.array([ pt[1] for pt in data])
        self.kernel = exponential_cov
        self.kernel_matrix = self.kernelise()
        print(self.kernel_matrix.shape)
        print(self.kernel_matrix)
        self.alphas = []

    def kernelise(self):
        return self.kernel(self.x,self.x)

    def train(self, s=0.01):
        self.alphas = np.zeros(len(self.data))
        print('alpja shape', self.alphas.shape)
        print(self.alphas)
        #gammas = np.array(-2*self.y)
        for _ in range(2000):
            #alpha_ks = [self.alphas[j]*self.kernel_matrix[:,j] for j in range(len(self.alphas))]
            #k_section = np.sum(self.kernel_matrix, axis=1)
            #print(k_section)
            #gammas = np.sum(self.alphas*k_section) - self.y
            gam = np.array([ np.sum( [self.alphas[j]*self.kernel_matrix[i, j] for j in range(len(self.data))], axis=0) - self.y[i] for i in range(len(self.y))])
            print('gam', gam)
            #gammas = np.sum(alpha_ks) - self.y
            self.alphas = self.alphas - 2*s*(gam)
            # weights = np.sum(self.alphas*self.x, axis=0) #CHECK
            # gammas = 2*np.dot(weights, self.x) - self.y

    def predict(self, input):
        return np.sum(self.alphas*self.kernel(self.x, input))

kernel_model = KernelisedRegressor(data)
kernel_model.train()
x_pred = np.linspace(-5, 5, 200)
predictions = np.array([ kernel_model.predict(x) for x in x_pred ])
plt.plot(x_pred, predictions)
plt.show()