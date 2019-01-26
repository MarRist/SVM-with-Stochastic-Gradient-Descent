
# coding: utf-8

# In[57]:


# Martina Risteska (ID: 1003421781)

'''
Question 2 - Training SVM with SGD

'''

import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sklearn.metrics as metrics
import zipfile 



np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  



class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        params_updated = []
        for g, w in zip(grad, params):
        
            self.vel = self.beta*(self.vel) + g
            w = w - self.lr*(self.vel)
            params_updated.append(w)
        
        return np.asarray(params_updated)
    
    

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        n = X.shape[0]
        loss = np.zeros((n, ))
        z_history = np.zeros((n,))
        
        for idx, data_point in enumerate(X):
            wx = np.dot(data_point,self.w)
            z = y[idx]*(wx)
            z_history[idx] = z
            loss[idx] = max(1 - z, 0)  
        return loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        n = X.shape[0]
        m = X.shape[1]
        gradient = np.zeros((m,))
        hinge_loss_gradient = np.zeros((m,))
        
        for idx, data_point in enumerate(X):
            wx = np.dot(data_point,self.w)
            z = y[idx]*(wx)
         
            if(z < 1):
                hinge_loss_gradient = hinge_loss_gradient + (y[idx])*np.transpose(data_point).reshape(m,)
        
        no_bias_weights = np.concatenate((np.zeros(1), self.w[1:m]), axis = 0)
        gradient = no_bias_weights - (self.c)/float(n)*hinge_loss_gradient
        
        return gradient

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        n = X.shape[0]
        y_pred = np.zeros((n,))
        
        for idx, data_point in enumerate(X):
            
            if(data_point.dot(self.w) >= 0):
                y_pred[idx] = 1
            
            if(data_point.dot(self.w) < 0):
                y_pred[idx] = -1
            
        return y_pred
    
    
def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    

    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params([w], [func_grad(w)])
        w_history.append(w)
        
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    feature_count = train_data.shape[1]
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    
    # create a SVM class and initialize the weights randomly between 0.0 and 0.1
    SVM_object = SVM(penalty, feature_count)
    
    for i in range(iters):  
        
        X_batch, y_batch = batch_sampler.get_batch() # sample a random batch
    
        # Optimize and update the history
        gradient = SVM_object.grad(X_batch, y_batch)
        SVM_object.w = optimizer.update_params(SVM_object.w, gradient)
        
    return SVM_object

def visualize(param_history1, param_history2):
    
    k = range(0,len(param_history1))
    
    fig = plt.figure(figsize=(15,10))
    font = {'size' : 15}
    plt.rc('font', **font) 
    plt.title("SGD with Momentum", fontsize = 22)
    plt.plot(k, param_history1, 'r', label = "β = 0.0")
    plt.plot(k, param_history2, 'b', label = "β = 0.9")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Time Step (T)", fontsize = 20)
    plt.ylabel("Parameter Value", fontsize = 20)
    plt.show()
    
def plot_images(parameters1, parameters2):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
    
    ax1.set_title('Parameters for β = 0.0')
    ax1.imshow(parameters1.reshape(28,28), cmap='gray')
    ax1.axis("off")
    
    ax2.set_title('Parameters for β = 0.1')
    ax2.imshow(parameters2.reshape(28,28), cmap='gray')
    ax2.axis("off")
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == '__main__':
    
    # PART 2.1 Verification of the SGD with momentum implementation
    learning_rate = 1.0
    beta = [0.0, 0.9]
    parameter_history = []
    
    for b in beta:
        gradient_decent = GDOptimizer(learning_rate, b)
        parameter_history.append(optimize_test_function(gradient_decent))
    
    visualize(parameter_history[0], parameter_history[1])
    
    
    # PART 2.2/2.3 Training SVM
    # Load data 
    X_train, y_train, X_test, y_test = load_data()
    X_train = np.c_[ np.ones(X_train.shape[0]), X_train] # creating a design matrix
    X_test = np.c_[ np.ones(X_test.shape[0]), X_test] # creating a design matrix
    
    # Create a batch sampler to generate random batches from data with a batch size of m = 100
    BATCHES = 100
    learning_rate = 0.05
    c = 1.0  # penalty term
    T = 500
    beta1 = 0.0
    beta2 = 0.1
    
    # create two optimizers with different beta values
    SGD_optimizer_1 = GDOptimizer(learning_rate, beta1)
    SGD_optimizer_2 = GDOptimizer(learning_rate, beta2)
    
    # Train the SVM classifier
    opt_SVM1 = optimize_svm(X_train, y_train, c, SGD_optimizer_1, BATCHES, T)
    opt_SVM2 = optimize_svm(X_train, y_train, c, SGD_optimizer_2, BATCHES, T)
    
    # Plot the the learned weights as 28x28 image (disregard the bias term)
    plot_images(opt_SVM1.w[1:opt_SVM1.w.shape[0]], opt_SVM2.w[1:opt_SVM2.w.shape[0]])
    
    
    # Make a prediction on TEST and TRAIN data
    # TRAIN data
    prediction1_train = opt_SVM1.classify(X_train)
    prediction2_train = opt_SVM2.classify(X_train)
    # TEST data
    prediction1_test = opt_SVM1.classify(X_test)
    prediction2_test = opt_SVM2.classify(X_test)
    
    print("The train loss for beta=0.0 is %.16f" % opt_SVM1.hinge_loss(X_train, y_train).mean())
    print("The train loss for beta=0.1 is %.16f:" % opt_SVM2.hinge_loss(X_train, y_train).mean())
    print("Classification accuracy on train data for beta=0.0:", metrics.accuracy_score(y_train, prediction1_train)*100)
    print("Classification accuracy on train data for beta=0.1:", metrics.accuracy_score(y_train, prediction2_train)*100)
    
    print("-------------------------------")
    
    print("The test loss for beta=0.0 is %.16f" % opt_SVM1.hinge_loss(X_test, y_test).mean())
    print("The test loss for beta=0.1 is %.16f" % opt_SVM2.hinge_loss(X_test, y_test).mean())
    print("Classification accuracy on test data for beta=0.0:", metrics.accuracy_score(y_test, prediction1_test)*100)
    print("Classification accuracy on test data for beta=0.1:", metrics.accuracy_score(y_test, prediction2_test)*100)
    
    pass

