# SVM with Stochastic Gradient Descent
This repository contains projects that were written for Machine Learning course at University of Toronto.

This is an implementation of Stochastic Gradient Descent with momentum β and learning rate α. The implemented algorithm is then used to approximately optimize the SVM objective.

The algoritham is tested on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). MNIST is a digit classification dataset consisting of 28 × 28 grayscale images of hand-drawn digits labelled from 0 to 9. In particular, this implementation solves a binary classification problem by trying to classify 4 vs. 9 (the hardest 1-vs-1 pair) and discards the rest of the classes. 

The dataset is split into 80% train and 20% test, and the images are converted to vectors by flattening them to a vector of size 784.

For training and evaluating the SVM classifier, run `SVM_with_SGD`.

### Descrption of code implementation:

* In this inplementation, two SVM models are being trained using gradient descent with a learning rate of α = 0.05, a penalty of C = 1.0, minibatch sizes of m = 100, and T = 500 total iterations. For the first model use β = 0 and for the second use β = 0.1.

* The stochastic gradient decent with momentum β and learning rate α is given as follows:

![eq0](https://latex.codecogs.com/gif.latex?v_%7Bt&plus;1%7D%20%3D%20%5Cbeta%20v_t%20&plus;%20%5Cnabla%20L%28w_t%29)

![eq1](https://latex.codecogs.com/gif.latex?%24%24x_%7Bt&plus;1%7D%20%3D%20x_t%20-%20%5Calpha%20v_%7Bt&plus;1%7D%24%24)

