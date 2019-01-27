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

* The SVM objective is given as:

![eq2](https://latex.codecogs.com/gif.latex?%5Ctextbf%7Bw%7D%5E%7B*%7D%2C%20b%5E%7B*%7D%20%3D%20%5Carg%5Cmin%20%5Cfrac%7B1%7D%7B2%7D%7C%7C%5Ctextbf%7Bw%7D%7C%7C%5E2%20&plus;%20%5Cfrac%7BC%7D%7BN%7D%5Csum_%7Bi%20%3D1%7D%5EN%20%5Cmax%20%5CBig%28%201%20-%20y%5E%7B%28i%29%7D%28%5Ctextbf%7Bw%7D%5E%7BT%7D%20%5Ctextbf%7Bx%7D%5E%7B%28i%29%7D%20&plus;%20b%29%2C%200%20%5CBig%29)

where the w are the SVM model parameters, b is the bias term, C is the penalty parameter for misclassifying the classes, and N is the batch size. The first term in the objective is the regularization term where the second one is known as the hinge loss.

* The gradient of the hinge loss was calculated using sub-gradients:

![eq3](https://latex.codecogs.com/gif.latex?%5Cnabla_w%20Hinge%20Loss%20%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20-y%5E%7B%28i%29%7D%20%5Ctextbf%7Bx%7D%5E%7B%28i%29%7D%2C%20%26%20y%5E%7B%28i%29%7D%28%5Ctextbf%7Bw%7D%5ET%20%5Ctextbf%7Bx%7D%5E%7B%28i%29%7D%20&plus;%20b%29%20%3C%201%5C%5C%200%2C%20%26%20y%5E%7B%28i%29%7D%28%5Ctextbf%7Bw%7D%5ET%20%5Ctextbf%7Bx%7D%5E%7B%28i%29%7D%20&plus;%20b%29%20%5Cgeq%201%20%5Cend%7Bmatrix%7D%5Cright.)

Note that, the gradient at the "kink" of the hinge loss was taken to be zero. The overall gradient of the soft-SVM classifier was calculated as follows: 

![eq4](https://latex.codecogs.com/gif.latex?%5Cnabla_w%20L%20%3D%20%5Ctextbf%7Bw%7D%20-%20%5Cfrac%7BC%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y%5E%7B%28i%29%7D%20%5Ctextbf%7Bx%7D%5E%7B%28i%29%7D)
