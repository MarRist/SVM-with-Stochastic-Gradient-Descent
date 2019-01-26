# SVM with Stochastic Gradient Descent
This repository contains projects that were written for Machine Learning course at University of Toronto

1. SGD with Momentum
The stochastic gradient decent with momentum β and learning rate α is given as follows:
![eq 1](https://latex.codecogs.com/gif.latex?v_%7Bt&plus;1%7D%20%3D%20%5Cbeta%20v_t%20&plus;%20%5Cnabla%20L%28w_t%29)
![eq 2](https://latex.codecogs.com/gif.latex?x_%7Bt&plus;1%7D%20%3D%20x_t%20-%20%5Calpha%20v_%7Bt&plus;1%7D)

It can be noticed that the update rule now has the accumulated gradient (or known as the velocity) which is the
main difference between the gradient decent. However, if β = 0 the gradient decent with momentum becomes a
steepest gradient decent.
