# SVM with Stochastic Gradient Descent
This repository contains projects that were written for Machine Learning course at University of Toronto

1. SGD with Momentum
The stochastic gradient decent with momentum β and learning rate α is given as follows:
vt+1 = βvt + ∇L(wt)
xt+1 = xt − αvt+1
We can notice that the update rule now has the accumulated gradient (or known as the velocity) which is the
main difference between the gradient decent. However, if β = 0 the gradient decent with momentum becomes a
steepest gradient decent.
The implementation of the SGD with momentum was verified by finding the min of f(w) = 0.01w
2 with w0 =
10.0, α = 1.0 for both β = 0.0 and β = 0.9 performed in T = 200 iterations
