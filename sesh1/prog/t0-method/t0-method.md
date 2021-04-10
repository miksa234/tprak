# Iterative solution to the solution of the D'Agostini Bias (aka. t0-Method)

    *   Construct statistical covariance matrix
    *   Construct Jacobi matrix of model function in terms of the parameters
    *   Guess parameters p0

## Iterate

    *   Construct System covariance matrix with the guess p0
    *   Fill Jacobi matrix with p0 = Design Matrix
    *   Claculate dp
    *   p0 = p0 + alpha*dp; alpha \in [0, 1]

