# Iterative solution to the solution of the D'Agostini Bias (aka. t0-Method)

    *   Construct statistical covariance matrix
    *   Construct design matrix
    *   Guess parameters p0

## Iterate

    *   Construct System covariance matrix with the guess p0
    *   Fill design matrix with p0
    *   Claculate dp
    *   p0 = p0 + dp

