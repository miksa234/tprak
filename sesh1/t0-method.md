# Iterative solution to the solution of the D'Agostini Bias (aka. t0-Method)

    *   Construct statistical covariance matrix
    *   Guess parameters y0 = f(x; {p0})
    *   Construct System covariance matrix with the guess y0

## Iterate

    *   Find minimum of the chi^2 function
    *   Take minimum \hat{p} and construct the system covariance matrix

Note system covariance matrix is different for one experiment than with two
independent experiments
