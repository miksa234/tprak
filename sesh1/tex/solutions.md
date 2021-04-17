# Findings

## Code
The code is structured as follows:

    *   Construct statistical covariance matrix
    *   Construct Jacobi matrix of model function in terms of the parameters
    *   Guess parameters p0

Then we iterate the following:

    *   Construct System covariance matrix with the guess p0
    *   Fill Jacobi matrix with p0 = Design Matrix
    *   Claculate dp
    *   p0 = p0 + alpha*dp; alpha \in [0, 1]
    *   (calculate the chi^2 function)

Additionally we can calculate the chi^2 function at each iteration.

The guess used for all fits was determined by standard least-square fit
provided by 'scipy'. p0 = [0.9, 0.2, 0.81, 0.04, 0.02, -1, 0.84, 1.55]   # in GeV
First let us look at results of single experiments fitted separately:

    SND -------> Pictures
    CMD2
    KLOE
    BABAR

    Table with fits and chisq, pvalue

Then we can fit multiple experiments togehter if we align the given data and
construct block diagonal system covariance matricies. 4 experiments fited with
2 together gives 6 combinations.

    SND-CMD2
    SND-KLOE
    SND-BABAR
    CMD2-KLOE
    CMD2-BABAR
    KLOE-BABAR
    6 Picturs

    Table with fits and chisq, pvalue


Here we fitted all experiments together.

    ALL EXPERIMENTS
    1 Picture

    Table with fits and chisq, pvalue


Discussion schau ma mal was rauskommt

Reference code   git://popovic.xyz/tprak.git
