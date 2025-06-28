# HEP Statistical Analysis Suite

A ROOT/C++ implementation demonstrating advanced statistical methods used in high energy physics.

## Statistical Methods Implemented

- **Maximum Likelihood Fitting** - Custom likelihood functions with TMinuit
- **Signal + Background Modeling** - Gaussian signal over exponential background
- **Profile Likelihood** - Parameter uncertainty estimation
- **Confidence Intervals** - Frequentist interval calculation
- **Hypothesis Testing** - Likelihood ratio tests and significance calculation
- **Toy Monte Carlo Studies** - Bias testing and coverage validation
- **Systematic Uncertainties** - Nuisance parameter treatment
- **Bayesian Analysis** - Prior/posterior calculations
- **Multivariate Analysis** - Fisher discriminants and ROC curves
- **Goodness-of-Fit Tests** - Chi-square, KS, Anderson-Darling tests

## Usage

```bash
# Compile
g++ -std=c++17 -o hep_stats hep_stats.cpp $(root-config --cflags --libs) -lMinuit

# Run
./hep_stats
```

## Output

- Numerical results for all statistical tests
- ROOT plots demonstrating fits and distributions
- Comprehensive demonstration of HEP statistical analysis workflow

## Requirements

- ROOT 6.20+
- C++17 compiler
- Basic ROOT libraries (Core, Hist, Graf, Minuit)
