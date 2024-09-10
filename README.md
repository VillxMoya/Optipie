# Option Pricing Tool

The first version of the tool currently supports European Options, computing prices using the Black-Scholes equation alongside Greeks and computation of Implied Volatility. It also offers plotting of all the previously mentioned calculations, including:

* Payoff graphs for short and long positions if desired for the options.

* Greeks plotted against Spot price.

* Implied Volatility against Strike.

Implied volatility is computed using a Numerical approach with the Newton-Raphson method, and providing an initial point calculated using either the Inflection point of the call price or the Brenner-Subrahmanyam formula. Both are available options to choose from, in the computations of the Implied Volatility.

In the Python notebook, you will find examples of use for some of the functions developed by the tool as comparisons against market values using Yahoo Finance option data. 

Implied Volatility calculations come from: 

"A review on implied volatility calculation" Giuseppe Orlando and Giovanni Tagliatela, retrieved from Journal of Computational and Applied Mathematics Volume 320, August 17:
https://www.sciencedirect.com/science/article/pii/S0377042717300602?fr=RR-2&rr=8c1056fddccfcfdc

This is an ongoing project that will have future revisions. 
