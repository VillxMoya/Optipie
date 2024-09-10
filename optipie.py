import warnings
from datetime import datetime
from math import exp, log, sqrt, pi
from scipy.stats import norm

try:
    import numpy as np
except ImportError:
    print("Numpy is needed for calculations")

try:
    import pandas as pd
except ImportError:
    print("Pandas is needed for calculations")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is needed for plotting")

try:
    import yfinance as yf
except ImportError:
    print("yfinance is needed for data fetching")    

def time_delta(expiration : str, format_t = "%Y-%m-%d", start=None): 

    """
        Computes time remaining until expiry.
        
        Parameters
        ----------
        expiration : Expiry date.

        format_t : Format of the given dates. Default is %Y-%m-%d.

        start = Start Date to compute. Defaults to the day the function is executed.

    """

    if start == None:
        st = datetime.strptime(datetime.now().strftime(format_t), format_t)
    else: 
        st = datetime.strptime(start, format_t)
    ex = datetime.strptime(expiration, format_t)
    dif = ex - st
    return dif.days/365

def get_data(name : str, expiry : str, type_opt : str):

    """
        Fetch data of options using Yahoo Finance. 
        
        Parameters
        ----------
        name : Ticker of the underlying

        expiry : Expiry date of the option

        type_opt : Type of option, call or put.

    """

    eq = yf.Ticker(name)
    if type_opt == "call":
        eq_opt = eq.option_chain(date = expiry).calls
    if type_opt == "put":
        eq_opt = eq.option_chain(date = expiry).puts
    return eq_opt

def Newton(f, x0):

    """
        Newton-Raphson step to compute IV. 
        
        Parameters
        ----------
        f : Function to reduce

        x0 : starting point

    """

    def df(x):
        dx = 0.00001
        return (f(x+dx)-f(x)) /dx #1st Derivative of the obective function
    return x0 - f(x0)/df(x0) #Aproximation towards the point 

class prices(): 

    def __init__(self, price:float, strike:float, volatility:float, expiry:str, risk_free:float, mkt_price = None, div_yield = 0, type_opt=None, start=None, format_t = "%Y-%m-%d" ):
        
        """
            MAIN PARAMETERS
            ---------------

            price : Spot Price of the underliying of the option.
            
            strike: Strike of the selected option.

            volatility : Expected or Implied volatility for option calculations.

            expiry : Expiry date of the option. Default format is %Y-%m-%d.

            risk_free : Risk Free Rate for Black-Scholes calculations. 

            mkt_price : Price of the option in the market. Used to compute IV and other variables. Default is None can be introduced later.

            div_yield : Dividend yield to compute option price. Default is 0.

            type_opt : Type of option - call or put. Default is call.

            start : Start date for calculation on time spent. Default is the day computing the price.

            format_t : Applies to expiry -> Format to be applied for the calculation of time spent. Default format is %Y-%m-%d.

        """        
        
        self.price = price
        self.strike = strike
        self.volatility = volatility
        self.expiry = expiry
        self.risk_free = risk_free 
        self.start = start
        self.format_t = format_t
        self.type_opt = type_opt
        self.div_yield = div_yield
        self.time = time_delta(self.expiry, start=self.start, format_t = self.format_t)

    def call_price(self, volatility = None, strike = None) -> float:

        """
            Black-Scholes Call price.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            volatility: float
                Expected or Implied Volatility used for calculating price.                   
            strike: float
                Strike price of the call.

            Returns: The Black-Scholes call price.
        """

        if volatility == None:
            volatility = self.volatility
        if strike == None:
            strike = self.strike
        d1 = ( (log(self.price / strike) ) + (self.risk_free - self.div_yield + (volatility**2) / 2) * self.time ) / (volatility * sqrt(self.time))
        d2 = d1 - volatility*sqrt(self.time)
        priceC = exp(-self.div_yield * self.time) * self.price * norm.cdf(d1) - strike * exp( - self.risk_free * self.time )*norm.cdf(d2)
        return priceC
    
    def put_price(self, volatility=None, strike = None):

        """
            Black-Scholes put price.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            volatility: float
                Expected or Implied Volatility used for calculating price.                   
            strike: float
                Strike price of the put.

            Returns: The Black-Scholes put price.
        """

        if volatility == None:
            volatility = self.volatility
        if strike == None:
            strike = self.strike
        d1 = ( (log(self.price / strike) ) + (self.risk_free - self.div_yield + (volatility**2) / 2) * self.time ) / (volatility * sqrt(self.time))
        d2 = d1 - volatility*sqrt(self.time)
        priceP = strike * exp( - self.risk_free * self.time ) * norm.cdf(-d2) - self.price * norm.cdf(-d1) * exp(-self.div_yield * self.time)
        return  priceP
    
    def set_vola(self, volatility : float):
        self.volatility = volatility
    
    def set_mkt_price(self, mkt_price : float):
        self.mkt_price = mkt_price
    
    def set_strike(self, strike : float):
        self.strike = strike

class greeks(prices):
    def __init__(self, price, strike, expiry, risk_free, volatility, mkt_price = None, div_yield = 0, type_opt=None, start=None, format_t = "%Y-%m-%d" ):
        super().__init__(price, strike, expiry, risk_free, volatility, mkt_price, div_yield, type_opt, start, format_t)
        
    def delta(self, price = None):

        """
            Black-Scholes delta.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            price: Spot price of the underliying.

            Returns: The Black-Scholes Delta.
        """

        if price == None:
            price = self.price
        d1 = ( (log(price / self.strike) ) + (self.risk_free - self.div_yield + (self.volatility**2) / 2) * self.time ) / (self.volatility * sqrt(self.time))
        if self.type_opt == "call":
            return norm.cdf(d1) * exp(-self.div_yield*self.time)
        if self.type_opt == "put":
            return norm.cdf(d1) * exp(-self.div_yield*self.time) - 1
        else: 
            print("Type must be call or put.")
            return None

    def gamma(self, price = None):

        """
            Black-Scholes Gamma.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            price: Spot price of the underliying.

            Returns: The Black-Scholes Gamma.
        """

        if price == None:
            price = self.price
        d1 = ( (log(price / self.strike) ) + (self.risk_free - self.div_yield + (self.volatility**2) / 2) * self.time ) / (self.volatility * sqrt(self.time))
        return ( norm.pdf(d1) * exp(-self.div_yield*self.time) ) / (self.price * self.volatility * sqrt(self.time))

    def vega(self, price = None, volatility = None):

        """
            Black-Scholes Vega.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            price: Spot price of the underliying.
            volatility : Expected or Implied Volatility used for calculating price of the option.

            Returns: The Black-Scholes Vega.
        """

        if price == None:
            price = self.price
        if volatility == None:
            volatility = self.volatility
        d1 = ( log(price / self.strike)  + (self.risk_free - self.div_yield + (volatility**2) / 2) * self.time ) / (volatility * sqrt(self.time)) 
        return self.price * norm.pdf(d1) * sqrt(self.time) * exp(-self.div_yield*self.time) / 100

    def theta(self, day = 365, price = None): 

        """
            Black-Scholes Theta.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            day : Number of days for the return of the theta. 
                    365 : Daily Theta
                    1 : Anual Theta

            price: Spot price of the underliying.

            Returns: The Black-Scholes Theta.
        """

        if price == None: 
            price = self.price
        d1 = ( (log(price / self.strike) ) + (self.risk_free - self.div_yield + (self.volatility**2) / 2) * self.time ) / (self.volatility * sqrt(self.time))
        a = ((price * exp(-self.div_yield * self.time) * norm.pdf(d1) * self.volatility) / (2 * sqrt(self.time)))
        if self.type_opt == "call":
            theta = - a - self.risk_free * self.strike * exp(-self.risk_free * self.time)*norm.cdf(d1-self.volatility*sqrt(self.time)) + self.div_yield * price * exp(-self.div_yield*self.time) * norm.cdf(d1)
            return theta / day
        if self.type_opt == "put":
            theta = - a + self.risk_free * self.strike * exp(-self.risk_free * self.time)*norm.cdf(-d1+self.volatility*sqrt(self.time)) - self.div_yield * price * exp(-self.div_yield*self.time) * norm.cdf(-d1)
            return theta / day
        else: 
            print("Type must be call or put.")  
            return None      

    def rho(self, day = 1, price = None): 

        """
            Black-Scholes Rho.

            Main parameters are used via de creation of the class type. 

            Optional Parameters
            -------------------

            day : Number of days for the return of the Rho. 
                    365 : Daily Rho
                    1 : Anual Rho

            price: Spot price of the underliying.

            Returns: The Black-Scholes Rho.
        """

        if price == None:
            price = self.price
        d1 = ( (log(price / self.strike) ) + (self.risk_free - self.div_yield + (self.volatility**2) / 2) * self.time ) / (self.volatility * sqrt(self.time))
        if self.type_opt == "call":
            rho = self.strike * self.time * exp(-self.risk_free * self.time) * norm.cdf(d1-self.volatility*sqrt(self.time)) / 100
            return rho / day
        if self.type_opt == "put":
            rho = -self.strike * self.time * exp(-self.risk_free * self.time) * norm.cdf(-d1+self.volatility*sqrt(self.time)) / 100
            return rho / day
        else: 
            print("Type must be call or put.")
            return None        
        
class implied_vola(greeks):
    def __init__(self, price, strike, expiry, risk_free, volatility, mkt_price = None, div_yield = 0, type_opt=None, start=None, format_t = "%Y-%m-%d" ):
        super().__init__(price, strike, expiry, risk_free, volatility, mkt_price, div_yield, type_opt, start, format_t)

    def inflection(self):

        """
            Volatility Inflection point.

            Main parameters are used via de creation of the class type. 

            Computes Inflection point to stablish as initial point for the Newton Raphson Algorithm calculations.

            Returns: Inflection Volatility point.
        """

        m = self.price / (self.strike * exp(-self.risk_free*self.time))
        return sqrt(2 * (abs(log(m)))/self.time)

    def brenner(self):

        """
            Volatility Brenner-Subrahmanyam point.

            Main parameters are used via de creation of the class type. 

            Computes Brenner-Subrahmanyam point to stablish as initial point for the Newton Raphson Algorithm calculations.

            Returns: Brenner-Subrahmanyam Volatility point.
        """

        b = (self.call_price() + self.put_price())/(self.price*2)
        return sqrt(2*pi/self.time)*b
    
    def Newton_IV(self, mkt_price : float, x0 : float, error : float) -> float:

        """
            Basic Newton-Raphson algorithm.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------

            mkt_price : Option price in the market.

            x0 : Initial point to compute Newton-Raphson.

            error : Tolerance allowed to select last volatility point.

            Returns: Implied Volatility for Black-Scholes.
        """
        
        CBS_IV = lambda vol : self.call_price(volatility = vol) - mkt_price
        while(abs(Newton(x0 = x0, f =  CBS_IV) - x0) > error):
            x0 = Newton(x0 = x0, f = CBS_IV) #Newton Raphson algorithm
        return x0

    def IV(self, mkt_price=None, error = 10**(-10), x0 = None, meth = "brenner") -> float: 

        """
            Specific Newton-Raphson algorithm.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------

            mkt_price : Option price in the market.

            x0 : Initial point to compute Newton-Raphson.

            error : Tolerance allowed to select last volatility point.

            meth : Method selected to compute the initial point for the algorithm.

            Returns: Implied Volatility for Black-Scholes.
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if mkt_price == None:
            mkt_price = self.mkt_price

        if x0 == None:    
            if meth == "brenner":
                x0 = self.brenner()
            if meth == "infl":
                x0 = self.inflection()

        p = self.call_price(volatility = x0)
        v = self.vega(volatility = x0) * 100
        while (abs((p - mkt_price)/v)) > error:
            x0 = x0 - ((p - mkt_price) / v)
            p = self.call_price(volatility = x0)
            v = self.vega(volatility = x0) * 100

        warnings.filterwarnings("default", category=RuntimeWarning)
        
        return x0
    
class plot(implied_vola):
    def __init__(self, price, strike, volatility, expiry, risk_free, mkt_price=None, div_yield = 0, type_opt=None, start=None, format_t = "%Y-%m-%d"):
        super().__init__(price, strike, expiry, risk_free, volatility, mkt_price, div_yield, type_opt, start, format_t)

    def plot_payoff(self, long = 0, cost = 0):
        
        """
            Payoff graph with buy or sell position including price of the option.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------

            long : long or short position of the option.
                    0 : long
                    1 : short
            
            cost : Price of the plotted option.

            Returns: Payoff Graph.
        """

        prices = np.arange(-self.strike, 3*self.strike, 0.01*self.strike)
        pyoff = []

        for i in prices:
            if self.type_opt == "call":
                a = (1-long)*max(0,i - self.strike) + (long)*min(0,i - self.strike) - (self.call_price()*(-1)**long)*cost #LONG CALL
                beven = self.strike + self.call_price()*(-1)**long
                x_ = [beven]*len(prices)
            elif self.type_opt == "put":
                a = (1-long)*max(0, self.strike - i) + (long)*min(0, self.strike - i) - (self.put_price()*(-1)**long)*cost #LONG PUT
                beven = self.strike - self.put_price()*(-1)**long
                x_ = [beven]*len(prices)
            pyoff.append(a)
            
        pyoff = np.array(pyoff)
        fig, ax = plt.subplots()
        y_ = [0]*len(prices)
        ax.plot(prices, pyoff)
        if cost == 1:
            ax.plot(prices, y_, color = "black")
            ax.plot(x_, pyoff)
        plt.xlabel("Strike")
        plt.ylabel('Profit and loss')
        plt.show()

    def plot_greeks(self, delta = 0, gamma = 0, vega = 0, theta = 0, rho = 0, all = 0):

        """
            Greeks graph.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------

            Any of the deltas can be activated or desactivated.
                0 : deactivated 
                1 : activated
            
            all (1) : every delta is activated 

            Returns: Greeks Graph.
        """

        prices = np.arange(0.8*self.price, 1.2*self.price, 0.01*self.price) #super in the money delta
        delta_p = []
        gamma_p = []
        vega_p = []
        theta_p = []
        rho_p = []

        if all == 1: 
            delta = gamma = vega = theta = rho = 0
            print("Returning plot of all Greeks against spot")

        for i in prices:
            a = self.delta(price = i) * (delta + all)
            b = self.gamma(price = i) * (gamma + all)
            c = self.vega(price = i) * (vega + all)
            d = self.theta(price = i) * (theta + all)
            e = self.rho(price = i) * (rho + all)
            delta_p.append(a)
            gamma_p.append(b)
            vega_p.append(c)
            theta_p.append(d)
            rho_p.append(e)

        fig, ax = plt.subplots()

        if delta == 1 or all == 1:
            ax.plot(prices, delta_p, label = "Delta")
        if gamma == 1 or all == 1:
            ax.plot(prices, gamma_p, label = "Gamma")
        if vega == 1 or all == 1:
            ax.plot(prices, vega_p, label = "Vega")
        if theta == 1 or all == 1:
            ax.plot(prices, theta_p, label = "Theta")
        if rho == 1 or all == 1:
            ax.plot(prices, rho_p, label = "Rho")

        ax.legend()
        plt.xlabel("Price")
        plt.title(f"Greeks for {self.type_opt} at {self.strike} for {self.expiry}")
        plt.show()

    def plot_IV(self, name : str, marg = 0.3):

        """
            Implied Volatility graph, calculated by the function own model against market provided.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------
            name : Ticker of the underliying to retrieve market IVs. Data is fetched via Yahoo Finance with function get_data()
            
            marg : margin to expand beyond spot price of the underliying to compute IVs.

        """

        eq_opt = get_data(name, self.expiry, self.type_opt)
        a = (1-marg)*self.strike
        b = (1+marg)*self.strike
        list = eq_opt[(eq_opt.strike > a) & (eq_opt.strike < b)]
        list = list[["strike", "lastPrice", "impliedVolatility"]]
        IV_b = []
        og_strike = self.strike
        for s, p in zip(list.strike, list.lastPrice): 
            self.set_strike(s)
            i = self.IV(mkt_price = p)
            IV_b.append(i)
        self.set_strike(og_strike)
        fig, ax = plt.subplots()
        ax.plot(list.strike, IV_b, label="self IVs")
        ax.plot(list.strike,list.impliedVolatility, label = "Market")
        plt.legend()
        plt.xlabel("Strike")
        plt.ylabel("IV")
        plt.title("Implied Volatility: Self vs Market")
        plt.show()
        
class options(plot, prices):
    
    def __init__(self, price:float, strike:float, volatility:float, expiry:str, risk_free:float, mkt_price = None, div_yield = 0, type_opt=None, start=None, format_t = "%Y-%m-%d"):
        self.price = price
        self.strike = strike
        self.volatility = volatility
        self.expiry = expiry
        self.risk_free = risk_free 
        self.start = start
        self.format_t = format_t
        self.type_opt = type_opt
        self.div_yield = div_yield
        self.mkt_price = mkt_price
        self.time = time_delta(self.expiry, start=self.start, format_t = self.format_t)

    def CallPrice(self, IV = 0, mkt_price = None) -> float:

        """
            Call price, calculated by default with IV.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------
            IV : Implied Volatility to compute.
                0 : IV computed by the function's model.
                None : Class informed volatility
                Any : Volatility specified 
            
            mkt_price : Price for the option in the market to compute IV.

        """

        if IV == 0:
            if mkt_price != None:
                vola = self.IV(mkt_price = mkt_price)
            else:
                vola = self.IV(mkt_price = self.mkt_price)
        elif IV == None: 
            vola = self.volatility
        else:
            vola = IV
        return self.call_price(volatility = vola)
    
    def PutPrice(self, IV = True, mkt_price : float = None) -> float:

        """
            Call price, calculated by default with IV.

            Main parameters are used via de creation of the class type. 
            
            Optional Parameters
            -------------------
            IV : Implied Volatility to compute.
                0 : IV computed by the function's model.
                None : Class informed volatility
                Any : Volatility specified 
            
            mkt_price : Price for the option in the market to compute IV.

        """

        if IV == 0:
            if mkt_price != None:
                vola = self.IV(mkt_price = mkt_price)
            else:
                vola = self.IV(mkt_price = self.mkt_price)
        elif IV == None: 
            vola = self.volatility
        else:
            vola = IV
        return self.put_price(volatility = vola)
    
    def detail_opt(self): 

        """
            Table with details for the specific option.

        """

        if self.type_opt == "call":
            self.det = pd.DataFrame(data = [[self.type_opt, self.expiry, self.strike, self.call_price()]], 
                                    columns = ["Type Option", "Expiry", "Strike", "Price"])
            return self.det
        if self.type_opt == "put": 
            self.det = pd.DataFrame(data = [[self.type_opt, self.expiry, self.strike, self.put_price()]], 
                                    columns = ["Type Option", "Expiry", "Strike", "Price"])
            return self.det
        else: 
            return print("Type of Option needed")
        
    def greeks_res(self):

        """
            Table with greeks for the specific option.

        """

        if self.type_opt == "call" or self.type_opt == "put":
            self.greeks = pd.DataFrame(data = [[f"{self.type_opt} {self.expiry}", self.delta(), self.gamma(), self.vega(), self.theta(), self.rho()]], 
                                       columns = ["Option","Delta", "Gamma", "Vega", "Theta", "Rho"])
            return self.greeks
        else: 
            return print("Type of Option needed")

    def describe(self): 

        """
            Two brief tables with the main details of the option.

        """

        return self.detail_opt(), self.greeks_res()
    

    