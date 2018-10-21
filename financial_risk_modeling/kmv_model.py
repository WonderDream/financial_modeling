# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:41:21 2018

The Merton's (1974) model treats a firm's stock as a call option of its asset, in which the strike price is the debt.
Moody's KMV model is the most well known implemention of the Merton's (1974) model.
This code try to estimate the unobserved asset value and the unkown parameters following the Duan et al. (2005), 
in which the parameters is estimated using a transformed-data MLE method in an expectation and maximization framework.


References:
Duan, J. C., Gauthier, G., & Simonato, J. G. (2005). On the equivalence of the KMV and maximum likelihood methods for structural credit risk models. Groupe d'études et de recherche en analyse des décisions.
Vassalou, M., & Xing, Y. (2004). Default risk in equity returns. The journal of finance, 59(2), 831-868.
Crosbie, P., & Bohn, J. (2003). Modeling default risk.
Dwyer, D., Kocagil, A., & Stein, R. (2004). The Moody’s KMV RiskCalc v3. 1 Model: Next-generation technology for predicting private firm credit risk. Moody’s KMV.                                                                 

@author: Jisong Zhu
@Email: zhujs.xy@gmail.com
"""
from __future__ import division
from scipy.stats import norm
from scipy import optimize
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.getLogger().setLevel(level = logging.INFO)

##BS Formula
def equity_pricing_by_merton_formula(asset_value, strike_price, risk_free_rate, volatility, time_horizon):
    
    d1 = (math.log(asset_value/strike_price) + (risk_free_rate + 0.5*volatility**2)*time_horizon)/(volatility*math.sqrt(time_horizon))
    d2 = d1 - volatility*math.sqrt(time_horizon)
    
    equity_value = asset_value*norm.cdf(d1) - strike_price*math.exp(-risk_free_rate*time_horizon)*norm.cdf(d2)
    return equity_value

##Estimation of the asset value from the equity value using the Newton-Raphson method
def asset_value_estimating_by_equity_pricing_equation(equity_value, debt_value, risk_free_rate, volatility, time_horizon):
    equity_value = np.asarray(equity_value)
    debt_value = np.asarray(debt_value)
    risk_free_rate = np.asarray(risk_free_rate)
    scalar_input_flag = False
    if equity_value.ndim == 0:
        scalar_input_flag = True
        equity_value = equity_value[None]
        debt_value = debt_value[None]
        risk_free_rate = risk_free_rate[None]
    else:
        if len(debt_value) == 1:
            debt_value = list(debt_value) * len(equity_value)
        if len(risk_free_rate) == 1:
            risk_free_rate = list(risk_free_rate) * len(equity_value)
    estimated_asset_value = []
    for k in range(len(equity_value)):
        equity_pricing_equation = lambda asset_value : equity_pricing_by_merton_formula(asset_value, debt_value[k], risk_free_rate[k], volatility, time_horizon) - equity_value[k]
        asset_value = optimize.newton(equity_pricing_equation, debt_value[k] + debt_value[k])
        estimated_asset_value.append(asset_value)
    if scalar_input_flag:
        return estimated_asset_value[0]
    else:
        return estimated_asset_value
 

def log_likelihood_func_for_estimated_asset_value(expected_return, volatility, estimated_asset_value, debt_value_series, risk_free_rate_series, time_horizon, interval):
    
    logging.debug('mu:%.2f, vo:%.2f' % (expected_return, volatility))
    n = len(estimated_asset_value) - 1
    value_df = pd.DataFrame({'asset_value':estimated_asset_value, 'debt_value':debt_value_series, 'risk_free_rate':risk_free_rate_series})
    value_df['X'] = value_df['debt_value']*np.exp(-value_df['risk_free_rate']*time_horizon)
                                                                 
    value_df['previous_asset_value'] = value_df['asset_value'].shift(1)
    value_df['Return'] = np.log(value_df['asset_value']) - np.log(value_df['previous_asset_value'])
    value_df.dropna(inplace = True)
    log_likelihood_V = -n/2*math.log(2*math.pi*(volatility**2)*interval) - 1/2*sum((value_df['Return'] - (expected_return - 1/2*volatility**2)*interval)**2/(interval*volatility**2)) - sum(np.log(value_df['asset_value']))
    
   
    return log_likelihood_V
      

def kmv_parameters_estimating_by_transformed_mle(equity_value_series, debt_value_series, risk_free_rate_series, expected_return_guess, volatility_guess, time_horizon = 1, interval = 1/250, tol = 1e-6):
    
    expected_return_est, volatility_est = expected_return_guess, volatility_guess
    opt_counter = 0
    neg_log_likelihood = np.inf
    while True:
        opt_counter += 1
        
        if opt_counter > 0:
            previous_neg_log_likelihood = neg_log_likelihood
        #Estep: estimating the asset value given the parameters volatility, barrier
        estimated_asset_value = asset_value_estimating_by_equity_pricing_equation(equity_value_series, debt_value_series, risk_free_rate_series, volatility_est, time_horizon)
        
        
        #Mstep: estimating the parameters by mle given the estimated asset value in Estep
        neg_log_likelihood_func = lambda params : - log_likelihood_func_for_estimated_asset_value(params[0], params[1], estimated_asset_value, debt_value_series, risk_free_rate_series, time_horizon, interval)
        neg_log_likelihood = neg_log_likelihood_func((expected_return_est, volatility_est))
        
        logging.info('the neg_log_likelihood is %.2f' % neg_log_likelihood)
        bounds = ((-.2, .5), (0.01, .5))
        opt_result = optimize.minimize(neg_log_likelihood_func, (expected_return_est, volatility_est), method = 'L-BFGS-B', bounds = bounds, options = {'disp':True})
        logging.info('last mu: %.2f, ml_opt mu: %.2f; last vo:%.2f, ml_opt vo:%.2f' % (expected_return_est, opt_result.x[0], volatility_est, opt_result.x[1]) )

        expected_return_est, volatility_est = opt_result.x
        
        estimated_asset_value = asset_value_estimating_by_equity_pricing_equation(equity_value_series, debt_value_series, risk_free_rate_series, volatility_est, time_horizon)
        estimated_return = np.log(estimated_asset_value[1:]) - np.log(estimated_asset_value[:-1])
        volatility_est = np.std(estimated_return)/math.sqrt(interval)
        expected_return_est = np.mean(estimated_return)/interval + 0.5*volatility_est**2 
        
        if (abs(neg_log_likelihood - previous_neg_log_likelihood) <= tol and opt_counter >= 20) or opt_counter > 100:
            logging.info('Optimization finished!!\n')
            break
        
       
    return(estimated_asset_value, expected_return_est, volatility_est)
        


##simulation & test
def asset_value_simulation(expected_return, volatility, V_0 = 1, length = 251, interval = 1/250):
    epsilon = np.random.normal(size = length)
    epsilon[0] = 0
    log_V_t = math.log(V_0) + (expected_return - volatility**2/2)*np.array(range(length))*interval + volatility*math.sqrt(interval)*np.cumsum(epsilon)
    asset_value_series = np.exp(log_V_t)
    return asset_value_series

def usage_demo():
    expected_return = 0.12
    volatility = 0.30
    asset_value_series = asset_value_simulation(expected_return, volatility)
    
    actual_expected_return = np.mean(np.log(asset_value_series[1:]) - np.log(asset_value_series[:-1]))*250
    actual_volatily = np.std(np.log(asset_value_series[1:]) - np.log(asset_value_series[:-1]))*math.sqrt(250)
    logging.info('actual_expected_return:%.2f, actual_volatily: %.2f\n' % (actual_expected_return, actual_volatily))
    
    
    
    debt_value_series = [.5]*251
    risk_free_rate_series = [0.05]*251
    volatility_guess = 0.40
    expected_return_guess = 0.2 
    time_horizon = 1
       
    equity_value_series = [equity_pricing_by_merton_formula(asset_value_series[k], debt_value_series[k], risk_free_rate_series[k], volatility, time_horizon) for k in range(len(asset_value_series))]
    estimated_asset_value, expected_return_est, volatility_est, barrier_est = kmv_parameters_estimating_by_transformed_mle(equity_value_series, debt_value_series, risk_free_rate_series, expected_return_guess, volatility_guess, time_horizon)                                                                           
    
    distance_to_default = (np.log(np.array(estimated_asset_value)/np.array(debt_value_series)) + (expected_return_est - 0.5*volatility_est**2)*time_horizon)/(volatility_est*math.sqrt(time_horizon))
    plt.plot(distance_to_default)
    default_likelihood_indicators = norm.cdf(-distance_to_default)
    plt.plot(default_likelihood_indicators)
