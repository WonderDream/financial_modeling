# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:41:21 2018

In Brockman and Turtle (2003), the firm’s equity is viewed as a European down-and-out call option.
This code implement the KMV model using equity price formula of the Brockman and Turtle (2003) model.

References:
Duan, J. C., Gauthier, G., & Simonato, J. G. (2005). On the equivalence of the KMV and maximum likelihood methods for structural credit risk models. Groupe d'études et de recherche en analyse des décisions.
                                                                                        

@author: Jisong Zhu
@Email: zhujs.xy@gmail.com
"""
from __future__ import division
from scipy.stats import norm
from scipy import optimize
import math
import numpy as np
import pandas as pd
import logging


logging.getLogger().setLevel(level = logging.INFO)

def equity_pricing_by_bt_formula(asset_value, strike_price, risk_free_rate, volatility, barrier, time_horizon):
    if strike_price >= barrier:
        actual_strike_price = strike_price
    else:
        actual_strike_price = barrier
    a = (math.log(asset_value/actual_strike_price) + (risk_free_rate + 0.5*(volatility**2))*time_horizon)/(volatility*math.sqrt(time_horizon))
    b = (math.log(barrier**2/(asset_value*actual_strike_price)) + (risk_free_rate + 0.5*(volatility**2))*time_horizon)/(volatility*math.sqrt(time_horizon))
    
    eta = risk_free_rate/(volatility**2) + 0.5
    equity_value = asset_value*norm.cdf(a) - \
                                       strike_price*math.exp(-risk_free_rate*time_horizon)*norm.cdf(a - volatility*math.sqrt(time_horizon)) - \
                                       asset_value*((barrier/asset_value)**(2*eta))*norm.cdf(b) + \
                                                   strike_price*math.exp(-risk_free_rate*time_horizon)*((barrier/asset_value)**(2*eta - 2))*norm.cdf(b - volatility*math.sqrt(time_horizon))
    return equity_value

#equity_pricing_equation = lambda asset_value : equity_pricing_by_bt_formula(asset_value, F, vo, K, t, r) - S

def asset_value_estimating_by_equity_pricing_equation(equity_value, debt_value, risk_free_rate, volatility, barrier, time_horizon):
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
        equity_pricing_equation = lambda asset_value : equity_pricing_by_bt_formula(asset_value, debt_value[k], risk_free_rate[k], volatility, barrier, time_horizon) - equity_value[k]
        asset_value = optimize.newton(equity_pricing_equation, debt_value[k] + debt_value[k])
        estimated_asset_value.append(asset_value)
    if scalar_input_flag:
        return estimated_asset_value[0]
    else:
        return estimated_asset_value
def log_likelihood_func_for_estimated_asset_value(expected_return, volatility, barrier, estimated_asset_value, debt_value_series, risk_free_rate_series, time_horizon, interval):
    
    logging.debug('mu:%.2f, vo:%.2f, K:%.2f\n' % (expected_return, volatility, barrier))
    n = len(estimated_asset_value) - 1
    V_0 = estimated_asset_value[0]
    s = volatility*math.sqrt(time_horizon)
    value_df = pd.DataFrame({'asset_value':estimated_asset_value, 'debt_value':debt_value_series, 'risk_free_rate':risk_free_rate_series})
    value_df['X'] = value_df['debt_value']*np.exp(-value_df['risk_free_rate']*time_horizon)
    value_df['actual_strike_price'] = value_df['debt_value'].apply(lambda strike_price: strike_price if strike_price >= barrier else barrier)
    value_df['a'] = (np.log(value_df.asset_value/value_df.actual_strike_price) + (value_df.risk_free_rate + 0.5*(volatility**2))*time_horizon)/(volatility*math.sqrt(time_horizon))
    value_df['b'] = (np.log(barrier**2/(value_df.asset_value*value_df.actual_strike_price)) + (value_df.risk_free_rate + 0.5*(volatility**2))*time_horizon)/(volatility*math.sqrt(time_horizon))
    value_df['eta'] = value_df['risk_free_rate']/(volatility**2) + 0.5
    value_df['dee_equity_to_asset'] = value_df['asset_value']*(1/math.sqrt(2*math.pi))*np.exp(-0.5*value_df['a']**2)*(1/(s*value_df['asset_value'])) \
            + norm.cdf(value_df['a']) \
                     - value_df['X']*(1/math.sqrt(2*math.pi))*np.exp(-0.5*(value_df['a'] - s)**2)*(1/(s*value_df['asset_value'])) \
                               - (value_df['asset_value']*((barrier/value_df['asset_value'])**(2*value_df['eta'])))*(-1/math.sqrt(2*math.pi))*np.exp(-0.5*value_df['b']**2)*(1/(s*value_df['asset_value'])) \
                                 - value_df['asset_value']*(-barrier**(2*value_df['eta'])*(2*value_df['eta'])*(value_df['asset_value']**(-2*value_df['eta'] - 1)))*norm.cdf(value_df['b']) \
                                           - (barrier/value_df['asset_value'])**(value_df['eta'])*norm.cdf(value_df['b'])\
                                             + (value_df['X']*(barrier/value_df['asset_value'])**(2*value_df['eta'] - 2))*(-1/math.sqrt(2*math.pi))*np.exp(-0.5*(value_df['b'] - s)**2)*(1/(s*value_df['asset_value']))\
                                               + value_df['X']*(-barrier**(2*value_df['eta'] - 2)*(2*value_df['eta'] + 2)*(value_df['asset_value']**(-2*value_df['eta'] + 1)))*norm.cdf(value_df['b'] - s)
                                                                     
    value_df['previous_asset_value'] = value_df['asset_value'].shift(1)
    value_df['Return'] = np.log(value_df['asset_value']) - np.log(value_df['previous_asset_value'])
    value_df.dropna(inplace = True)
    log_likelihood_V = -n/2*math.log(2*math.pi*(volatility**2)*interval) - 1/2*sum((value_df['Return'] - (expected_return - 1/2*volatility**2)*interval)**2/(interval*volatility**2)) - sum(np.log(value_df['asset_value']))
    
    #try to fix the limit range of math.exp
    temp0 = norm.cdf(((expected_return - volatility**2/2)*n*interval + math.log(barrier/V_0))/(math.sqrt(n*interval)*volatility))
    if temp0 > 1e-6:
        temp1 = math.exp(2/(volatility**2)*(expected_return-volatility**2/2)*math.log(barrier/V_0))*temp0
    else:
        temp1 = 0
    log_likelihood_BT_V = log_likelihood_V + sum(np.log(1 - np.exp((-2/(interval*volatility**2))*np.log(value_df['previous_asset_value']/barrier)*np.log(value_df['asset_value']/barrier)))) \
                                                - math.log(norm.cdf(((expected_return - volatility**2/2)*n*interval - math.log(barrier/V_0))/(math.sqrt(n*interval)*volatility)) - temp1)
    log_likelihood_BT_S = log_likelihood_BT_V - sum(np.log(np.abs(value_df['dee_equity_to_asset'])))
    return log_likelihood_BT_S
      

def kmv_parameters_estimating_by_transformed_mle(equity_value_series, debt_value_series, risk_free_rate_series, expected_return_guess, volatility_guess, barrier_guess, time_horizon = 1, interval = 1/250, tol = 1e-6):
    
    expected_return_est, volatility_est, barrier_est  = expected_return_guess, volatility_guess, barrier_guess
    opt_counter = 0
    neg_log_likelihood = np.inf
    while True:
        opt_counter += 1
        if opt_counter > 0:
            previous_neg_log_likelihood = neg_log_likelihood
        #Estep: estimating the asset value given the parameters volatility, barrier
        estimated_asset_value = asset_value_estimating_by_equity_pricing_equation(equity_value_series, debt_value_series, risk_free_rate_series, volatility_est, barrier_est, time_horizon)
        
        
        #Mstep: estimating the parameters by mle given the estimated asset value in Estep
        neg_log_likelihood_func = lambda params : - log_likelihood_func_for_estimated_asset_value(params[0], params[1], params[2], estimated_asset_value, debt_value_series, risk_free_rate_series, time_horizon, interval)
        neg_log_likelihood = neg_log_likelihood_func((expected_return_est, volatility_est, barrier_est))
        
        logging.info('the neg_log_likelihood is %.2f' % neg_log_likelihood)
        barrier_upper_bound = min(estimated_asset_value)
        bounds = ((-.2, .5), (0.05, .5), (0.01,barrier_upper_bound - .01))
        opt_result = optimize.minimize(neg_log_likelihood_func, (expected_return_est, volatility_est, barrier_est), method = 'L-BFGS-B', bounds = bounds, options = {'disp':True})
        logging.info('last mu: %.2f, ml_opt mu: %.2f; last vo:%.2f, ml_opt vo:%.2f; last K:%.2f, ml_opt K:%.2f\n' % (expected_return_est, opt_result.x[0], volatility_est, opt_result.x[1], barrier_est, opt_result.x[2]) )
        expected_return_est, volatility_est, barrier_est = opt_result.x
        estimated_asset_value = asset_value_estimating_by_equity_pricing_equation(equity_value_series, debt_value_series, risk_free_rate_series, volatility_est, barrier_est, time_horizon)
        estimated_return = np.log(estimated_asset_value[1:]) - np.log(estimated_asset_value[:-1])
        volatility_est = np.std(estimated_return)/math.sqrt(interval)
        expected_return_est = np.mean(estimated_return)/interval + 0.5*volatility_est**2                       
        
        if (abs(neg_log_likelihood - previous_neg_log_likelihood) <= tol and opt_counter >= 20) or opt_counter > 100:
            logging.info('Optimization finished!!\n')
            break
    return(estimated_asset_value, expected_return_est, volatility_est, barrier_est)
        


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
    
    K = 0.4
    debt_value_series = [.5]*251
    risk_free_rate_series = [0.05]*251
    volatility_guess = 0.40
    barrier_guess = 0.5
    expected_return_guess = 0.2 
    time_horizon = 1
    equity_value_series = [equity_pricing_by_bt_formula(asset_value_series[k], debt_value_series[k], risk_free_rate_series[k], volatility, K, time_horizon) for k in range(len(asset_value_series))]
    
    
    estimated_asset_value, expected_return_est, volatility_est, barrier_est = kmv_parameters_estimating_by_transformed_mle(equity_value_series, debt_value_series, risk_free_rate_series, expected_return_guess, volatility_guess, barrier_guess, time_horizon)                                                                           
    
    logging.info('estimated_expected_return:%.2f, estimated_volatily: %.2f\n' % (expected_return_est, volatility_est))
