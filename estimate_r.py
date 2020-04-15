# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:29:13 2020

@author: David Billingsley
"""

import pandas as pd
import numpy as np
import scipy.special as sc
import scipy.stats as st
import matplotlib.pyplot as plt

rdf = pd.read_csv('us_covid_data_latest.csv', dtype={'fips':str})

rdf = rdf.drop(['deaths'], axis = 1)

incidence = pd.pivot_table(rdf, values='cases', index = 'date', aggfunc=np.sum)

#gets the difference with the previous row.
incidence['actual_inc'] = incidence['cases'].diff()


incidence = incidence.rename(columns = {'cases':'cumulative_inc'})
incidence = incidence.replace(np.NaN, 1.0)
incidence.index = pd.to_datetime(incidence.index)

poisson = np.random.poisson

#discretized gamma function
#see Chakarborty 2012
def gamma_discrete(k, theta, x):
    
    return sc.gammaincc(k, x/theta) - sc.gammaincc(k, ((x+1)/theta))

def plot_gamma_discrete(k, theta):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    for i in range(len(x)):
        y[i] = gamma_discrete(k, theta, x[i])
    
    
    plt.bar(x,y, align = 'edge', label = 'discrete')
    
    
    gamma_x = np.linspace(0, 12, 100)
    gamma_y = (1/sc.gamma(k)) * (1/theta**k) * np.power(gamma_x, (k-1)) * np.exp(-1 * np.divide(gamma_x, theta))

    
    plt.plot(gamma_x, gamma_y, color= 'red', label = 'continuous')
    plt.legend()
    plt.title('Gamma Distribution, k = ' + str(k) + ', θ = ' + str(theta))
    plt.xticks(np.arange(0,12))
    plt.xlabel('Generation Interval (days)')
    plt.ylabel('Probability')
    
k = 6.14
theta = 0.719

#now to integrate over the column up to a certain date.

#creating a column delta representing the difference in days since
#2020-01-21...makes it simpler to perform the integration calculation
incidence = incidence.reset_index()
incidence = incidence.reset_index()
incidence = incidence.rename(columns = {'index':'delta'})

#gets the incidence up to a specific date
def prior_incidence(date):
    
    return incidence.loc[incidence['date'] <= date]

#this gives you the denominator in the renewal equation
#called moment_int because it is a moment integral
def moment_int(prior):
    
    #working backwards over the days, from the last date in prior, and generate 
    #probabilities that the generating time is equal to that time difference
    #i.e., in the renewal equation start at day i, and go from j = 0 to j = i
    #calculating probability that generating time is i-j..., then assign that
    #probablility to the i-jth row
    probabilities = prior['delta'].apply(lambda x: gamma_discrete(k, theta, prior.shape[0]-x))
    
    #multiply those probabilities by the prior incidence on day j...
    products = probabilities * prior['actual_inc']
    
    #...then add up and return the sum
    return products.sum()

incidence['mom_ints'] = incidence['date'].apply(lambda x: moment_int(prior_incidence(x)))

incidence['R_t']= incidence['actual_inc']/incidence['mom_ints']

    
inc_dropzero = incidence[incidence['R_t'] != 0]    

#We'll take a moving average over 11 days. Let's say we enact a change to policy
#on day i. Since mean generation time is mean 4.41 with stdev 3.17, then we can say 95% confidence that the incidence 
#after 4.41 + 2*3.17 = 10.75 days, none of the new cases will be have arisen before
#we enacted the policy on day i. I.e., you are taking average, but only as far back as there could have been direct transmission.
#since you know that any cases on day i did not come directly from someone who was infected
#before day i-11. So this gives you a sense of whether your policy changes are having an effect or not.
inc_dropzero['R_t_mov_ave'] = inc_dropzero['R_t'].rolling(window = 11, center = True).mean()

#Plot
ytix = np.linspace(0, 5, 21)
inc_dropzero.plot(x='date', y='R_t_mov_ave', title = '11-Day Moving Average of Lower Bound on R(t)', grid = True, yticks = ytix, fontsize=8)    


#bootstrap to get confidence interval

sim_r_t = incidence['actual_inc'].apply(lambda x: poisson(x, 10000))
sims = np.divide(sim_r_t, incidence['mom_ints'])
uppers = sims.apply(lambda x: np.percentile(x, 95))
lowers = sims.apply(lambda x: np.percentile(x, 5))

incidence['upper 95% CI'] = uppers
incidence['lower 95% CI'] = lowers

tail = 11


plt.figure()

ax = incidence[incidence['actual_inc'] != 0].tail(tail).plot(x='date', y=['R_t', 'upper 95% CI', 'lower 95% CI'], color = ['b', '#808080', '#808080'])



print(incidence.tail(1))

#notes/thoughts : Poisson error in some way accounts for undercounting. Proprotional to the incidence. 
#if undercounting rate is the same across some period, then it doesn't make a difference to the calculation
#if undercounting is higher in the past, as it probably was, then our earlier R estimates will tend to overestimate. I
#If undercounting has decreased more or less monotically, then the earlier the estimate, the greater the overestimating. 
#this is what we'd hope because those early R estimates are really outrageous.
#next step might be to get an estimate for undercounting and apply that. 
