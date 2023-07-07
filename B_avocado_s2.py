# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:32:20 2023

@author: kaiyi
"""

import numpy as np
import itertools
import time
import pandas as pd

def v_true_combinations(v_12,v_22,v_32, price_dis,b):
    # define initial values for each product
    v_1 = np.array([v_12/np.exp((1/3)*price_dis[0]*b[0]), v_12])
    v_2 = np.array([v_22/np.exp((1/3)*price_dis[1]*b[1]), v_22])
    v_3 = np.array([v_32/np.exp((1/3)*price_dis[2]*b[2]), v_32])
    # generate all combinations
    combinations = list(itertools.product(v_1, v_2, v_3))

    # convert combinations to numpy array
    data = np.array(combinations)

    return data

def utility_samples(T,v_12,v_22,v_32, price_dis,b,xi):
    w_0 = np.zeros(2)
    w_1 = np.log(np.array([v_12/np.exp((1/3)*price_dis[0]*b[0]), v_12]))
    w_2 = np.log(np.array([v_22/np.exp((1/3)*price_dis[1]*b[1]), v_22]))
    w_3 = np.log(np.array([v_32/np.exp((1/3)*price_dis[2]*b[2]), v_32]))
    
    w_matrix = np.array([w_0,w_1,w_2,w_3])
    
    u_T = np.empty((T,4,2)) 
    for t in range(T):
        for i in range(len(w_matrix)):
            u_T[t,i,:] = w_matrix[i]+xi[t][i]
    
    return u_T

def u_combination(u_matrix):
    k = u_matrix.shape[1]#2
    n = u_matrix.shape[0]#4
    temp = u_matrix[0,0]*np.ones(shape = (k**(n-1),n))
    comb = list(itertools.product(u_matrix[1], u_matrix[2],u_matrix[3]))
    
    temp[:,1:] = comb
    return temp
#=============================================================================
def customer_simulation(u_matrix, price_comb_index):
    I = np.argmax(u_matrix[price_comb_index])
    return I
#calculate the choice proability based on the v_matrix('no-purchase' option is included)
def choice_probs_matrix(v_matrix):#(8,4)
    temp = np.zeros(shape = (v_matrix.shape[0],v_matrix.shape[1]+1))
    for k in range(v_matrix.shape[0]):
        temp[k][1:]=v_matrix[k]/(1+np.sum(v_matrix[k]))
        temp[k][0]=1/(1+np.sum(v_matrix[k]))    
    return temp
#------------------------------------------------------------------------------
# Input the origianl prices for N products and related K disocunts to get price K^N combiantions
class Combination():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.collection = np.zeros(shape=(len(self.x),len(self.y)))
    def the_set(self):        
        for xx in range(len(self.x)):
            self.collection[xx,:]=self.x[xx]*self.y
            
        self.collection = np.array([cc for cc in itertools.product(*self.collection)])
        return self.collection
#--------------------------------------------------------------------------------
# Using enumeration method to find the optimal price combination based on the 
# input choice probabilities matrix
def optimal_row_index(probs_matrix,prices):
    #probs_matrix = choice_probs_matrix(w_matrix)   
    e_r_matrix = prices*probs_matrix[:,1:]
    r_sumrow = np.sum(e_r_matrix,axis=1)
    return np.argmax(r_sumrow,axis=0)

def expected_revenue(price_vect, choice_probs_vect):
    return np.sum(price_vect*choice_probs_vect[1:])

# update the sales data once we observe customer choice in each tome period
def index_match(ct):
    temp_t = np.zeros(4)
    temp_t[ct]=1
    return temp_t

def TS_Beta(T,price_org,discounts,b_samples,xi):
    # Input--------------------------------------------------------------------
    price_combs = Combination(price_org,discounts).the_set()
    price_dis = np.array([2.7, 1.5, 0.9])
    #--------------------------------------------------------------------------
    #-----------Benchmark Setting & Optimal Solution--------------------------------------------------------
    v_12,v_22,v_32 = (0.06375, 0.74375, 1.3175)
    v_matrix_true =  v_true_combinations(v_12,v_22,v_32, price_dis,b_samples)
    u_true_T = utility_samples(T+1,v_12,v_22,v_32, price_dis,b_samples,xi)
    
    choice_probs_matrix_true = choice_probs_matrix(v_matrix_true)
    # get the row index of true optimal price combiantion
    optimal_index_true = optimal_row_index(choice_probs_matrix_true,price_combs)
    
    r_opt_true = expected_revenue(price_combs[optimal_index_true], 
                                  choice_probs_matrix_true[optimal_index_true])
    
    #----------------------------------------------------------------------------
    #---Initialization----------------------------------------------------------
    N = 3 # numbers of products
    K = 2 # numbers of dicounts (Including the original discount)
    M = K**N # numbers of price combiantions(numbers of arms)
    #--------------------------------------------------------------------------
    #Setting the priors--------------------------------------------------------
    n_priors = np.ones(M)
    V_priors = np.ones(shape = (M,N))
    #--------------------------------------------------------------------------
    n_posts = n_priors
    V_posts = V_priors   
    v_matrix_ts = np.zeros(shape=(M,N))
    t = 1    
    #--------------------------------------------------------------------------
    revenue_t = []
    regret_t = []
    
    #--------------------------------------------------------------------------
    arm_select = []
    while t<=T:
        for m in range(M):
            for i in range(N):
                sampled_theta = np.random.beta(n_posts[m],V_posts[m][i])
                
                v_matrix_ts[m][i] = (1/sampled_theta)-1
        #----------------------------------------------------------------------
        # Select a price combaintion with the max expected revenue based on sampled v matrix
        choice_probs_matrix_ts = choice_probs_matrix(v_matrix_ts)
        optimal_index_ts = optimal_row_index(choice_probs_matrix_ts,price_combs)
        
        
        r_opt_ts = expected_revenue(price_combs[optimal_index_ts], 
                                      choice_probs_matrix_true[optimal_index_ts])
        #------------------------------------------------------------------------
        # offer the selected price combination to customers repeadly
        # untile we observe 'no-purchase' outcome or we reach to end of the selling season.
        customers_choices = []
        while True:
            #customer choice behaviour at time period t
            u_matrix_t = u_true_T[t]
            u_combination_t = u_combination(u_matrix_t)
            ct = customer_simulation(u_combination_t, optimal_index_ts)
            
            # to record customer choice at time period t: if prodcut i is chosen,
            # then, the index i of C_t equals to 1.
            C_t = index_match(ct)
            # calculate the revenue obtained by TS at time period t-----------
            r_ts_t = np.sum(price_combs[optimal_index_ts]*C_t[1:])
            # Record the revenue obtained by TS during the whole selling season
            revenue_t.append(r_ts_t)
            #-------------------------------------------------------------------
            arm_select.append(optimal_index_ts) # record the price combination be selected at time period t
            
            regret_t.append(r_opt_true-r_opt_ts)
            t = t+1
            if ct==0 or t>T:
                customers_choices.append(ct)#record the sales of each product in an epoch
                break
            else:
                customers_choices.append(ct)#record the sales of each product in an epoch
        sales_epoch = np.array([customers_choices.count(1),
                                customers_choices.count(2),
                                customers_choices.count(3)])
        #----------------------------------------------------------------------
        # Posterior Update-----------------------------------------------------
        n_posts[optimal_index_ts] = n_posts[optimal_index_ts] + 1
        V_posts[optimal_index_ts] = V_posts[optimal_index_ts] + sales_epoch
            
    return revenue_t, regret_t, arm_select
#========================================================================================

#========================================================================================
def TS_Beta_varsInstances(T,price_org, discounts, b_samples, simulation_times):
    revenu_perT_collection = np.zeros(shape = (T,simulation_times))
    regret_perT_collection = np.zeros(shape = (T,simulation_times))
    arm_selected_perT_collection = np.zeros(shape = (T,simulation_times))
    
    for run in range(simulation_times):
        np.random.seed(run)
        xi = np.random.gumbel(0,1,size=(T+1,4))
        revenu_perT_collection[:,run], regret_perT_collection[:,run], arm_selected_perT_collection[:,run] = TS_Beta(T,price_org,discounts,b_samples,xi)
   
    return revenu_perT_collection, regret_perT_collection, arm_selected_perT_collection

#=================================================================================
#==============================================================================                              
start = time.time()


T = 240*90
b_samples = np.ones(3)


data_revenue, data_regret, data_arms = TS_Beta_varsInstances(T,
                                    np.array([3.6, 2.0, 1.2]),
                                    np.array([1.00, 0.75]),
                                    b_samples,
                                    400)
#------------------------------------------------------------------------------
#-----average revenue_t based on simulation_times = ---------------------------
beta_revenue_t_average = pd.DataFrame(np.mean(data_revenue, axis=1))
b_revenue_filename = "b_revenue_t_{}_{}.csv".format(T, "_".join(map(str, b_samples)))
beta_revenue_t_average.to_csv(b_revenue_filename)
#------------------------------------------------------------------------------
#-----average regret_t based onb simulation_times = ---------------------------
beta_regret_t_average = pd.DataFrame(np.mean(data_regret, axis=1))
b_regret_filename = "b_regret_t_{}_{}.csv".format(T, "_".join(map(str, b_samples)))
beta_regret_t_average.to_csv(b_regret_filename)
#---------------------------------------------------------------------------------
#-----Price combination selected collection------------------------------------
beta_arms = pd.DataFrame(data_arms)
B_arms_filename = "b_arms_t_{}_{}.csv".format(T, "_".join(map(str, np.round(b_samples,2))))
beta_arms.to_csv(B_arms_filename)