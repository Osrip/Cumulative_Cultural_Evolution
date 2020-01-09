# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:01:36 2018

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
import individual
import generation
import gen_series



##############################################################

def main():
    gens = gen_series.gen_series()
    #gens.run(#GENERATIONS, #INDS)
    #Standard:
    #100,2000
    gens.run(100, 2000, len_behav_seq = 100, learning_events = 1,  mutation_var = 0.01,
             selection_dist_genotype = "lin" , opt_behav_seq = None, copy_method = "fittest_ind", 
             learning_events_env_change = 5, var_env_change_ind_life = 0.05, time_steps_env_change_ind = 10,
             selection_dist_behav = "uni", env_change_ind_life = False, learn_and_copy_char_wise = False, 
             env_change = True, only_ind_learning = False, math_optimization = False)
    
    
    gen_att_list = gens.attrs_tolist("genotype")
    
    
    
    
    savename = "PLOT"
    gen_arr = [0,1,5,20, -1]
    
    locs, labels = gens.linplot_attr(['fitness', 'genotype'], savename = "plots\\test")
    
    nam = gens.attrs_toDF("genotype")
    alle = gens.attrs_toDF()
    all_attrs_behav = gens.attrs_toDF("fitness", "behav_seq")
    
    '''Math Optimization Results'''
    if False:
        fitness_inds = gens.attrs_toDF("fitness")
        tail = fitness_inds.tail(10)
        tail_arr = tail.values
        arg_max = np.unravel_index(tail_arr.argmax(), tail_arr.shape)
         
        max_res = tail.values.max()
        
        behav_seq_inds = gens.attrs_toDF("behav_seq")
        behav_seq_inds_tail = behav_seq_inds.tail(10)
        x_y_res = behav_seq_inds_tail.iloc[arg_max]
        
        def conversion(in_float):
            #Calculate back from behav_seq float bounds of 0 and 1 back to -10 and 10:
            out_float = (in_float - 0.5) * 20 
            return out_float
            
        print("The maximum possible value for the function is ", max_res, " where x = ",
              conversion(x_y_res[0]), " and y = ", conversion(x_y_res[1]))
    
    
    
    gens.out_bounders_in_behav_seq()
    
if __name__ == '__main__':
    main()





