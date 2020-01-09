# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:43:43 2018

@author: Admin
"""
import numpy as np


class individual:
    
    
    '''
    Attributes:
        self.genotype: Float between 0 and one : How much percent does individual copy; 
            1 - genotype: How much does individual learn individually
        self.behav_seq: behavioural sequence: Sequence of floats
    '''
    def __init__(self, seed_list = None):
        self.__seed_list = seed_list
        self.genotype = 0.0
        self.behav_seq = []
        
    
    def initialize_first_gen(self, genotype, behav_seq):
        self.genotype = genotype
        self.behav_seq = behav_seq
    
    def gen_genotype(self, input_ind, mutation_var = 0.05):
        '''
        Generate Genotype
        input_ind: Individual from previous generation, that genotype is inherited from
        '''
        old_genotype = input_ind.genotype
        
        #Mutation
        mean = old_genotype 
        if self.__seed_list != None:
            np.random.seed(self.__seed_list[0])
        new_genotype = np.random.normal(mean, mutation_var, 1)
        
        # In case new genotype >1 or <0 repeat calculation:
        ct = 0
        while new_genotype > 1 or new_genotype < 0:
            #if ct == 0:
             #   raise Warning("No Seed used -- run not reproducable")
            #if self.seed_list != None:
                #np.random.seed(self.__seed_list[1])
                #!!!!!!!Wenns beim 2. mal nicht klappt gibts nen Problem, da nur 2 SEEDS !!!!!!!!!!
            new_genotype = np.random.normal(mean, mutation_var, 1)
            ct += 1
            if ct == 500 :
                raise RuntimeWarning("Caught up in while loop")
        new_genotype = float(new_genotype)
        self.genotype = new_genotype
                
    def return_attrs(self):
        return  self.__dict__.keys()
    
    def do_calculations(self,**kwargs):
        
        '''
        Calculates all kinds of necessary parameters for the individual.
        Has to be executed after all necessary attributes have been added to 
        the individual.
        (is there better solution than this??)
        '''

                
        self.calculate_fitness(**kwargs)
        #self.env_change_ind_life(**kwargs)
        

        
    def env_change_ind_life(self, env_change_ind_life_list, **kwargs):
        '''
        Simulates learning in environmental change during individual live time
        env_change_ind_life:
        '''
        env_change_ind_life = False
        for key, value in kwargs.items():
            if key == "env_change_ind_life":
                env_change_ind_life = True
                
        if env_change_ind_life == True:
            for changed_opt_behav_seq in env_change_ind_life_list:
                
               # changed_opt_behav_seq = self.__opt_behav_seq_env_change(opt_behav_seq, var = var_env_change_ind_life) #, mutation_var = mutation_var_env_change_ind_life
                self.__individual_learning_ind_life(changed_opt_behav_seq, **kwargs)
                
            
                
    def __individual_learning_ind_life(self, changed_opt_behav_seq, var = 0.05, **kwargs):
        #mutation_var of individual learners could be a genetic trait as well,
        #maybe also what kind of distribution they use!
        #I should not have to use different learning functions, should be the same as before#
        
        learning_events_env_change = 1
        for key, value in kwargs.items():
            if key == "learning_events_env_change":
                learning_events_env_change = value
        
        for i, char in enumerate(changed_opt_behav_seq):
            rand_float = float(np.random.uniform(0, 1, 1))
                #if True :
            for j in range(learning_events_env_change):
                if rand_float > self.genotype:
                    try_char = np.random.normal(loc = char, scale = var)
                    old_fit = self.fitness
                    self.behav_seq[i] = try_char
                    self.calculate_fitness(**kwargs)
                    new_fit = self.fitness
                    if new_fit < old_fit:
                        self.behav_seq[i] = char
                    

            
              
            
    def calculate_fitness(self, **kwargs):
        '''
        The fitness is calculated by first calculating the distance from eac of the characters
        to ech other (resulting in an array with positive difference values)
        All of the values are added up and then divided by the length of the behav_seq
        
        The optimal behavioural sequence is handed over via **kwargs
        
        Example
        self.__calculate_fitness(opt_behav_seq = [.....])
        '''
        ####Handling  kwargs######
        opt_behav_seq_available = False
        math_optimization = False
        for key, value in kwargs.items():
            if key == "test_genotype" and value == True:
                self.fitness = self.genotype
            if key == "math_optimization":
                math_optimization = value
            if key == "opt_behav_seq":
                opt_behav_seq_available = True
                opt_behav_seq = value
                
        ##### Fitness Calculation #####
        if math_optimization == False:
            diff_arr = np.absolute(np.array(opt_behav_seq)- np.array(self.behav_seq))
            avg_dist = np.sum(diff_arr) / len(self.behav_seq)
            self.fitness = 1 - avg_dist
        
        else:
            '''
            Mathematical optimization
            '''
            x = (self.behav_seq[0] - 0.5) * 20
            y = (self.behav_seq[1] - 0.5) * 20
            
            res = 5 + 3 * x - 4 * y - x**2 + x * y - y**2
            self.fitness = res
                
                #Zu Testzwecken: 
                #self.fitness = self.genotype
            
        if opt_behav_seq_available == False:
            raise Exception(
                    """
                    Function '__calculate_fitness()' did not receive 'opt_behav_seq' 
                    (optimal behavioural sequence). It is passed on through different 
                    functions via **kwargs : 
                    self.__calculate_fitness(opt_behav_seq = [--your behavioural sequence---])
                    """
                    )
    