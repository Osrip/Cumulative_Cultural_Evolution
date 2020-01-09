# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 18:34:32 2018

@author: Admin
"""
import individual
import distributions
import numpy as np


class generation:
    
    '''
    Attributes:
        public:
        self.ind_list:
            List of all individual objects in the generation
        
        private:
        self.__seed (optional):
            When initializing the first generation, a seed can be set 
        in order to make the random numbers of a run reproducable
        self.__env_change_ind_life:
            Represents the environmental change within the generation
            List of slighlty modified opt_behav_seq with the 
            Length is equal to the time steps of environmental change
            ( defined by )
    '''
    
    def __init__(self ):
        self.ind_list = []


    
    def first_generation(self, gen_size, len_behav_seq = 100, seed = None, **kwargs):
        '''
        Initialize first generation
        This function is only used in order to initialize the very first generation!
        For the following generations use the "inherit()" function 
        
        gen_size: Number of Individuals in Generation
        
        Please specify 'opt_behav_seq' (optimal behavioural sequence):
            opt_behav_seq = [---your behav_seq ---]
        
        len_behav_seq (optional): The length of the behavioural sequences
        seed (optional): If a seed is given, the run can be reproduced using the exact same seed 
        '''
        
        # SEEDING
        self.__seed = seed
        
        if self.__seed != None:
            seed_list1 = generation.__create_seed_list(gen_size * 100, self.__seed)
            #Creating a second seed list
            seed_list2 = generation.__create_seed_list(gen_size, self.__seed + 1111)
            seed_list3 = generation.__create_seed_list(gen_size, self.__seed + 2222)
            
    
        #Start of the real function
        
        ind_list = []
        for i in range(gen_size):
            if self.__seed != None:
                ind_seed_list = seed_list1[i*100 : (i+1)*100]
                ind = individual.individual(seed_list = ind_seed_list)
                np.random.seed(seed_list2[i])
                
            else:
                ind = individual.individual()
            #Initialize genotype    
            ind.genotype = np.random.uniform(low = 0.0, high = 1.0)
                   
            #Initialize behav_seq
            if self.__seed != None:
                np.random.seed(seed_list3[i])
            ind.behav_seq = list(np.random.uniform(low = 0.0, high = 1.0, size = len_behav_seq))
             
            ind_list.append(ind)
            
        
        self.ind_list = ind_list
        self.do_calculations(**kwargs)
        self.__create_env_change_ind_life(**kwargs)
        self.__simulate_env_change_ind_life(**kwargs)
        
        
    def __create_seed_list(size, seed):
        '''
        SEEDING
        Create a seed for every individual of the group in case seed is not None
        This again is done with a seeded random parameter
        Therefore the whole run remains predictable, but there is no systematic 
        Error in th random nubers o the indiviuals (i think)
        '''
        
        
        np.random.seed(seed)
        
        seed_list = np.random.randint(low = 0, high = 2**32 - 1, size = size , dtype = "int64")
            #A seed is an integer between 0 and 2**32 - 1
           
            
        return seed_list
    
###NO BEHAV SEQ INHERITED??!!! --> no learn and copy done yet####        
    
        
    def inherit(self, prev_gen, mutation_var = 0.001, selection_dist_genotype = "lin", selection_dist_behav = 'exp', copy_method = "fittest_ind", **kwargs):
        '''
        Creates the attributes for a new generation, based on last generation's
        attributes "inheriting" the genetic and cultural information from the 
        previous generation to the new generation.
        Afterwards you still have to execute .do_calculations
        
        prev_gen : generation eobject containing the previous generation
        
        Please specify 'opt_behav_seq' (optimal behavioural sequence):
            opt_behav_seq = [---your behav_seq ---]
        
        --optional parameters--
        
        mutation_var: variance of normal distribution estimating genetic mutation
        
        selection_dist: selection distribution estimating probability of 
            individual to have a child according to its fitness
            two possibilities: 
                selection_dist = "lin" (preadjusted)
                    linear probability distribution
                election_dist = "exp"
                    exponential probability distribution
                    
        copy_method: 
            Method used for copying
            two possibilities:
                copy_method = "fittest_ind"
                    character in behav_seq is most likely copied from fittest individual
                copy_method = "fittest_char"
                    character that is closest to result in opt_behav_seq is most likely copied
                
        ''' 
        learn_and_copy_char_wise = False
        for key, value in kwargs.items():
            if key == "learn_and_copy_char_wise":
                learn_and_copy_char_wise = value
                
        self.__genetic_inheritance(prev_gen, mutation_var = mutation_var, 
                                   selection_dist_genotype = selection_dist_genotype)  
        if learn_and_copy_char_wise: 
            
            self.__learn_and_copy_char_wise(prev_gen, selection_dist = selection_dist_behav,
                                            copy_method = copy_method, **kwargs)
        else:
            
            self.__learn_and_copy_all(prev_gen, selection_dist = selection_dist_behav, **kwargs)
        self.do_calculations(**kwargs)
        #self.__create_env_change_ind_life(**kwargs)
        #self.__simulate_env_change_ind_life(**kwargs)

    
    def __genetic_inheritance(self, prev_gen, mutation_var = 0.001, selection_dist_genotype = "lin"):
        self.__seed = prev_gen.__seed
        
        gen_size = len(prev_gen.ind_list) 
        prev_ind_list = prev_gen.ind_list
        
        if self.__seed != None:
            seed_list4 = generation.__create_seed_list(gen_size, self.__seed + 3333)

        #original order of fitnesses of individuals
        try:
            fit_orig_order = prev_gen.attrs_tolist("fitness")
            
        except AttributeError:
            raise AttributeError('''
                                 The individuals of a generation don't have a 
                                 'fitness' as an attribute. In order to calculate
                                 it run --generation_object--.do_calculatins()
                                 ''')
            
        fit_orig_order = np.array(fit_orig_order)
        
        #create array that would sort fit_orig_order
        sort_arr = np.argsort(fit_orig_order)
        #invert sort_arr for descending order
        sort_arr = sort_arr[::-1] 
        
        ### inheritance of genotype ###
        new_ind_list = []
        for i in range(gen_size):
            new_ind = individual.individual()
            if self.__seed != None:
                np.random.seed(seed_list4[i])

            if selection_dist_genotype == "lin":
                index_sort_arr = np.array(np.random.triangular(0, 0, len(sort_arr)), dtype = int)
                
            elif selection_dist_genotype == "exp":
                # Exponential random parameter is used, making it most likely that 
                # genetic information of fittest individual is inherited
            
                index_sort_arr = np.array(np.random.exponential(scale = len(sort_arr) / 8 ), dtype = int)
                
                #in case calculated index exceeds index of sort_arr, try again:
                ct = 0
                while index_sort_arr >= (len(sort_arr) - 1) :
                    print("exp rand par exceeded   ", len(sort_arr) )
                    #!!!!!!LÖSUNG FÜR SEEDS SUCHEN, am besten fetten stack bauen für jedes individuum!!!!!!!
                    #if ct == 0:
                     #   raise Warning("No Seed used -- run not reproducable")
                    index_sort_arr = np.array(np.random.exponential(scale = len(sort_arr) / 8), dtype = int)
                    ct += 1
                    if ct == 500:
                        raise RuntimeWarning("Caught up in while loop")
           
            # The index in the sort_arr, that we chose is the index - or number of -individual-
            # in the original order that we chose to inherit genotype from (to be the parent)
            index_orig_order = sort_arr[index_sort_arr]
            parent_ind = prev_ind_list[index_orig_order]
            new_ind.gen_genotype(parent_ind, mutation_var)
            new_ind_list.append(new_ind)
        
        self.ind_list = new_ind_list
        
    def __learn_and_copy_all(self, prev_gen, selection_dist = "exp", **kwargs):
        
        copy_fidelity_factor = 0.05
        fine_adjust_learning_var = 0.05     #0.05
        
        learning_events = 50 #This is the maximum number of possible learning events. On average there are half of those events
        var_learning_events = learning_events * 0.05
        only_ind_learning = False
        
        for key, value in kwargs.items():
            if key == "copy_fidelity_factor" :
                copy_fidelity = value
            if key == "opt_behav_seq":
                opt_behav_seq = value
            if key == "learning_events":
                learning_events = value
            if key == "var_learning_events":
                var_learning_events = val
            if key == "fine_adjust_learning_var":
                fine_adjust_learning_var = value
            if key == "only_ind_learning":
                only_ind_learning = value
                '''
                #Only_ind_learning allows only individual learning; 
                This is useful to see average fitness that can be achieved via ind_learing
                '''
                
        
        len_behav_seq = len(prev_gen.ind_list[0].behav_seq)
        
        try:
            fit_orig_order = prev_gen.attrs_tolist("fitness")
            
        except AttributeError:
            raise AttributeError('''
                                 The individuals of a generation don't have a 
                                 'fitness' as an attribute. In order to calculate
                                 it run --generation_object--.do_calculatins()
                                 ''')
            
        fit_orig_order = np.array(fit_orig_order)
        
        #create array that would sort fit_orig_order
        sort_arr = np.argsort(fit_orig_order)
        #invert sort_arr for descending order
        sort_arr = sort_arr[::-1] 
        
        
        '''
        COPY
        '''
        if only_ind_learning == False:
            for i_ind, ind in enumerate(self.ind_list):
                ind.behav_seq = [None] * len_behav_seq
                
                for i, char in enumerate(ind.behav_seq):
                    #COPY
                    #rand_float = float(np.random.uniform(0, 1, 1))
                    
                    copied_char = self.__copy_fittest_ind(i, prev_gen, sort_arr, selection_dist = selection_dist)
                    #copied_char = self.__copy_fittest_char(i, prev_gen, opt_behav_seq, selction_dist = "exp")
                    
                    #if False:
                    if copy_fidelity_factor != None:
                        '''!!!!!!!!!!!!!! Hier vllt genotyp nicht nur mit 
                        copy fidelity factor nicht nur multiplizieren sondern quadratischen einfluss geben'''
                        copied_char = distributions.truncnorm(loc = copied_char, 
                                                              scale = ((1 - ind.genotype) * copy_fidelity_factor), 
                                                              lower = 0.0, upper = 1.0)
                        '''
                        dev = np.random.normal(loc = 0.0, scale = ((1 - ind.genotype) * copy_fidelity_factor))
                        old_copied_char = copied_char
                        copied_char += dev
                        ct = 0
                        
                        
                        while copied_char > 1 or copied_char < 0:
                            #print(copied_char)
                            #print("old_copied_char", copied_char)
                            dev = np.random.normal(loc = 0.0, scale = ((1 - ind.genotype) * copy_fidelity_factor))
                            copied_char = old_copied_char + dev
                            ct += 1
                            if ct > 300:
                                raise Exception("Caught up in while loop")
                        ''' 
           
                    ind.behav_seq[i] = copied_char
        '''
        LEARN
        '''
        for i_ind, ind in enumerate(self.ind_list):
            if only_ind_learning == True:
                #len_behav_seq fixed to 100 in case of only_ind_learning == True
                ind.behav_seq = list(np.random.uniform(low = 0.0, high = 1.0, size = 100))
            for i, char in enumerate(ind.behav_seq):
                #LEARN
                actual_learning_events = int( np.random.normal(loc = (learning_events * (1 - ind.genotype)),
                                                               scale =  var_learning_events))
                if actual_learning_events < 0:
                    actual_learning_events = 0 
                ind.calculate_fitness(**kwargs)
                old_char = char
                for j in range(actual_learning_events):
                    try_char = distributions.truncnorm(loc = old_char, scale = fine_adjust_learning_var,
                                                       lower = 0.0, upper = 1.0)
                    old_fit = ind.fitness
                    ind.behav_seq[i] = try_char
                    ind.calculate_fitness(**kwargs)
                    new_fit = ind.fitness
                    if new_fit > old_fit:
                        ind.behav_seq[i] = try_char
                    else:
                        ind.behav_seq[i] = old_char
                    old_char = ind.behav_seq[i]
           
                    
               
                
            
    def __learn_and_copy_char_wise(self, prev_gen, selection_dist = "exp", copy_method = "fittest_ind", **kwargs):
        '''
        This function is executed after the execution of "self.inherit()"
        Char wise goes through behav_seq and either gains new char through either
        copying or individual learning, depending on genotype
        '''
        ######## handling **kwargs #############
        copy_fidelity = None
        individual_learning = True
        learning_events = 20
        
        for key, value in kwargs.items():
            if key == "copy_fidelity" :
                copy_fidelity = value
            if key == "opt_behav_seq":
                opt_behav_seq = value
            if key == "individual_learning":
                individual_learning = value
            if key == "learning_events":
                learning_events = value
                
        
        len_behav_seq = len(prev_gen.ind_list[0].behav_seq)
        
        try:
            fit_orig_order = prev_gen.attrs_tolist("fitness")
            
        except AttributeError:
            raise AttributeError('''
                                 The individuals of a generation don't have a 
                                 'fitness' as an attribute. In order to calculate
                                 it run --generation_object--.do_calculatins()
                                 ''')
            
        fit_orig_order = np.array(fit_orig_order)
        
        #create array that would sort fit_orig_order
        sort_arr = np.argsort(fit_orig_order)
        #invert sort_arr for descending order
        sort_arr = sort_arr[::-1] 
        
        
        for i_ind, ind in enumerate(self.ind_list):
            ind.behav_seq = [None] * len_behav_seq
            for i, num in enumerate(ind.behav_seq):
                rand_float = float(np.random.uniform(0, 1, 1))
                #if True :
                if rand_float < ind.genotype:
                    #Copying: Social learning
                    if copy_method == "fittest_ind":
                        copied_char = self.__copy_fittest_ind(i, prev_gen, sort_arr,
                                                              selection_dist = selection_dist)
                    elif copy_method == "fittest_char":
                        copied_char = self.__copy_fittest_char(i, prev_gen,
                                                               opt_behav_seq, selection_dist = selection_dist)
                    
                    if copy_fidelity != None and copy_fidelity != False :
                        dev = np.random.normal(loc = 0.0, scale = copy_fidelity)
                        old_copied_char = copied_char
                        copied_char += dev
                        ct = 0
                        
                        while copied_char > 1 or copied_char < 0:
                            #print(copied_char)
                            #print("old_copied_char", copied_char)
                            dev = np.random.normal(loc = 0.0, scale = copy_fidelity * 0.01)
                            copied_char = old_copied_char + dev
                            ct += 1
                            if ct > 300:
                                raise Exception("Caught up in while loop")
                        
                        
                    ind.behav_seq[i] = copied_char
                else:
                    #Learn individually
                    
                    if individual_learning == False :
                        
                        ind.behav_seq[i] = prev_gen.ind_list[i_ind].behav_seq[i]
                    else:
                        #######TEST######
                        #ind.behav_seq[i] = prev_gen.ind_list[i_ind].behav_seq[i]
                        
                        #Try different random numbers and pick the one that leads to highest fitness of individual
                        #Lets algorithm take a look at solution.
                        #But could be implemented without by replacing all characters with random characters and then
                        #calculating fitness for each charactar in trials
                        #!!!!!!!!!!!!!DOES NOT WORK WITH SEVERAL MAXIMA IN FITNESS FUNCTION!!!!!!!!
                        #(however it is implemented)
                        trials = np.random.uniform(size = learning_events)
                        trial_arr = np.array(trials)
                        opt_char = opt_behav_seq[i]
                        diff_arr = trial_arr - opt_char
                        fittest_char_at = np.argmin(diff_arr)
                        
                        fittest_char = trial_arr[fittest_char_at]
                        ind.behav_seq[i] = fittest_char
                        
                        
                        
                        
    
    def __copy_fittest_char(self, char_i, prev_gen, opt_behav_seq, selection_dist = "lin"):
        
        curr_char = opt_behav_seq[char_i]
        
        char_list = []
        for ind in prev_gen.ind_list:
            char_list.append(ind.behav_seq[char_i])
        
        
        curr_char_arr = np.full(len(char_list), curr_char)
        #Inverse fitness of each char
        char_inv_fit = np.absolute(curr_char_arr - np.array(char_list))
        #array that would sort char_inv_fit from largest to smallest number
        argsort_char_inv_fit = np.argsort(char_inv_fit)
        
        char_arr = np.array(char_list)
        sorted_char_arr = char_arr[argsort_char_inv_fit]
        
        sort_arr = sorted_char_arr
        if selection_dist == "lin":
            index_sort_arr = np.array(np.random.triangular(0, 0, len(sort_arr)), dtype = int)
        elif selection_dist == "exp":
            index_sort_arr = np.array(np.random.exponential(scale = len(sort_arr) / 8 ), dtype = int)
            
            #in case calculated index exceeds index of sort_arr, try again:
            ct = 0
            while index_sort_arr >= (len(sort_arr) - 1) :
                #print("exp rand par exceeded   ", ct )
                
                #!!!!!!LÖSUNG FÜR SEEDS SUCHEN, am besten fetten stack bauen für jedes individuum!!!!!!!
                #if ct == 0:
                 #   raise Warning("No Seed used -- run not reproducable")
                index_sort_arr = np.array(np.random.exponential(scale = len(sort_arr) / 8), dtype = int)
                ct += 1
                if ct == 500:
                    raise RuntimeWarning("Caught up in while loop")
           
        copied_char = sorted_char_arr[index_sort_arr]
        return copied_char
        
            
        
    def __copy_fittest_ind(self, char_i, prev_gen, sort_arr, selection_dist = "exp"):
        '''
        For every character in behav_seq a new individual is chosen that the character is copied from.
        It is more likely that an individual with a large fitness is chosen. 
        However that does not mean that the particular character that is copied 
        leads to the higher fitness of the individual but only the whole sequence.
        
        So the individuals only base their choice from which individual to copy
        the character based on their overall fitness.
        
        !!!!!!!!!!!!
        This means that the longer the behav_seq, the less efficient copying gets,
        because it becomes less likely that the copied character actually leads
        to the high fitness of the individual that is copied from
        !!!!!!!!!!!!
        
        arguments:
            char_i:
                character index ... index of current character in behav_seq
            prev_gen:
                generation object of previous generation    
            sort_arr: 
                array that would sort fit_orig_order
        '''
        if selection_dist == "uni":
            len_ind_list = len(prev_gen.ind_list)
            uni_rand = np.random.randint(0, len_ind_list)
            copied_ind = prev_gen.ind_list[uni_rand]
            copied_char = copied_ind.behav_seq[char_i]
            return copied_char
        
        elif selection_dist == "lin":
            index_sort_arr = np.array(np.random.triangular(0, 0, len(sort_arr)), dtype = int)
            
        elif selection_dist == "exp":
            # Exponential random parameter is used, making it most likely that 
            # genetic information of fittest individual is inherited
        
            index_sort_arr = np.array(np.random.exponential(scale = len(sort_arr) / 8 ), dtype = int)
            
            #in case calculated index exceeds index of sort_arr, try again:
            ct = 0
            while index_sort_arr >= (len(sort_arr) - 1) :
                #print("exp rand par exceeded   ", ct )
                
                #!!!!!!LÖSUNG FÜR SEEDS SUCHEN, am besten fetten stack bauen für jedes individuum!!!!!!!
                #if ct == 0:
                 #   raise Warning("No Seed used -- run not reproducable")
                index_sort_arr = np.array(np.random.exponential(scale = len(sort_arr) / 8), dtype = int)
                ct += 1
                if ct == 500:
                    raise RuntimeWarning("Caught up in while loop")
        
        index_orig_order = sort_arr[index_sort_arr]
        copied_ind = prev_gen.ind_list[index_orig_order]
        
        copied_char = copied_ind.behav_seq[char_i]
        return copied_char
        
        

    
    def attrs_tolist(self, *attributes):
        '''
        Returns lists of all attributes of individual object specified in *attributes variables
        
        Example:
            attribute_tolist(["genotype", "phenotype"])
            returns list of all genotypes and phenotypes of individuals in generation
            (in order of input )
            
        If only one attribute is specified not a list of lists is outputted but only one list:
            
        Example:
             attribute_tolist("genotype")
             would only return a list of all genotypes
            
        '''
        
        self.__all_inds_same_attr()
        
        list_of_attribute_lists = []
        for att in attributes:
            ind_attribute_list = []
            for ind in self.ind_list:
                ind_attribute = getattr(ind, att)
                ind_attribute_list.append(ind_attribute)
            list_of_attribute_lists.append(ind_attribute_list)
            
            if len(list_of_attribute_lists) == 1:
                return list_of_attribute_lists[0]
        return list_of_attribute_lists
    
    def attrs_todict(self, *attributes):
        '''
        Returns a dict of attributes
        If attributes are not specified, all attributes are returned.
        Otherwise only the attributes specified in *attributes are returned
        
        In case the name "-public-" appears in *attributes, only the public 
        attributes are returned
        
        Example1:
            gen.dict_of_attr()
            returns all attributes
        Example2:
            gen.dict_of_attr(["genotype", "phenotype"])
            returns dict only of genotype and phenotype#
        Exampe3:
            gen.dict_of_attr("-public-")
            returns only public attributes
        
        It is assumed that all individuals of one generation have the same attributes
        '''
        self.__all_inds_same_attr()
        
        public_in_attrs = False
        if "-public-" in attributes:
            attributes = list(attributes)
            attributes.remove("-public-")
            attributes = tuple(attributes)
            public_in_attrs = True
        
        if len(attributes) == 0 or ( len(attributes) == 1 and public_in_attrs):
            ind1 = self.ind_list[0]
            attr_keys = ind1.__dict__.keys()
                  
        
        else :
            attr_keys = attributes
            
        if public_in_attrs:
            new_attr_keys = []
            for en in attr_keys:
                if not en.startswith("_"):
                    new_attr_keys.append(en)
            attr_keys = new_attr_keys
        
        dict_of_attribute_lists = {}
        for key in attr_keys:
            ind_attribute_list = []
            for ind in self.ind_list:
                ind_attribute = getattr(ind, key)
                ind_attribute_list.append(ind_attribute)
            dict_of_attribute_lists[key] = ind_attribute_list
        
        return dict_of_attribute_lists
                
        
        
    def __all_inds_same_attr(self):
        '''
        Checks whether all individuals have the same object attributes
        '''
        
        out_boo = True
        
        ind1 = self.ind_list[0]
        
        for ind in self.ind_list:
            boo = ind.__dict__.keys() == ind1.__dict__.keys()
            out_boo = boo and out_boo
            
        if out_boo == False:
            raise Exception(
                    '''Individuals of this generation have different attributes!
                    They are supposed to have similar attributes''')
        else:
            return out_boo
            
    def ind_attr_names(self):
        '''
        Checks whether all individuals in generation object have the same attributes
        and returns them. Otherwise returns error message
        '''
        if self.__all_inds_same_attr():
            return self.ind_list[0].__dict__.keys()
    
    def __create_env_change_ind_life(self, **kwargs):
        '''
        Creates the class attribute self.env_change_ind_life, which is a list
        of slightly modified opt_behav_seq
        '''

        time_steps_env_change_ind = 1
        var = 0.05
        for key, value in kwargs.items():
            if key == "var_env_change_ind_life":
                #optional
                    var = value
            if key == "opt_behav_seq":
                opt_behav_seq = value
            if key == "time_steps_env_change_ind":
                #optional
                time_steps_env_change_ind = value

        env_change_ind_life = []
        for i in range(time_steps_env_change_ind):
            
            new_opt_behav_seq = []
            for old_char in opt_behav_seq:
                new_char = distributions.truncnorm(loc = old_char, scale = var, lower = 0.0, upper = 1.0)
            new_opt_behav_seq.append(new_char)
            
            env_change_ind_life.append(new_opt_behav_seq)
        self.env_change_ind_life = env_change_ind_life    

    def __simulate_env_change_ind_life(self, **kwargs):
        '''
        Simulates 
        '''
        env_change_ind_life = False
        for key, value in kwargs.items():
            if key == "env_change_ind_life":
                env_change_ind_life = value
                
        if env_change_ind_life:
            for ind in self.ind_list:
                ind.env_change_ind_life(self.env_change_ind_life, **kwargs)
            
    
    def do_calculations(self, **kwargs):
        '''
        Calculates all kinds of necessary parameters for each individual in the generation.
        Has to be executed after all necessary attributes have been added to 
        the individual.
        (is there a better solution than this??)
        '''
        
        for ind in self.ind_list:
            ind.do_calculations(**kwargs)
    
    

            