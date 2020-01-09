# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:39:14 2018

@author: Admin
"""
import generation
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distributions

plt.rcParams.update({'font.size': 13})

class gen_series:
    '''
    gen_series
    Contains a series of generations
    
    Attributes:
        self.gen_list:
            A list of all generations startin with the oldest and ending with
            the most recent generation
        self.opt_behav_seq:
            Optimal behavioural fitness, that would lead to maximal fitness of
            an individual living in the current environment
    '''
    
    def __init__(self):
        self.gen_list = []
        
    
    def run(self, num_generations, num_inds, len_behav_seq = 100, seed = None, opt_behav_seq = None, mutation_var = 0.0001, selection_dist_genotype = "lin", selection_dist_behav = "exp", copy_method = "fittest_ind", copy_fidelity = None, individual_learning = True, env_change_ind_life = False, env_change = False, env_change_var = 0.05, **kwargs):
        '''
        Runs a simulation creating a gen_series
        num_generations: number of generations
        num_inds: number of individuals per generation
        
        optional:
            len_behav_seq: Length of the behavioural sequence of each individual
                (custom length: 100)
            opt_behav_seq: [list of floats] Set optimal behavioural sequence manually
            
        **kwargs:
            learning_events: 
                every number definitely a learning event
            learning_events_env_change:
                not necessarily every event a learning event, depends on genotype
            env_change_ind_life = False
            var_env_change_ind_life = 0.05
            time_steps_env_change_ind = 1
            
        '''
        if opt_behav_seq == None:
            self.__generate_optimal_behav_seq(len_behav_seq)
        else:
            self.opt_behav_seq = opt_behav_seq
            len_behav_seq = len(opt_behav_seq)
        
        gen0 = generation.generation()
        gen0.first_generation(num_inds, len_behav_seq = len_behav_seq, seed = seed, opt_behav_seq = self.opt_behav_seq, env_change_ind_life = env_change_ind_life, **kwargs)
        sys.stdout.write("Generation 1")
        #gen0.do_calculations()
        gen_list = []
        gen_list.append(gen0)
        
        for i in range(num_generations - 1):
            '''!!!!Adding env_change_ind_life later on!!!!!!!'''
            '''
            if i == 250:
                print("Environmental change during individual's life from generation 250" )
                env_change_ind_life = True
            '''    
            if env_change == True:
                self.__env_change(env_change_var)
            #For test purposes
            sys.stdout.write(" " + str(i + 2))
            #print("Generation", i + 2)
            gen_next = generation.generation()
            gen_prev = gen_list[-1]
            gen_next.inherit(gen_prev, mutation_var = mutation_var, selection_dist_genotype = selection_dist_genotype, selection_dist_behav = selection_dist_behav,  copy_method = copy_method, copy_fidelity = copy_fidelity, opt_behav_seq = self.opt_behav_seq, individual_learning = individual_learning, env_change_ind_life = env_change_ind_life, **kwargs)
            #gen_next.do_calculations()
            gen_list.append(gen_next)
        self.gen_list = gen_list
        sys.stdout.write("\n")
        
    def append(self, num_generations):
        """
        Appends a number of generations specified by "num_generations" to
        the current gen_series
        """
        
        for i in range(num_generations):
            gen_prev = self.gen_list[-1]
            gen_next = generation.generation()
            gen_next.inherit(gen_prev)
            #gen_next.do_calculations()
            self.gen_list.append(gen_next)
            
        
        
    
    def attrs_tolist(self, *attributes):
        '''
        Returns a list of a dict with the attributes of all individuals 
        in a generation.
        The attributes to be returned can be specified as *args
        (attribute1, atribute2,...)
        If one attribute is named "-public-" only public attributes will be returned
        If the attributes are not specified, all attributes are returned
        '''
        
        attr_list = []
        for gen in self.gen_list:
            attr_dict = gen.attrs_todict(*attributes)
            attr_list.append(attr_dict)
        return attr_list
    
    def attrs_toDF(self, *attributes):
        '''
        Returns a pandas DataFrame with the attributes of all individuals 
        in all generations.
        The attributes to be returned can be specified as *args
        (attribute1, atribute2,...)
        If one attribute is named "-public-" only public attributes will be returned
        If the attributes are not specified, all attributes are returned
        '''
        final_attr_dict = {}
        for i, gen in enumerate(self.gen_list):
            curr_attr_dict = gen.attrs_todict(*attributes)
            #Rename keys
            for key, en in curr_attr_dict.items():
                final_attr_dict["Gen" + str(i) + "_" + key] = en
        df = pd.DataFrame.from_dict(final_attr_dict, orient = "index")
        return df
    
    def out_bounders_in_behav_seq(self):
        behav_seqs = self.attrs_toDF("behav_seq")
        gen, ind = np.shape(behav_seqs)
        tot_ct = 0
        bnd_ct = 0
        for i in range(gen):
            for j in range(ind):
                behav_seq = behav_seqs.iloc[i, j]
                for char in behav_seq:
                    tot_ct += 1
                    if char < 0.0 or char > 1.0:
                        bnd_ct += 1
        print(bnd_ct, "chars out of", tot_ct, "exceed bounds of 0 and 1 in behav_seq")
    
    def avg_attr(self, attr_name):
        '''
        Returns list with average attribute of each generation
        '''
        try:
            all_fits = self.attrs_tolist(attr_name)
        except AttributeError:
            raise AttributeError("""
                                 generation object has no attribute 'fitness' 
                                 Probably you forgot to run '.do_calculations()'
                                 on the object, which calculates the fitness
                                 """)
        avg_fits = []
        for en in all_fits:
            fits_in_gen = en[attr_name]
            avg_fit = sum(fits_in_gen) / len(fits_in_gen)
            avg_fits.append(avg_fit)
            
        return avg_fits
    
    def attr_names(self):
        '''
        Returns the attribute names of the individual and checks whether all
        individuals in gen_series object have the same attributes
        '''
        if self.__all_inds_same_attrs and self.__all_gens_same_attrs:
            out_dict = {"Individuals": self.gen_list[0].ind_list[0].__dict__.keys(),
                    "Generations": self.gen_list[0].__dict__.keys(),
                    "Gen_series": self.__dict__.keys()}
        return out_dict
                    
                    
        
        
    
    def plot_avg_attr(self, attr_name = "fitness", save_path = None, ylog = False, ylim = None):
        '''
        Plots average fitness of each generation
        '''
        fits = self.avg_attr(attr_name)
        plt.figure()
        if ylog == False:
            plt.plot(fits)
        else:
            plt.semilogy(fits)
        if ylim != None:
            plt.ylim(ylim)
        plt.ylabel("Average " + attr_name)
        plt.xlabel("Generation")
        if save_path != None:
            plt.savefig(save_path + ".png", dpi = 300, bbox_inches = "tight")
    
    def __rename_meltDF(melt_df2):
        '''
        Helper function for linplot_attr
        '''
        x_labels = []
        for en in melt_df2['variable']:
            num = str()
            for char in list(filter(str.isdigit, en)):
                num += str(char)
            x_labels.append(int(num))   
        melt_df2['x_labels'] = x_labels
        return melt_df2

    def linplot_attr(self, attrs, savename = None):
        '''
        Creates a plot with confidence intervals
        attr_names: list of attributes to be plotted
        '''
        plt.figure(figsize = (10, 10))
        if len(attrs) == 1:
            attr = attrs[0]
            out_df = self.attrs_toDF(attr)
            melt_df = out_df.T.melt()
            #melt_dfL = melt_df.apply(lambda x: int(filter(str.isdigit, x)))
            melt_df = gen_series.__rename_meltDF(melt_df)
            sns.lineplot(x = 'x_labels', y = 'value', data = melt_df)

            plt.xlabel('Generation')
            plt.ylabel(attr)
        elif len(attrs) == 2:
            
            ax1 = plt.subplot(211)
            out_df = self.attrs_toDF(attrs[0])
            melt_df = out_df.T.melt()
            #melt_dfL = melt_df.apply(lambda x: int(filter(str.isdigit, x)))
            melt_df = gen_series.__rename_meltDF(melt_df)
            ax1 = sns.lineplot(x = 'x_labels', y = 'value', data = melt_df)
            plt.xlabel('Generation')
            plt.ylabel(attrs[0])
            
            ax2 = plt.subplot(212)
            out_df = self.attrs_toDF(attrs[1])
            melt_df = out_df.T.melt()
            #melt_dfL = melt_df.apply(lambda x: int(filter(str.isdigit, x)))
            melt_df = gen_series.__rename_meltDF(melt_df)
            ax2 = sns.lineplot(x = 'x_labels', y = 'value',  data = melt_df)
            plt.xlabel('Generation')
            plt.ylabel(attrs[1])
        
        ymin, ymax = plt.ylim()
        locs, labels = plt.yticks()
        #locs = [ymin].append(locs)
        locs = locs.tolist()
        
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        '''
        locs.append(ymin)
        locs.append(ymax)
        locs = np.array(locs)
        labels.append('Ind. Learn')
        labels.append('Copy')
        
        labels[0] = 'Ind.Learn'
        labels[-1] = 'Copy'
        plt.yticks([ymin, ymax], ['ind learning', 'copying'])
        #ax2.set_yticklabels(labels)
        #ax2.annotate('abc',xy=(0.6,-0.8),xytext=(0.9,-0.8),
            #annotation_clip=False)
        '''
        
        a=ax2.get_yticks().tolist()
        a = [str(round(x,2)) for x in a]
        a[1]='Ind. Learn'
        a[-2] = 'Copy'
        ax2.set_yticklabels(a)
        '''
        a=ax2.get_yticks().tolist()
        a = [str(round(x,2)) for x in a]
        a.append('Ind. Learn')
        a.append('Copy')
        #ax2.set_yticklabels(a)
        locs.append(ymin)
        locs.append(ymax)
        #ax.set_yticklocs(locs)

        plt.yticks(locs, a)
        '''
        if savename != None:
            plt.savefig(savename+'.png', dpi=300, bbox_inches = "tight")
        plt.show()
        return locs, a
        
                
                
            
    def plot_attr_single_gen(self, gen_nums, attr_name = "fitness", plot_type = "distribution", save_path = None):
        '''
        Plots fitness within a single generation
        ARGUMENTS
        positional:
            gen_nums: int OR list : number of the plotted generation
                negative numbers allowed to count backwards
                OR list of generation numbers to be plottet
        optional:
            attr_name:
                the name of the attribute to be plotted
                attribute has to have the datatype int or float
                preset: "fitness"
                also suggested: "genotype"
            plot_type:
                "distribution" (preset)
                "bin"
            save_path: specifies path that plot is saved to
        '''
        if type(gen_nums) == int:
            gen_nums = [gen_nums]
        
        for gen_num in gen_nums:
        
            fit_list = self.gen_list[gen_num].attrs_tolist(attr_name)
            num_all_gens = len(self.gen_list) 
            if gen_num < 0:
                gen_num = num_all_gens + gen_num
            
            plt.figure()
            plt.title("Generation #" + str(gen_num))
            if "distribution" in plot_type:
                sorted_fit_list = np.sort(fit_list)[::-1]
                plt.plot(sorted_fit_list)
                plt.ylabel(attr_name)
                plt.xlabel("Individuals sorted acc. to their {}".format(attr_name))
            elif "bin" in plot_type:
                plt.hist(fit_list)
                plt.ylabel("# of Individuals")
                plt.xlabel("Fitness")
                       
                
            if save_path != None:
                plt.savefig(save_path + "_"+ attr_name + "_" + "_Generation_nr" + str(gen_num) + ".png", dpi = 300, bbox_inches = "tight")
            

    
    def __generate_optimal_behav_seq(self, len_behav_seq):
        opt_behav_seq = np.random.uniform(0, 1, len_behav_seq)
        opt_behav_seq = list(opt_behav_seq)
        self.opt_behav_seq = opt_behav_seq
    
    def __all_inds_same_attrs(self):
        '''
        Checks whether all attributes of individuals in gen_series object are the same
        '''
        out_boo = True
        first_gen_attrs = self.gen_list[0].ind_attr_names()
        for i, gen in enumerate(self.gen_list[1::]):
            out_boo = out_boo and (gen.ind_attr_names() == first_gen_attrs)
            if out_boo == False:
                raise Exception(
                        """
                        Not every individual has the same attributes in gen_sereis object.
                        First difference appearing in  generation {}
                        """.format(i)
                        )
        return out_boo
    
    def __all_gens_same_attrs(self):
        '''
        Checks whether all attributes of generations in _gen_series are the same
        '''
        out_boo = True
        first_gen_attrs = self.gen_list[0].__dict__.keys()
        for i, gen in enumerate(self.gen_list[1::]):
            out_boo = out_boo and (gen.__dict__.keys() == first_gen_attrs)
            if out_boo == False:
                raise Exception(
                        """
                        Not every generation has the same attributes in gen_sereis object.
                        First difference appearing in  generation {}
                        """.format(i)
                        )
        return out_boo
    def __env_change(self, env_change_var = 0.05):
        prev_opt_behav_seq = self.opt_behav_seq
        new_opt_behav_seq = []
        for old_char in prev_opt_behav_seq:
            new_char = distributions.truncnorm(loc = old_char, scale = env_change_var, lower = 0.0, upper = 1.0)
            new_opt_behav_seq.append(new_char)
        self.opt_behav_seq = new_opt_behav_seq
        
            
            
    
        
            
        
        
    
        