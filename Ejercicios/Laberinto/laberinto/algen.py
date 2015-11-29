#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ejercicio del Laberinto

Taller de la PyConEs 2015: Simplifica tu vida con sistemas complejos y algoritmos genéticos

Este script contiene las funciones y clases necesarias para el algoritmo genético del
ejercicio del laberinto.


Este script usa arrays de numpy, aunque no debería ser difícil para alguien con experiencia
sustituírlos por otras estructuras si es necesario.

También usa la librería Matplotlib en las funciones que dibujan resultados."""



import random as random



#This function was taken from Eli Bendersky's website
#It returns an index of a list called "weights", 
#where the content of each element in "weights" is the probability of this index to be returned.
#For this function to be as fast as possible we need to pass it a list of weights in descending order.
def weighted_choice_sub(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i



            
generate_random_binary_list = lambda n: [random.randint(0,1) for b in range(1,n+1)]




class Individual (object):
    
    def __init__(self, genome):
        
        self.genome = genome        
        self.traits = {}
        self.performances = {}
        self.fitness = 0



def generate_genome (dict_genes):
    
    #We first calculate the total number of bits that the genome must contain
    number_of_bits = sum([dict_genes[trait] for trait in dict_genes])
    
    #And we return a random genome of this length
    return generate_random_binary_list(number_of_bits)



def calculate_traits (individual, dict_genes):
    #This function must decipher the genome and return the traits of the individual.
    #Normally, the genome contains binary numerical values for the different traits.
    
    dict_traits = {}
    index = 0
    
    for trait in dict_genes:
        dict_traits[trait] = int(''.join(str(bit) for bit in individual.genome[index : index+dict_genes[trait]]), 2)
        index += dict_genes[trait]
        
    individual.traits = dict_traits




def immigration (society, target_population,
                 calculate_performances, calculate_fitness, 
                 dict_genes, Object = Individual, *args):
    
    while len(society) < target_population:
        
        new_individual = Object (generate_genome (dict_genes), args)
        calculate_traits (new_individual, dict_genes)
        calculate_performances (new_individual)
        calculate_fitness (new_individual)
        
        society.append (new_individual)





def crossover (society, reproduction_rate, mutation_rate,
               calculate_performances, calculate_fitness, 
               dict_genes, Object = Individual, *args):
    
    #First we create a list with the fitness values of every individual in the society
    fitness_list = [individual.fitness for individual in society]
    
    #We sort the individuals in the society in descending order of fitness.   
    society_sorted = [x for (y, x) in sorted(zip(fitness_list, society), key=lambda x: x[0], reverse=True)] 
    
    #We then create a list of relative probabilities in descending order, 
    #so that the fittest individual in the society has N times more chances to reproduce than the least fit,
    #where N is the number of individuals in the society.
    probability = [i for i in reversed(range(1,len(society_sorted)+1))]
    
    #We create a list of weights with the probabilities of non-mutation and mutation
    mutation = [1 - mutation_rate, mutation_rate]    
    
    #For every new individual to be created through reproduction:
    for i in range (int(len(society) * reproduction_rate)):
        
        #We select two parents randomly, using the list of probabilities in "probability".
        father, mother = society_sorted[weighted_choice_sub(probability)], society_sorted[weighted_choice_sub(probability)]
        
        #We randomly select two cutting points for the genome.
        a, b = random.randrange(0, len(father.genome)), random.randrange(0, len(father.genome))
        
        #And we create the genome of the child putting together the genome slices of the parents in the cutting points.
        child_genome = father.genome[0:min(a,b)]+mother.genome[min(a,b):max(a,b)]+father.genome[max(a,b):]
        
        #For every bit in the not-yet-born child, we generate a list containing 
        #1's in the positions where the genome must mutate (i.e. the bit must switch its value)
        #and 0's in the positions where the genome must stay the same.
        n = [weighted_choice_sub(mutation) for ii in range(len(child_genome))]
        
        #This line switches the bits of the genome of the child that must mutate.
        mutant_child_genome = [abs(n[i] -  child_genome[i]) for i in range(len(child_genome))]
        
        #We finally append the newborn individual to the society
        newborn = Object(mutant_child_genome, args)
        calculate_traits (newborn, dict_genes)
        calculate_performances (newborn)
        calculate_fitness (newborn)
        society.append(newborn)        



def tournament(society, target_population):
    
    while len(society) > target_population:
        
        #index1, index2 = random.randrange(0, len(society)), random.randrange(0, len(society))
        
        #if society[index1].fitness > society[index2].fitness:
        #    society.pop(index2)
        #else:
        #    society.pop(index1)
            
        fitness_list = [individual.fitness for individual in society]
        society.pop(fitness_list.index(min(fitness_list)))