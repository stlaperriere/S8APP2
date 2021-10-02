# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import os
import sys
import time
import numpy as np
import logging

sys.path.append('../..')
from torcs.optim.core import TorcsOptimizationEnv, TorcsException

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)



################################
# Define hyperparameters here
################################
cross_rate = 0.8 # Crossover rate to determine if crossover is performed
mutation_rate = 0.5 # Mutation rate to determine if mutation ocures
population_size = 100
boundaries = [[0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [1, 10], [0, 90], [0, 90]]
n_bits = 8
n_gen = 15
n_dead = 25
################################
# Define helper functions here
################################
def fitness_function(speed, dist, fuel):
    return speed

def mutation():
    pass

def reproduction(p1, p2):
    first_half = p1[:int(len(p1)/2)]
    second_half = p2[int(len(p2)/2):]
    return np.concatenate((first_half, second_half))

def select_parent(scores, dead_indexes):
    while(True):
        index = select_by_fitness(scores)
        if index not in dead_indexes:
            return index
    
def select_by_fitness(scores):
    max = sum(scores)
    pick = np.random.uniform(0, max)
    current = 0
    for i, score in enumerate(scores):
        current += score
        if current > pick:
            return i
        
def select_k_lowest(scores, k):
    arr = np.array(scores)
    return arr.argsort()[:k]

# Return n bits at index from a bit list
def get_bits(bit_list, index, n_bits):
    return bit_list[index: index + n_bits]

def bit_to_int(bit_list):
    result = 0
    for bit in bit_list:
        result = (result << 1) | bit

def scale_value(value, n, new_max, new_min):
    return ((new_max - new_min) / 2**n ) * (value - 2**n) + new_max

def main():
    
    try:
        maxEvaluationTime = 40.0  # sec
        with TorcsOptimizationEnv(maxEvaluationTime) as env:
            
            # Create n randoms parents
            population = [np.random.randint(0, 2, n_bits*len(boundaries)).tolist() for _ in range(population_size)]
            
            best_person, best_score = 0, fitness_function(-1, -1, -1)
            
            # Loop a few times for demonstration purpose
            for i in range(n_gen):
                logger.info("Generation : %d", i)
                scores = []
                for j, person in enumerate(population):
                    parameters = {'gear-2-ratio':               np.array([scale_value(int("".join(str(x) for x in person[:n_bits]), 2), n_bits, 5, 0.1)]),
                                  'gear-3-ratio':               np.array([scale_value(int("".join(str(x) for x in person[n_bits:2*n_bits]), 2), n_bits, 5, 0.11)]),
                                  'gear-4-ratio':               np.array([scale_value(int("".join(str(x) for x in person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.11)]),
                                  'gear-5-ratio':               np.array([scale_value(int("".join(str(x) for x in person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.11)]),
                                  'gear-6-ratio':               np.array([scale_value(int("".join(str(x) for x in person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.11)]),
                                  'rear-differential-ratio':    np.array([scale_value(int("".join(str(x) for x in person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1.1)]),
                                  'rear-spoiler-angle':         np.array([scale_value(int("".join(str(x) for x in person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0)]),
                                  'front-spoiler-angle':        np.array([scale_value(int("".join(str(x) for x in person[7*n_bits:]), 2), n_bits, 90, 0)])}
                    
                    # Perform the evaluation with the simulator and p2 not in dead_indexes
                    observation, _, _, _ = env.step(parameters)
                    
                    """
                    # Display simulation results
                    logger.info('##################################################')
                    logger.info('Results:')
                    logger.info('Time elapsed (sec) =   %f', maxEvaluationTime)
                    logger.info('Top speed (km/h)   =   %f', observation['topspeed'][0])
                    logger.info('Distance raced (m) =   %f', observation['distRaced'][0])
                    logger.info('Fuel used (l)      =   %f', observation['fuelUsed'][0])
                    logger.info('##################################################')
                    """
                    # Calcul du score
                    scores.append(fitness_function(observation['topspeed'][0], observation['distRaced'][0], observation['fuelUsed'][0]))
                  
                    # Find best person
                    if scores[j] > best_score:
                        
                        best_person, best_score = population[j], scores[j]
                        logger.info(" ")
                        logger.info("new best = %f", best_score)
                
                
                if i < n_gen:
                    # Survivor selection
                    dead_indexes = select_k_lowest(scores, n_dead)
                    
                    # Create a child for each dead person
                    for j in range(n_dead):
                        # Select 2 parents that survived
                        p1 = select_parent(scores, dead_indexes)
                        p2 = select_parent(scores, dead_indexes)
                       
                        # Create a new child
                        child = reproduction(population[p1], population[p2])
                        
                        # Replace a dead person with the new child
                        population[dead_indexes[j]] = child
                
            parameters = {'gear-2-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[:n_bits]), 2), n_bits, 5, 0.1)]),
                          'gear-3-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[n_bits:2*n_bits]), 2), n_bits, 5, 0.1)]),
                          'gear-4-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.1)]),
                          'gear-5-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.1)]),
                          'gear-6-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.1)]),
                          'rear-differential-ratio':    np.array([scale_value(int("".join(str(x) for x in best_person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1)]),
                          'rear-spoiler-angle':         np.array([scale_value(int("".join(str(x) for x in best_person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0)]),
                          'front-spoiler-angle':        np.array([scale_value(int("".join(str(x) for x in best_person[7*n_bits:]), 2), n_bits, 90, 0)])}
                    
            # Perform the evaluation with the simulator and p2 not in dead_indexes
            observation, _, _, _ = env.step(parameters)
            
            # Display simulation results
            logger.info(' ')
            logger.info(' ')
            logger.info('##################################################')
            logger.info('Winner Results:')
            logger.info('Time elapsed (sec)      =   %f', maxEvaluationTime)
            logger.info('Top speed (km/h)        =   %f', observation['topspeed'][0])
            logger.info('Distance raced (m)      =   %f', observation['distRaced'][0])
            logger.info('Fuel used (l)           =   %f', observation['fuelUsed'][0])
            logger.info('Input parameters:')
            logger.info('gear-2-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[:n_bits]), 2), n_bits, 5, 0.1))
            logger.info('gear-3-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[n_bits:2*n_bits]), 2), n_bits, 5, 0.1))
            logger.info('gear-4-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.1))
            logger.info('gear-5-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.1))
            logger.info('gear-6-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.1))
            logger.info('rear-differential-ratio =   %f', scale_value(int("".join(str(x) for x in best_person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1))
            logger.info('rear-spoiler-angle      =   %f', scale_value(int("".join(str(x) for x in best_person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0))
            logger.info('front-spoiler-angle     =   %f', scale_value(int("".join(str(x) for x in best_person[7*n_bits:]), 2), n_bits, 90, 0))
            logger.info('##################################################')
                        
                        
                
    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()





"""
# Uncomment to use the default values in the TORCS simulator
   
parameters = {'gear-2-ratio': np.array([2.5]),
              'gear-3-ratio': np.array([1.5]),
              'gear-4-ratio': np.array([1.5]),
              'gear-5-ratio': np.array([1.5]),
              'gear-6-ratio': np.array([1.0]),
              'rear-differential-ratio': np.array([4.5]),
              'rear-spoiler-angle': np.array([14.0]),
              'front-spoiler-angle': np.array([6.0])}

parameters =   {'gear-2-ratio': np.array([1.7]), 
                'gear-3-ratio': np.array([1.6]), 
                'gear-4-ratio': np.array([2.8]), 
                'gear-5-ratio': np.array([0.1]), 
                'gear-6-ratio': np.array([0.3]), 
                'rear-differential-ratio': np.array([8.6]), 
                'rear-spoiler-angle': np.array([50.1]), 
                'front-spoiler-angle': np.array([1.5])}


# Uncomment to generate random values in the proper range for each variable
parameters = env.action_space.sample()

# Generate a random vector of parameters in the proper interval
logger.info('Generated new parameter vector: ' + str(parameters))
"""