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
import matplotlib.pyplot as plt

sys.path.append('../..')
from torcs.optim.core import TorcsOptimizationEnv, TorcsException

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)



################################
# Define hyperparameters here
################################
mutation_rate = 0.025 # Mutation rate to determine if mutation ocures
population_size = 75
boundaries = [[0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [1, 10], [0, 90], [0, 90]]
n_bits = 8
n_gen = 100
n_dead = 15
n_bits_muta = 3
fitness_name = "Speed (km/h)" # "Speed (km/h)" or "Fuel Consumption (km/L)"
################################
# Define helper functions here
################################
def fitness_function(speed, dist, fuel, fitness_name):
    # Return the speed as the fitness value (we want to maximise top speed)
    if fitness_name == "Speed (km/h)":
        return speed
    # Return the fuel economy as the fitness value (we want to maximise fuel economy)
    elif fitness_name == "Fuel Consumption (km/L)":
        return dist / (fuel * 1000)


# Mutation function where we flip a random number of bits
def mutation(p, n_bits_muta, n_bits):
    for i in range(n_bits_muta):
        index = np.random.randint(0, 8 * n_bits)
        p[index] = 1 - p[index]
    return p

def reproduction(p, n_bits):
    genes = [p[0][:n_bits],
             p[1][n_bits:2*n_bits],
             p[2][2*n_bits:3*n_bits],
             p[3][3*n_bits:4*n_bits],
             p[4][4*n_bits:5*n_bits],
             p[5][5*n_bits:6*n_bits],
             p[6][6*n_bits:7*n_bits],
             p[7][7*n_bits:]]
    return np.concatenate((genes))

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
            
            best_person, best_score = 0, fitness_function(-1, -1, -1, fitness_name)
            current_best_person, current_best_score = 0, fitness_function(-1, -1, -1, fitness_name)
            
            # Data for plot
            avg_fitness = []
            best_fitness = []
            
            # Loop a few times for demonstration purpose
            for i in range(n_gen):
                logger.info("Generation : %d", i)
                scores = []
                for j, person in enumerate(population):
                    parameters = {'gear-2-ratio':               np.array([scale_value(int("".join(str(x) for x in person[:n_bits]), 2), n_bits, 5, 0.101)]),
                                  'gear-3-ratio':               np.array([scale_value(int("".join(str(x) for x in person[n_bits:2*n_bits]), 2), n_bits, 5, 0.101)]),
                                  'gear-4-ratio':               np.array([scale_value(int("".join(str(x) for x in person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.101)]),
                                  'gear-5-ratio':               np.array([scale_value(int("".join(str(x) for x in person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.101)]),
                                  'gear-6-ratio':               np.array([scale_value(int("".join(str(x) for x in person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.101)]),
                                  'rear-differential-ratio':    np.array([scale_value(int("".join(str(x) for x in person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1)]),
                                  'rear-spoiler-angle':         np.array([scale_value(int("".join(str(x) for x in person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0)]),
                                  'front-spoiler-angle':        np.array([scale_value(int("".join(str(x) for x in person[7*n_bits:]), 2), n_bits, 90, 0)])}
                    
                    # Perform the evaluation with the simulator and p2 not in dead_indexes
                    observation, _, _, _ = env.step(parameters)
                    
                    # Calcul du score
                    scores.append(fitness_function(observation['topspeed'][0], observation['distRaced'][0], observation['fuelUsed'][0], fitness_name))
                  
                    
                    # Find all time best person
                    
                    if scores[j] > best_score:
                        best_person, best_score = population[j], scores[j]
                        logger.info("New all time best = %f", best_score)

                    
                    
                    
                
                if i < n_gen:
                    # Survivor selection
                    dead_indexes = select_k_lowest(scores, n_dead)
                    
                    # Create a child for each dead person
                    for j in range(n_dead):
                        # Select 2 parents that survived
                        
                        parents = [population[select_parent(scores, dead_indexes)] for x in range(8)]
                        # Create a new child
                        child = reproduction(parents, n_bits)
                        
                        # Replace a dead person with the new child
                        population[dead_indexes[j]] = child
                        
                # Probability of mutation of the person
                for j, person in enumerate(population):
                    if np.random.random() < mutation_rate:
                        population[j] = mutation(person, n_bits_muta, n_bits)
                    
                # Get the current best

                best_i = np.argmax(scores)
                current_best_person, current_best_score = population[best_i], scores[best_i]
                logger.info("Best player in the current gen = %f", current_best_score)
                
                
            
                avg_fitness.append(np.mean(np.array(scores)))
                best_fitness.append(current_best_score)
                
            plt.plot(avg_fitness, color='blue', label='Average') 
            plt.plot(best_fitness, color='orange', label='Best') 
            plt.ylabel(fitness_name)
            plt.xlabel("Générations")
            plt.title("Best And Average " + fitness_name + " Over Generations")
            plt.show()
            
            # SIMULATE THE CURRENT BEST PLAYER #############################################################################
            parameters = {'gear-2-ratio':               np.array([scale_value(int("".join(str(x) for x in current_best_person[:n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-3-ratio':               np.array([scale_value(int("".join(str(x) for x in current_best_person[n_bits:2*n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-4-ratio':               np.array([scale_value(int("".join(str(x) for x in current_best_person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-5-ratio':               np.array([scale_value(int("".join(str(x) for x in current_best_person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-6-ratio':               np.array([scale_value(int("".join(str(x) for x in current_best_person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.101)]),
                          'rear-differential-ratio':    np.array([scale_value(int("".join(str(x) for x in current_best_person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1)]),
                          'rear-spoiler-angle':         np.array([scale_value(int("".join(str(x) for x in current_best_person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0)]),
                          'front-spoiler-angle':        np.array([scale_value(int("".join(str(x) for x in current_best_person[7*n_bits:]), 2), n_bits, 90, 0)])}

            # Perform the evaluation with the simulator and p2 not in dead_indexes
            observation, _, _, _ = env.step(parameters)
            
            # Display simulation results
            logger.info(' ')
            logger.info(' ')
            logger.info('##################################################')
            logger.info('Best Results at Last Generation:')
            logger.info('Time elapsed (sec)      =   %f', maxEvaluationTime)
            logger.info('Top speed (km/h)        =   %f', observation['topspeed'][0])
            logger.info('Distance raced (m)      =   %f', observation['distRaced'][0])
            logger.info('Fuel used (l)           =   %f', observation['fuelUsed'][0])
            logger.info('Fuel consumption (km/l) =   %f', observation['distRaced'][0] / (observation['fuelUsed'][0] * 1000))
            logger.info('Input parameters:')
            logger.info('gear-2-ratio            =   %f', scale_value(int("".join(str(x) for x in current_best_person[:n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-3-ratio            =   %f', scale_value(int("".join(str(x) for x in current_best_person[n_bits:2*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-4-ratio            =   %f', scale_value(int("".join(str(x) for x in current_best_person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-5-ratio            =   %f', scale_value(int("".join(str(x) for x in current_best_person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-6-ratio            =   %f', scale_value(int("".join(str(x) for x in current_best_person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('rear-differential-ratio =   %f', scale_value(int("".join(str(x) for x in current_best_person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1))
            logger.info('rear-spoiler-angle      =   %f', scale_value(int("".join(str(x) for x in current_best_person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0))
            logger.info('front-spoiler-angle     =   %f', scale_value(int("".join(str(x) for x in current_best_person[7*n_bits:]), 2), n_bits, 90, 0))
            logger.info('##################################################')
                        
            # SIMULATE THE ALL TIME BEST PLAYER #############################################################################
            parameters = {'gear-2-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[:n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-3-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[n_bits:2*n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-4-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-5-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.101)]),
                          'gear-6-ratio':               np.array([scale_value(int("".join(str(x) for x in best_person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.101)]),
                          'rear-differential-ratio':    np.array([scale_value(int("".join(str(x) for x in best_person[5*n_bits:6*n_bits]), 2), n_bits, 10, 1)]),
                          'rear-spoiler-angle':         np.array([scale_value(int("".join(str(x) for x in best_person[6*n_bits:7*n_bits]), 2), n_bits, 90, 0)]),
                          'front-spoiler-angle':        np.array([scale_value(int("".join(str(x) for x in best_person[7*n_bits:]), 2), n_bits, 90, 0)])}

            # Perform the evaluation with the simulator and p2 not in dead_indexes
            observation, _, _, _ = env.step(parameters)
            
            # Display simulation results
            logger.info(' ')
            logger.info(' ')
            logger.info('##################################################')
            logger.info('All Time Best Results:')
            logger.info('Time elapsed (sec)      =   %f', maxEvaluationTime)
            logger.info('Top speed (km/h)        =   %f', observation['topspeed'][0])
            logger.info('Distance raced (m)      =   %f', observation['distRaced'][0])
            logger.info('Fuel used (l)           =   %f', observation['fuelUsed'][0])
            logger.info('Fuel consumption (km/l) =   %f', observation['distRaced'][0] / (observation['fuelUsed'][0] * 1000))
            logger.info('Input parameters:')
            logger.info('gear-2-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[:n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-3-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[n_bits:2*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-4-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[2*n_bits:3*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-5-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[3*n_bits:4*n_bits]), 2), n_bits, 5, 0.101))
            logger.info('gear-6-ratio            =   %f', scale_value(int("".join(str(x) for x in best_person[4*n_bits:5*n_bits]), 2), n_bits, 5, 0.101))
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

