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
import logging
from SimpleController import SimpleController

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

from skfuzzy import control as ctrl
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

def singletonmf(x, a):
    """
    Singleton membership function generator.
    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : constant
    Returns
    -------
    y : 1d array
        Singleton membership function.
    """
    y = np.zeros(len(x))

    if a >= np.min(x) and a <= np.max(x):
        idx = (np.abs(x - a)).argmin()
        y[idx] = 1.0

    return y

def createFuzzyController():
    # Create the fuzzy variables for inputs and outputs.
    # Defuzzification (defuzzify_method) methods for fuzzy variables:
    #    'centroid': Centroid of area
    #    'bisector': bisector of area
    #    'mom'     : mean of maximum
    #    'som'     : min of maximum
    #    'lom'     : max of maximum
    
    # poleAngle = ctrl.Antecedent(np.linspace(-180, 180, 1000), 'pole-angle')
    # poleVelocity = ctrl.Antecedent(np.linspace(-100, 100, 1000), 'pole-velocity')
    # force = ctrl.Consequent(np.linspace(-50, 50, 1000), 'force', defuzzify_method='centroid')
    
    angle = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 1000), 'angle')
    trackPos = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'trackPos')
    steer = ctrl.Consequent(np.linspace(-1, 1, 1000), 'steer', defuzzify_method='centroid')
    
    steer.accumulation_method = np.fmax
    
    # Create membership functions
    #poleAngle['negative'] = fuzz.trapmf(poleAngle.universe, [-181, -180, -15, 0])
    #poleAngle['null'] = fuzz.trapmf(poleAngle.universe, [-15, -5, 5, 15])
    #poleAngle['positive'] = fuzz.trapmf(poleAngle.universe, [0, 15, 180, 181])
    angle['droite'] = fuzz.trapmf(angle.universe, [-np.pi, -np.pi, -0.1, 0])
    angle['centre'] = fuzz.trimf(angle.universe, [-0.1, 0, 0.1])
    angle['gauche'] = fuzz.trapmf(angle.universe, [0, 0.1, np.pi, np.pi])
    
    trackPos['gauche'] = fuzz.trapmf(trackPos.universe, [-1, -1, -0.2, -0.1])
    trackPos['centre'] = fuzz.trapmf(trackPos.universe, [-0.2, -0.1, 0.1, 0.2])
    trackPos['droite'] = fuzz.trapmf(trackPos.universe, [0.1, 0.2, 1, 1])
    
    steer['gauche-toute'] = fuzz.trapmf(steer.universe, [-1, -1, -0.5, -0.3])
    steer['gauche'] = fuzz.trapmf(steer.universe, [-0.5, -0.3, -0.1, 0])
    steer['centre'] = fuzz.trimf(steer.universe, [-0.1, 0, 0.1])
    steer['droite'] = fuzz.trapmf(steer.universe, [0, 0.1, 0.3, 0.5])
    steer['droite-toute'] = fuzz.trapmf(steer.universe, [0.3, 0.5, 1, 1])
    
    rules = []
    #rules.append(ctrl.Rule(antecedent=(poleAngle['negative'] & poleVelocity['negative']), consequent=force['negative']))
    rules.append(ctrl.Rule(antecedent=(angle['gauche'] & trackPos['gauche']), consequent=steer['droite-toute']))
    rules.append(ctrl.Rule(antecedent=(angle['centre'] & trackPos['gauche']), consequent=steer['droite']))
    rules.append(ctrl.Rule(antecedent=(angle['droite'] & trackPos['gauche']), consequent=steer['centre']))
    rules.append(ctrl.Rule(antecedent=(angle['gauche'] & trackPos['centre']), consequent=steer['droite']))
    rules.append(ctrl.Rule(antecedent=(angle['centre'] & trackPos['centre']), consequent=steer['centre']))
    rules.append(ctrl.Rule(antecedent=(angle['droite'] & trackPos['centre']), consequent=steer['gauche']))
    rules.append(ctrl.Rule(antecedent=(angle['gauche'] & trackPos['droite']), consequent=steer['centre']))
    rules.append(ctrl.Rule(antecedent=(angle['centre'] & trackPos['droite']), consequent=steer['gauche']))
    rules.append(ctrl.Rule(antecedent=(angle['droite'] & trackPos['droite']), consequent=steer['gauche-toute']))
    
    # Conjunction (and_func) and disjunction (or_func) methods for rules:
    #     np.fmin
    #     np.fmax
    for rule in rules:
        rule.and_func = np.multiply
        rule.or_func = np.fmax
    
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
        
    return sim

def calculateSteer(fuzzyController, state):
    fuzzyController.input['angle'] = state['angle'][0]
    fuzzyController.input['trackPos'] = state['trackPos'][0]
    fuzzyController.compute()
    #fuzzyController.print_state()
    return fuzzyController.output['steer']

def drive(simpleController, fuzzyController, state):
    accel, brake = simpleController._calculateAcceleration(state)
    gear = simpleController._calculateGear(state)
    #steer = simpleController._calculateSteering(state)
    steer = calculateSteer(fuzzyController, state)

    action = {'accel': np.array([accel], dtype=np.float32),
              'brake': np.array([brake], dtype=np.float32),
              'gear': np.array([gear], dtype=np.int32),
              'steer': np.array([steer], dtype=np.float32)}
    return action

def main():
    # Creation du controleur de logique floue (steer)
    sim = createFuzzyController()
    
    # Creation du controleur simple
    simpleController = SimpleController()
    
    print('------------------------ RULES ------------------------')
    for rule in sim.ctrl.rules:
        print(rule)
    print('-------------------------------------------------------')
    
    # Display fuzzy variables
    for var in sim.ctrl.fuzzy_variables:
        var.view()
    plt.show()
    
    recordingsPath = os.path.join(CDIR, 'recordings')
    if not os.path.exists(recordingsPath):
        os.makedirs(recordingsPath)

    try:
        with TorcsControlEnv(render=False) as env:

            nbTracks = len(TorcsControlEnv.availableTracks)
            nbSuccessfulEpisodes = 0
            
            for episode in range(nbTracks):
                logger.info('Episode no.%d (out of %d)' % (episode + 1, nbTracks))
                startTime = time.time()

                observation = env.reset()
                trackName = env.getTrackName()

                nbStepsShowStats = 1000
                curNbSteps = 0
                done = False
                with EpisodeRecorder(os.path.join(recordingsPath, 'track-%s.pklz' % (trackName))) as recorder:
                    while not done:
                        # TODO: Select the next action based on the observation
                        # action = env.action_space.sample()
                        action = drive(simpleController, sim, observation)
                        recorder.save(observation, action)
    
                        # Execute the action
                        observation, reward, done, _ = env.step(action)
                        curNbSteps += 1
    
                        if observation and curNbSteps % nbStepsShowStats == 0:
                            curLapTime = observation['curLapTime'][0]
                            distRaced = observation['distRaced'][0]
                            logger.info('Current lap time = %4.1f sec (distance raced = %0.1f m)' % (curLapTime, distRaced))
    
                        if done:
                            if reward > 0.0:
                                logger.info('Episode was successful.')
                                nbSuccessfulEpisodes += 1
                            else:
                                logger.info('Episode was a failure.')
    
                            elapsedTime = time.time() - startTime
                            logger.info('Episode completed in %0.1f sec (computation time).' % (elapsedTime))

            logger.info('-----------------------------------------------------------')
            logger.info('Total number of successful tracks: %d (out of %d)' % (nbSuccessfulEpisodes, nbTracks))
            logger.info('-----------------------------------------------------------')

    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':    
    logging.basicConfig(level=logging.INFO)
    main()
