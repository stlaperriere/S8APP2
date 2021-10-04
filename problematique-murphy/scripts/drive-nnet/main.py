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

import math
import numpy as np
import scipy.io

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

"""def drive(self, state):
    accel, brake = self._calculateAcceleration(state)
    gear = self._calculateGear(state)
    steer = self._calculateSteering(state)
    
    action = {'accel': np.array([accel], dtype=np.float32),
              'brake': np.array([brake], dtype=np.float32),
              'gear': np.array([gear], dtype=np.int32),
              'steer': np.array([steer], dtype=np.float32)}
    return action
"""

"""
Define le gear, brake/acc et steer

"""

class NeuralNet(object):

    def __init__(self, lr, momentum):
        
        self.lr = lr
        self.momentum = momentum
        self.model = Sequential()
        
        #input
        self.model.add(Dense(units=4, input_shape=(6,), activation='tanh'))
        #hidden
        self.model.add(Dense(units=4, activation='relu'))
        #output
        self.model.add(Dense(units=4, activation='sigmoid'))
        
        print(self.model.summary())
        
        self.model.compile(optimizer=SGD(lr=self.lr,momentum=self.momentum), 
                           loss='mse')
        

    # entraine le réseau
    def _train(self, states, nb_of_epochs):
    
        # the data is made of multiple states
        
        data = []
        target = []
        descaledTarget = []
        
        for s in states:
            data.append({'angle':s['angle'], 
                        #s['curlLapTime'],
                        #s['damage'],
                        #s['distFromStart'],
                        #s['fuel'],
                        'gear':s['gear'],
                        'rpm':s['rpm'],
                        'speed':s['speed'],
                        'trackPos':s['trackPos']
                        #s...
                        })
            target.append([
                        s['accelCmd'][0],
                        s['brakeCmd'][0],
                        s['gearCmd'][0],
                        s['steerCmd'][0]
                        ])
        
        for t in target:
            descaledTarget.append(self._descaleTarget(t))
        
        scaledData = []
        for d in data:
            scaledData.append(self._scaleData(d))
            
        self.model.fit(np.array(scaledData), np.array(descaledTarget), batch_size=len(scaledData), 
                       epochs=nb_of_epochs, shuffle=True, verbose=1)
        
        self.model.save('ann-drive.h5')


    # massage le state pour le fit à notre input
    def _scaleData(self, state):
            
        # on normalize l'input de -1 à 1 pour une tanh
        angle = (state['angle'][0] / (2 * math.pi))
        speed_x = (2 * state['speed'][0] / 10) - 1
        speed_y = (2 * state['speed'][1] / 100) - 1
        track_pos = state['trackPos'][0]
        rpm = (2 * state['rpm'][0]  / 10000) - 1
        gear = (2 * (state['gear'] + 1) / 7) - 1
        return [ angle, speed_x, speed_y, track_pos, rpm, gear]
    
    
    def _descaleTarget(self, target):
        accel = np.clip(target[0], 0, 1)
        brake = np.clip(target[1], 0, 1)
        gear = np.clip((target[2] + 1) / 7, 0, 1)
        steer = np.clip((target[3] + 1) / 2, 0, 1)
        
        return [accel, brake, gear, steer]
    
    def _scaleTarget(self, data):
        
        accel = np.clip(data[0][0], 0, 1)
        brake = np.clip(data[0][1], 0, 1)
        gear = math.ceil((np.clip(data[0][2], 0, 1) * 7) - 1)
        steer = np.clip((data[0][3] * 2) - 1, -1, 1)
        
        return {'accel':np.array([accel], dtype=np.float32),
                'brake':np.array([brake], dtype=np.float32), 
                'gear':np.array([gear], dtype=np.float32), 
                'steer':np.array([steer], dtype=np.float32)}


    def _load(self):
        
        self.model = load_model('ann-drive.h5')
        

    # prédit selon l'état
    def _drive(self, state, reload=False):
        
        if reload:
            self._load()
            
        data = self._scaleData(state)
        prediction = self._scaleTarget(self.model.predict(np.array([data])))
        
        return prediction
        """
        action = {'accel': np.array([prediction[0]], dtype=np.float32),
              'brake': np.array([prediction[1]], dtype=np.float32),
              'gear': np.array([prediction[2]], dtype=np.int32),
              'steer': np.array([prediction[3]], dtype=np.float32)}
        return action"""
        

def main():

    recordingsPath = os.path.join(CDIR, 'recordings')
    trainingPath = os.path.join(CDIR, 'training')
    if not os.path.exists(recordingsPath):
        os.makedirs(recordingsPath)

    
    # test with one track
    # TODO: train with all tracks
    trainingFilename = os.path.join(trainingPath, 'track-alpine-1.pklz')
    episode = EpisodeRecorder.restore(trainingFilename)
    
    ann = NeuralNet(0.3, 0.9)
    ann._train(episode.states, 500);
    ann._load()
            

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
                        action = ann._drive(observation)
                        #action = env.action_space.sample()
                        #print(action)
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
