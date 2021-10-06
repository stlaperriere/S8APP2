###############################################################################
# Université de Sherbrooke
# Génie informatique
# S8 - APP2 - A21
# Samuel Laperrière - laps2022
# Raphaël Lebrasseur - lebr2112
# Charles Murphy - murc3002
###############################################################################

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
import os
import sys
import time
import logging

import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomUniform

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################


class NeuralNet(object):

    def __init__(self, lr, momentum):
        
        self.lr = lr
        self.momentum = momentum
        self.model = Sequential()
        
        #hidden
        self.model.add(Dense(units=14, 
                             input_shape=(19,), 
                             activation='tanh',
                             kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)))
        
        #output
        self.model.add(Dense(units=4, activation='tanh'))
        
        print(self.model.summary())
        
        self.model.compile(optimizer=SGD(lr=self.lr,
                                         momentum=self.momentum), 
                                         loss='mse',
                                         metrics=['accuracy'])
        

    # entraine le réseau
    def _train(self, states, nb_of_epochs, filename='ann-drive.h5'):
    
        # the data is made of multiple states
        
        data = []
        target = []
        descaledTarget = []
        lastBrakeCmd = 0
        
        for s in states:
            data.append({'angle':s['angle'], 
                        'gear':s['gear'],
                        'rpm':s['rpm'],
                        'speed':s['speed'],
                        'trackPos':s['trackPos'],
                        'track':s['track'],
                        'wheelSpinVel':s['wheelSpinVel']
                        })
            brakeCmd = (0.5 * s['brakeCmd'][0]) + (0.5 * lastBrakeCmd)
            target.append([
                        s['accelCmd'][0],
                        brakeCmd,
                        s['gearCmd'][0],
                        s['steerCmd'][0]
                        ])
            lastBrakeCmd = brakeCmd
        
        for t in target:
            descaledTarget.append(self._descaleTarget(t))
        
        scaledData = []
        for d in data:
            scaledData.append(self._scaleData(d))
            
        history = self.model.fit(np.array(scaledData),
                       np.array(descaledTarget), 
                       batch_size=len(scaledData), 
                       epochs=nb_of_epochs, 
                       shuffle=True, verbose=0)
        
        plt.plot(history.history['acc'])
        plt.show()
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss and accuracy')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['acc', 'loss'], loc='upper left')
        plt.show()
        
        self.model.save(filename)


    # massage le state pour le fit à notre input
    def _scaleData(self, state):
        
        MAX_SPEED_X = 150
        MAX_SPEED_Y = 20
        MAX_RPM = 10000
        MAX_DIST = 100
        MAX_SPIN_VEL = 150
        # on normalize l'input de -1 à 1 pour une tanh
        angle = np.clip(state['angle'][0] / (2 * math.pi), -1, 1)
        speed_x = state['speed'][0] / MAX_SPEED_X
        speed_y = state['speed'][1] / MAX_SPEED_Y
        track_pos = np.clip(state['trackPos'][0], -1, 1)
        rpm = state['rpm'][0] / MAX_RPM
        gear = state['gear'][0] / 6
        track_0 = state['track'][2] / MAX_DIST
        track_1 = state['track'][3] / MAX_DIST
        track_2 = state['track'][7] / MAX_DIST
        track_3 = state['track'][8] / MAX_DIST
        track_4 = state['track'][9] / MAX_DIST
        track_5 = state['track'][18-8] / MAX_DIST
        track_6 = state['track'][18-7] / MAX_DIST
        track_7 = state['track'][18-3] / MAX_DIST
        track_8 = state['track'][18-2] / MAX_DIST
        wheel_0 = state['wheelSpinVel'][0] / MAX_SPIN_VEL
        wheel_1 = state['wheelSpinVel'][1] / MAX_SPIN_VEL        
        wheel_2 = state['wheelSpinVel'][2] / MAX_SPIN_VEL
        wheel_3 = state['wheelSpinVel'][3] / MAX_SPIN_VEL
        return [ angle, 
                speed_x, speed_y, 
                track_pos, 
                rpm, gear,
                track_0, track_1, 
                track_2, track_3, 
                track_4, track_5, 
                track_6, 
                track_7, 
                track_8,
                wheel_0, wheel_1, wheel_2, wheel_3
                ]
    
    
    def _descaleTarget(self, target):
        # on adapte à une tanh
        accel = target[0]
        brake = target[1]
        gear = target[2] / 6
        steer = np.clip(target[3], -1, 1)
        
        return [accel, brake, gear, steer]
    
    def _scaleTarget(self, data):
        # on adapte depuis une tanh
        accel = np.clip(data[0][0], 0, 1)
        brake = np.clip(data[0][1], 0, 1)
        gear = round(np.clip(data[0][2], 0, 1) * 5) + 1
        steer = np.clip(data[0][3], -1, 1)
        
        return {'accel':np.array([accel], dtype=np.float32),
                'brake':np.array([brake], dtype=np.float32), 
                'gear':np.array([gear], dtype=np.float32), 
                'steer':np.array([steer], dtype=np.float32)}


    def _load(self, filename='ann-drive.h5'):
        
        self.model = load_model(filename)
        

    # prédit selon l'état
    def _drive(self, state, reload=False):
        
        if reload:
            self._load()
            
        data = self._scaleData(state)
        prediction = self._scaleTarget(self.model.predict(np.array([data])))
        
        return prediction
        

def main():

    recordingsPath = os.path.join(CDIR, 'recordings')
    if not os.path.exists(recordingsPath):
        os.makedirs(recordingsPath)

    
    trainingPath = os.path.join(CDIR, 'training')
    trainingData = []
    for f in os.listdir(trainingPath):
        trainingFilename = os.path.join(trainingPath, f)
        episode = EpisodeRecorder.restore(trainingFilename)
        trainingData += episode.states
        
    
    ann = NeuralNet(0.2, 0.9)
    #ann._train(trainingData, 1000, 'ann-drive-simple-tweaks-input.h5');
    ann._load('ann-drive-simple-low-pass-single-layer-020-4000.h5')
            

    try:
        with TorcsControlEnv(render=True) as env:

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
                        action = ann._drive(observation)
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
