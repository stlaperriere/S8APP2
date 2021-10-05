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
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import sys
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

def speed_gear_rpm(episode):
    plt.figure()
    plt.subplot(311)
    plt.plot(episode.speed[:, 0])
    plt.title('Car speed (longitudinal axis, X)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Speed [km/h]')
    plt.subplot(312)
    plt.plot(episode.t, episode.gear)
    plt.title('Tranmission gear')
    plt.xlabel('Time [sec]')
    plt.ylabel('Gear')
    plt.subplot(313)
    plt.plot(episode.t, episode.rpm)
    plt.title('Engine rpm')
    plt.xlabel('Time [sec]')
    plt.ylabel('RPM')
    plt.tight_layout()

def histogram_speedX(episode):
    speedX = []
    for item in episode.speed:
        speedX.append(item[0])
        
    plt.figure()
    plt.hist(speedX, label="speedX", bins=20)
    plt.legend()
    plt.title("Histogramme - Speed X - Alpine 1 - Drive Simple")
    plt.show()

def track(episode):
    track_left = []
    track_centre = []
    track_right = []
    
    for item in episode.track:
        track_left.append(item[8])
        track_centre.append(item[9])
        track_right.append(item[10])
        
    speedX = []
    for item in episode.speed:
        speedX.append(item[0])
    
    #plt.plot(track_left, label = "track_left")
    #plt.plot(track_centre, label = "track_centre")
    #plt.plot(track_right, label = "track_right")
    plt.figure()
    plt.plot(episode.steerCmd * 100, label = "steerCmd x 100")
    #plt.plot(episode.accelCmd * 100, label = "accelCmd")
    plt.plot(speedX, label = "speedX")
    plt.title("Vitesse dans les courbes - Drive Simple (accel) et Logique floue (steer)", fontsize = 18)
    plt.legend(fontsize=20)
    plt.show()
    

def track_pos_angle_steer(episode):
    plt.figure()
    plt.plot(episode.trackPos, label="trackPos")
    plt.plot(episode.angle, label = "angle")
    plt.plot(episode.steerCmd, label = "steerCmd")
    plt.legend()
    plt.title("TrackPos, Angle et SteerCmd - Logique floue (Steer) - Alpine 1")
    #plt.hist(episode.angle, bins=20)
    plt.show()
    #plt.hist(episode.trackPos)
    
def gear(episode):
    gear = []
    
    for item in episode.gear:
        gear.append(item[0])
    
    plt.figure()
    plt.plot(gear, label="Gear")
    plt.title("Vitesse  - Drive Bot (gear) - Alpine 1")

def comp_gear(episode_fuzzy, episode_simple, episode_bot):
    plt.figure()
    plt.title("Gear  - Logique floue (gear) - Alpine 1")
    plt.plot(episode_fuzzy.gear[:, 0])
    
    plt.figure()
    plt.title("Gear  - Drive Simple (gear) - Alpine 1")
    plt.plot(episode_simple.gear[:, 0])
    
    plt.figure()
    plt.title("Gear  - Drive Bot (gear) - Alpine 1")
    plt.plot(episode_bot.gear[:, 0])
    
    
def comp_steer(episode_fuzzy, episode_simple):
    plt.figure()
    plt.plot(episode_simple.steerCmd, label = "Simple")
    plt.plot(episode_fuzzy.steerCmd, label = "Fuzzy")
    
    plt.title("Comparatif - Floue vs Simple - Steer - Alpine 1")
    plt.legend()
    
def comp_accel(episode_fuzzy, episode_simple):
    
    speedX_fuzz = []
    for item in episode_fuzzy.speed:
        speedX_fuzz.append(item[0])
    
    speedX_simple = []
    for item in episode_simple.speed:
        speedX_simple.append(item[0])
        
    print(f'Vitesse moyenne - simple : {np.mean(speedX_simple)}')
    print(f'Vitesse moyenne - fuzzy : {np.mean(speedX_fuzz)}')
        
    
    #plt.plot(-episode_simple.brakeCmd, label = "brake - Simple")
    plt.figure()
    plt.plot(speedX_fuzz, label = "speed - Floue")
    #plt.plot(-episode_fuzzy.brakeCmd, label = "brake - Floue")
    plt.plot(speedX_simple, label = "speed - Simple")
    plt.title("Comparatif - Floue vs Simple - Vitesse - Alpine 1")
    plt.legend()
    
    
    
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    recordingFilename = ('./recordings/track-alpine-1.pklz')
    episode_fuzzy = EpisodeRecorder.restore(recordingFilename)
    
    recordingFilename = ('./recordings/track-alpine-1-steer-only.pklz')
    episode_fuzzy_steer_only = EpisodeRecorder.restore(recordingFilename)
    
    recordingFilename = ('../drive-simple/recordings/track-alpine-1.pklz')
    episode_simple = EpisodeRecorder.restore(recordingFilename)
    
    recordingFilename = ('../drive-simple/recordings/track-aalborg.pklz')
    episode_simple_aalborg = EpisodeRecorder.restore(recordingFilename)
    
    recordingFilename = ('../drive-bot/recordings/track-alpine-1.pklz')
    episode_bot = EpisodeRecorder.restore(recordingFilename)
    
    # Affichage des graphiques
    
    histogram_speedX(episode_simple)
    speed_gear_rpm(episode_simple_aalborg)
    #track(episode_fuzzy)
    #track_pos_angle_steer(episode)
    #gear(episode)
    
    comp_steer(episode_fuzzy_steer_only, episode_simple)
    comp_accel(episode_fuzzy, episode_simple)
    comp_gear(episode_fuzzy, episode_simple, episode_bot)
    
    #episode_simple.visualize(showObservations=True, showActions=True)
    #plt.show()
