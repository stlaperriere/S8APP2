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

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

def histogram_speedX(episode):
    speedX = []
    for item in episode.speed:
        speedX.append(item[0])
        
    plt.hist(speedX)

def track(episode):
    track_left = []
    track_centre = []
    track_right = []
    
    for item in episode.track:
        track_left.append(item[8])
        track_centre.append(item[9])
        track_right.append(item[10])
    
    plt.plot(track_left)
    plt.plot(track_centre)
    plt.plot(track_right)

def track_pos_angle_steer(episode):
    plt.plot(episode.trackPos, label="trackPos")
    plt.plot(episode.angle, label = "angle")
    plt.plot(episode.steerCmd, label = "steerCmd")
    plt.legend()
    plt.title("TrackPos, Angle et SteerCmd - Logique floue (Steer) - Alpine 1")
    #plt.hist(episode.angle, bins=20)
    plt.show()
    #plt.hist(episode.trackPos)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #recordingFilename = os.path.join(CDIR, 'recordings', 'track.pklz')
    recordingFilename = ('/media/sf_S8APP2/problematique-sam/scripts/drive-fuzzy/recordings/track-aalborg.pklz')
    episode = EpisodeRecorder.restore(recordingFilename)
    
    #histogram_speedX(episode)
    #track(episode)
    track_pos_angle_steer(episode)
    
    #episode.visualize(showObservations=True, showActions=True)
    #plt.show()
