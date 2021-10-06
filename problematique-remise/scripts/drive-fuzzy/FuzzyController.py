###############################################################################
# Université de Sherbrooke
# Génie informatique
# S8 - APP2 - A21
# Samuel Laperrière - laps2022
# Raphaël Lebrasseur - lebr2112
# Charles Murphy - murc3002
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

class FuzzyController(object):
    
    def __init__(self):
        self.steerController = self.createFuzzyControllerSteer("sugeno")
        self.accelController = self.createAccelController()
        self.shiftController = self.createShiftController()
        
    def createFuzzyControllerSteer(self, style):
        # Create the fuzzy variables for inputs and outputs.
        # Defuzzification (defuzzify_method) methods for fuzzy variables:
        #    'centroid': Centroid of area
        #    'bisector': bisector of area
        #    'mom'     : mean of maximum
        #    'som'     : min of maximum
        #    'lom'     : max of maximum
        
        angle = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 1000), 'angle')
        trackPos = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'trackPos')
        
        # Methode de defuzzification choisie: centre de gravite
        # Permet de pondérer la sortie adéquatement selon les entrées, compte tenu
        # que la direction est une quantité précise et continue.
        steer = ctrl.Consequent(np.linspace(-1, 1, 1000), 'steer', defuzzify_method='centroid')
        
        steer.accumulation_method = np.fmax
        
        # Create membership functions
        angle['droite'] = fuzz.trapmf(angle.universe, [-np.pi, -np.pi, -0.1, 0])
        angle['centre'] = fuzz.trimf(angle.universe, [-0.1, 0, 0.1])
        angle['gauche'] = fuzz.trapmf(angle.universe, [0, 0.1, np.pi, np.pi])
        
        trackPos['gauche'] = fuzz.trapmf(trackPos.universe, [-1, -1, -0.2, -0.1])
        trackPos['centre'] = fuzz.trapmf(trackPos.universe, [-0.2, -0.1, 0.1, 0.2])
        trackPos['droite'] = fuzz.trapmf(trackPos.universe, [0.1, 0.2, 1, 1])
        
        # Mamdani
        if style == "mamdani":
            steer['gauche-toute'] = fuzz.trapmf(steer.universe, [-1, -1, -0.5, -0.3])
            steer['gauche'] = fuzz.trapmf(steer.universe, [-0.5, -0.3, -0.1, 0])
            steer['centre'] = fuzz.trimf(steer.universe, [-0.1, 0, 0.1])
            steer['droite'] = fuzz.trapmf(steer.universe, [0, 0.1, 0.3, 0.5])
            steer['droite-toute'] = fuzz.trapmf(steer.universe, [0.3, 0.5, 1, 1])
        
        # Sugeno
        if style == "sugeno":
            steer['gauche-toute'] = singletonmf(steer.universe, -1)
            steer['gauche'] = singletonmf(steer.universe, -0.25)
            steer['centre'] = singletonmf(steer.universe, 0)
            steer['droite'] = singletonmf(steer.universe, 0.25)
            steer['droite-toute'] = singletonmf(steer.universe, 1)
        
        # Regles
        # Utilisation de l'opérateur de conjonction "AND" pour toutes les prémisses. On ne pourrait pas utiliser un OR,
        # parce qu'il y a des moments où on n'est pas au centre, mais qu'on se dirige vers le centre. À ce moment là,
        # il faut se diriger droit devant. Si on utilisait un OR, on n'aurait pas cette distinction; dès qu'on serait
        # désaxé ou offsetté, on cramperait, ce qui n'est pas souhaitable.
        # Aussi, le fait d'utiliser un AND permet de faire la distinction entre plus de cas de figure : si on a une 
        # trackPos à gauche, il faut cramper vers la droite soit beaucoup, soit un peu ou pas du tout. C'est la
        # conjonction avec l'angle qui permet de nuancer l'intensité du crampage.
        rules = []
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
        
        print('------------------------ RULES ------------------------')
        for rule in sim.ctrl.rules:
            print(rule)
        print('-------------------------------------------------------')
    
        # Display fuzzy variables
        for var in sim.ctrl.fuzzy_variables:
            var.view()
        plt.show()
            
        return sim
    
    def calculateSteer(self, state):
        self.steerController.input['angle'] = state['angle'][0]
        self.steerController.input['trackPos'] = state['trackPos'][0]
        self.steerController.compute()
        #fuzzyController.print_state()
        return self.steerController.output['steer']
    
    def createAccelController(self) :
        track_angle = ctrl.Antecedent(np.linspace(0, np.pi/2, 1000), 'track_angle')
        speed = ctrl.Antecedent(np.linspace(0, 150, 1000), 'speed')
        
        accel = ctrl.Consequent(np.linspace(-1, 1, 1000), 'accel', defuzzify_method='centroid')
        
        accel.accumulation_method = np.fmax
        
        track_angle['straight'] = fuzz.trapmf(track_angle.universe, [0, 0, 0.4, 0.6])
        track_angle['med'] = fuzz.trapmf(track_angle.universe, [0.4, 0.6, np.pi/4, 3*np.pi/8])
        track_angle['sharp'] = fuzz.trapmf(track_angle.universe, [1, 1.2, np.pi/2, np.pi/2])
        
        speed['slow'] = fuzz.trapmf(speed.universe, [0, 0, 30, 50])
        speed['med'] = fuzz.trapmf(speed.universe, [40, 50, 80, 100])
        speed['fast'] = fuzz.trapmf(speed.universe, [80, 90, 150, 150])
        
        accel['arriere-toute'] = singletonmf(accel.universe, -1)
        accel['arriere'] = singletonmf(accel.universe, -0.25)
        accel['neutre'] = singletonmf(accel.universe, 0)
        accel['avant'] = singletonmf(accel.universe, 0.25)
        accel['avant-toute'] = singletonmf(accel.universe, 1)
        
        rules = []
        rules.append(ctrl.Rule(antecedent=(speed['slow'] & track_angle['straight']), consequent=accel['avant-toute']))
        rules.append(ctrl.Rule(antecedent=(speed['med'] & track_angle['straight']), consequent=accel['avant-toute']))
        rules.append(ctrl.Rule(antecedent=(speed['fast'] & track_angle['straight']), consequent=accel['avant']))
        rules.append(ctrl.Rule(antecedent=(speed['slow'] & track_angle['med']), consequent=accel['avant']))
        rules.append(ctrl.Rule(antecedent=(speed['med'] & track_angle['med']), consequent=accel['avant']))
        rules.append(ctrl.Rule(antecedent=(speed['fast'] & track_angle['med']), consequent=accel['avant']))
        rules.append(ctrl.Rule(antecedent=(speed['slow'] & track_angle['sharp']), consequent=accel['avant-toute']))
        rules.append(ctrl.Rule(antecedent=(speed['med'] & track_angle['sharp']), consequent=accel['avant-toute']))
        rules.append(ctrl.Rule(antecedent=(speed['fast'] & track_angle['sharp']), consequent=accel['arriere-toute']))
        
        # Conjunction (and_func) and disjunction (or_func) methods for rules:
        #     np.fmin
        #     np.fmax
        for rule in rules:
            rule.and_func = np.multiply
            rule.or_func = np.fmax
        
        system = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(system)
        
        print('------------------------ RULES ------------------------')
        for rule in sim.ctrl.rules:
            print(rule)
        print('-------------------------------------------------------')
    
        # Display fuzzy variables
        for var in sim.ctrl.fuzzy_variables:
            var.view()
        plt.show()
            
        return sim
    
        #rules.append(ctrl.Rule(antecedent=(left_sensor['med'] | right_sensor['med']), consequent=accel['arriere']))
    
    def calculateAccel(self, state):
        
        sin10 = 0.17365
        cos10 = 0.98481
        
        curTrackPos = state['trackPos'][0]
        
        # checks if car is out of track
        if (curTrackPos < 1 and curTrackPos > -1):

            # Reading of sensor at +10 degree w.r.t. car axis
            rxSensor = state['track'][8]
            # Reading of sensor parallel to car axis
            cSensor = state['track'][9]
            # Reading of sensor at -5 degree w.r.t. car axis
            sxSensor = state['track'][10]

            # Track is straight and enough far from a turn so goes to max speed
            if cSensor >= rxSensor and cSensor >= sxSensor:
                angle = 0
            else:
                # Approaching a turn on right
                if rxSensor > sxSensor:
                    # Computing approximately the "angle" of turn
                    h = cSensor * sin10
                    b = rxSensor - cSensor * cos10
                    angle = np.arcsin(b * b / (h * h + b * b))

                # Approaching a turn on left
                else:
                    # Computing approximately the "angle" of turn
                    h = cSensor * sin10
                    b = sxSensor - cSensor * cos10
                    angle = np.arcsin(b * b / (h * h + b * b))
            #print(angle)
            self.accelController.input['track_angle'] = angle
            self.accelController.input['speed'] = state['speed'][0]
            
            self.accelController.compute()
            result = self.accelController.output['accel']
        else:
            # when out of track returns a moderate acceleration command
            result = 0.3

        #print(f'accel : {result}')
        
        accel = 0
        brake = 0
        
        if result > 0:
            accel = result
        
        if result < 0:
            brake = -result
        
        return accel, brake
    
    def createShiftController(self):
        rpm = ctrl.Antecedent(np.linspace(0, 8000, 1000), 'rpm')
        
        shouldShift = ctrl.Consequent(np.linspace(-1, 1, 1000), 'shouldShift', defuzzify_method='centroid')
        
        shouldShift.accumulation_method = np.fmax
        
        rpm['low'] = fuzz.trapmf(rpm.universe, [0, 0, 4000, 4500])
        rpm['med'] = fuzz.trapmf(rpm.universe, [4000, 4500, 5500, 6000])
        rpm['high'] = fuzz.trapmf(rpm.universe, [5500, 6000, 8000, 8000])
        
        shouldShift['downshift'] = singletonmf(shouldShift.universe, -1)
        shouldShift['stay'] = singletonmf(shouldShift.universe, 0)
        shouldShift['upshift'] = singletonmf(shouldShift.universe, 1)
        
        rules = []
        rules.append(ctrl.Rule(antecedent=(rpm['low']), consequent=shouldShift['downshift']))
        rules.append(ctrl.Rule(antecedent=(rpm['med']), consequent=shouldShift['stay']))
        rules.append(ctrl.Rule(antecedent=(rpm['high']), consequent=shouldShift['upshift']))
        
        # Conjunction (and_func) and disjunction (or_func) methods for rules:
        #     np.fmin
        #     np.fmax
        for rule in rules:
            rule.and_func = np.multiply
            rule.or_func = np.fmax
        
        system = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(system)
        
        print('------------------------ RULES ------------------------')
        for rule in sim.ctrl.rules:
            print(rule)
        print('-------------------------------------------------------')
    
        # Display fuzzy variables
        for var in sim.ctrl.fuzzy_variables:
            var.view()
        plt.show()
            
        return sim
    
    def calculateGear(self, state):
        #return 1
        #print(state['rpm'])
        self.shiftController.input['rpm'] = state['rpm']
        self.shiftController.compute()
        shouldShift = self.shiftController.output['shouldShift']
        
        #print(shouldShift)
        
        nextGear = state['gear'][0]
        
        if nextGear == 0: return nextGear + 1
        
        if shouldShift[0] > 0.5 and state['gear'][0] < 6:
            nextGear = nextGear + 1
        elif shouldShift[0] < -0.5 and state['gear'][0] > 1:
            nextGear = nextGear - 1
        
        #print(nextGear)
        return nextGear