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
        
        #print('------------------------ RULES ------------------------')
        #for rule in sim.ctrl.rules:
        #    print(rule)
        #print('-------------------------------------------------------')
    
        # Display fuzzy variables
        #for var in sim.ctrl.fuzzy_variables:
        #    var.view()
        #plt.show()
            
        return sim
    
    def calculateSteer(self, state):
        self.steerController.input['angle'] = state['angle'][0]
        self.steerController.input['trackPos'] = state['trackPos'][0]
        self.steerController.compute()
        #fuzzyController.print_state()
        return self.steerController.output['steer']
    
    def createAccelController(self) :
        left_sensor = ctrl.Antecedent(np.linspace(0, 100, 1000), 'left_sensor')
        right_sensor = ctrl.Antecedent(np.linspace(0, 100, 1000), 'right_sensor')
        centre_sensor = ctrl.Antecedent(np.linspace(0, 100, 1000), 'centre_sensor')
        
        accel = ctrl.Consequent(np.linspace(-1, 1, 1000), 'accel', defuzzify_method='centroid')
        
        accel.accumulation_method = np.fmax
        
        #left_sensor['near'] = fuzz.trapmf(left_sensor.universe, [0, 0, 25, 30])
        #left_sensor['med'] = fuzz.trapmf(left_sensor.universe, [25, 30, 60, 70])
        left_sensor['far'] = fuzz.trapmf(left_sensor.universe, [30, 35, 100, 100])
        
        centre_sensor['near'] = fuzz.trapmf(centre_sensor.universe, [0, 0, 25, 30])
        centre_sensor['med'] = fuzz.trapmf(centre_sensor.universe, [25, 30, 60, 70])
        centre_sensor['far'] = fuzz.trapmf(centre_sensor.universe, [60, 70, 100, 100])
        
        #right_sensor['near'] = fuzz.trapmf(right_sensor.universe, [0, 0, 25, 30])
        #right_sensor['med'] = fuzz.trapmf(right_sensor.universe, [25, 30, 60, 70])
        right_sensor['far'] = fuzz.trapmf(right_sensor.universe, [30, 35, 100, 100])
        
        accel['arriere-toute'] = singletonmf(accel.universe, -1)
        accel['arriere'] = singletonmf(accel.universe, -0.5)
        accel['neutre'] = singletonmf(accel.universe, 0)
        accel['avant'] = singletonmf(accel.universe, 0.5)
        accel['avant-toute'] = singletonmf(accel.universe, 1)
        
        rules = []
        rules.append(ctrl.Rule(antecedent=(centre_sensor['far'] | left_sensor['far'] | right_sensor['far']), consequent=accel['avant-toute']))
        rules.append(ctrl.Rule(antecedent=(centre_sensor['med']), consequent=accel['neutre']))
        rules.append(ctrl.Rule(antecedent=(centre_sensor['near']), consequent=accel['arriere-toute']))
        
        
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
        
        curSpeedX = state['speed'][0]
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

        else:
            # when out of track returns a moderate acceleration command
            accel = 0.3

        
        self.accelController.input['left_sensor'] = state['track'][8]
        self.accelController.input['centre_sensor'] = state['track'][9]
        self.accelController.input['right_sensor'] = state['track'][10]
        self.accelController.compute()
        result = self.accelController.output['accel']
        
        #print(f'accel : {result}')
        
        accel = 0
        brake = 0
        
        if result > 0:
            accel = result
        
        if result < 0:
            brake = -result
        
        return accel, brake
        