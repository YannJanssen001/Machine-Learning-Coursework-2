# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

#Importing the necessary libraries
from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from collections import defaultdict
import numpy as np


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        #Getting all of Pacman's legal actions that can be performed in the current state
        self.legalActions = state.getLegalPacmanActions()
        #Remove the STOP action from the legal actions
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)
        #Get the current position of Pacman
        self.position = state.getPacmanPosition()
        #Get the current state of the food in Pacman's game
        self.food = state.getFood()
        #Get the current position of the first ghost
        self.ghost = state.getGhostPosition(1)

    #The __eq__ method is used to check whether two objects are equal or not
    def __eq__(self, value: object) -> bool:
        #Checks if the given object is an instance of the GameStateFeatures class, and if it has
        #the position, food and ghost states as the current object. 
        return isinstance(value, GameStateFeatures) and self.position == value.position and \
              self.food == value.food and self.ghost == value.ghost

    #The __hash__ method is used to return an integer values that represents the object
    def __hash__(self) -> int:
        #Used when the object is used as a key in a dictionary, and here it is used to return
        #a hash value of a tuple that has the position, food and ghost states of the object.
        return hash((self.position, self.food, self.ghost))

class QLearnAgent(Agent):
    #Initializing the QLearning Agent, with given hyperparameters below.
    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """

        #Calls the parent class's constructor
        super().__init__()
        #alpha is the learning rate, (how much to update the Q value in each iteration) 
        self.alpha = float(alpha)
        #epsilon is the exploration rate,
        #(how much to explore new actions, instead of exploiting the best action)
        self.epsilon = float(epsilon)
        #gamma is known as the discount factor, (metric to value future rewards)
        self.gamma = float(gamma)
        #Maximum number of attempts to attempt each action in each state
        self.maxAttempts = int(maxAttempts)
        #Number of training sessions
        self.numTraining = int(numTraining)
        #Initialize number of games played to 0
        self.episodesSoFar = 0
        #Initialize the Q values and counts to 0
        self.qValues = defaultdict(float)
        self.counts = defaultdict(int)
        #Initialize the previous state and action to None
        self.previousState = None
        self.previousAction = None

# Accessor functions for the variable episodesSoFar controlling learning
    #Increasing number of games played by 1
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1
    #Retuning the number of games played so far
    def getEpisodesSoFar(self):
        return self.episodesSoFar
    #Return the number of training sessions
    def getNumTraining(self):
        return self.numTraining
    # Accessor functions for parameters
    #Setting the exploration rate to a given value
    def setEpsilon(self, value: float):
        self.epsilon = value
    #Return the learning rate, alpha
    def getAlpha(self) -> float:
        return self.alpha
    #Setting the learning rate, alpha, to a given value
    def setAlpha(self, value: float):
        self.alpha = value
    #Return the discount factor, gamma.
    def getGamma(self) -> float:
        return self.gamma
    #Return the maximum number of attempts to try each action in each state
    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        
        #Calculate the manhattan distance between Pacman and the ghost in the start state
        startGhostDistance = util.manhattanDistance(startState.getPacmanPosition(), 
                                    startState.getGhostPosition(1))
        #Calculate the manhattan distance between Pacman and the ghost in the end state
        endGhostDistance = util.manhattanDistance(endState.getPacmanPosition(), 
                                    endState.getGhostPosition(1))
        #Return a reward of 10, if the end state is win
        if endState.isWin():
            return 10
        #Return a rewards of -10, if the end state is lose
        elif endState.isLose(): 
            return -10
        #If the ghost is 2 units away from Pacman or less, return a reward of -5.
        elif endGhostDistance < 2:
            return -5
        #If Pacman has eaten the food, return a reward of 1 
        #(if there is less food in the end state in comparison to the start state)
        elif endState.getNumFood() < startState.getNumFood():
            return 1
        #If the Pacman approaches the ghost, return a reward of -2 (if the distance between Pacman 
        #and the ghost is less in the end state than in the start state)
        elif endGhostDistance < startGhostDistance:
            return -2
        #If none of the above apply, return a reward of -1
        else:
            return -1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    
    #Returns the Q value for a given state and action
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.qValues[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature

    #Returns the maximum Q-value for any action in a given state
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        values = []
        
        #Iterates over all legal actions in the given state
        for action in state.legalActions:
            #Appends the Q value of the state and action to the values' array
            values.append(self.getQValue(state, action))
        #Returns the maximum Q values of the state,
        #however if the list is empty then return 0.
        return max(values) if values else 0


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature

    #Based on the reward, and MaxQValue received from the next state, 
    #it updates the Q value for the given state and action
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """

        #Formula from Lecture 8, week 9, RL pt2.
        self.qValues[(state, action)] += self.alpha * (reward + self.gamma * self.maxQValue(nextState) - self.qValues[(state, action)])

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
        
    #Updates the visitation counts for a given state and action by 1
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.counts[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
        
    #Returns the number of times that the action has been taken in a given state
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature

    #Returns the utility values for a given state and action based on the counts
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        #Setting the condition that if the counts are greater than the maximum attempts,
        #and learning rate is greater than 0, return 0.
        if counts > self.maxAttempts and self.alpha > 0:
            return 0
        #Otherwise, return the corresponding utility value
        else:
            return utility
        
    #determines the best action 
    def bestAction(self, state: GameStateFeatures) -> Directions:
        """
        Args:
            state: The current state

        Returns:
            The best action according to the Q function
        """

        #Initializing the best actions to an empty array
        bestActions = []
        #Initializing the best value to negative infinity, so any value will be greater than it
        bestValue = - float('inf')
        #Iterating over all legal actions for that state
        for action in state.legalActions:
            #Getting the Q value for state and action
            value = self.getQValue(state, action)
            #Apply the exploration function to the Q value
            value = self.explorationFn(value, self.getCount(state, action))
            #If the value is greater than the current best value
            if value > bestValue:
                #then update/assign the best value to the value
                bestValue = value
                #Update/assign the best action list to the current action
                bestActions = [action]
            #If the value is equal to the best value,
            elif value == bestValue:
                #then append the action to the best action list
                bestActions.append(action)
        #Return a random choice from the best actions list
        return random.choice(bestActions)
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature

    #Implementing epsilon greedy algorithm to determines 
    #the best action to take in the current state
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        
        #Getting all legal actions that can be performed in the current state
        legal = state.getLegalPacmanActions()
        #Remove the STOP action from legal actions
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        #Getting features from the current state
        stateFeatures = GameStateFeatures(state)
        #Choosing a random action from the legal action list, with probability epsilon
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        #With probability 1 - epsilon, allow Q function to determine the best action
        else:
            action = self.bestAction(stateFeatures)
        #If the previous state exists and learning rate, alpha, is greater than 0
        if self.previousState and self.alpha > 0:
            #Calculate reward from previous state transitioning to current state
            reward = self.computeReward(self.previousState, state)
            #Perform Q learning update with the reward, previous state, action and current state.
            self.learn(GameStateFeatures(self.previousState), self.previousAction, reward, stateFeatures)
        #Update/assign the previous state and action to the current state and action
        self.previousState = state
        self.previousAction = action
        #Update the visitation counts for the current state and action
        self.updateCount(stateFeatures, action)
        #Return the action for the current state
        return action

    #Keep track of the number of games played, and set learning parameters to zero 
    #when we are done with the pre-set number of training episodes
    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        #Print the number of games played so far
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        #Increment the number of games played so far
        self.incrementEpisodesSoFar()
        #If the learning rate, alpha, is greater than 0
        if self.alpha > 0:
            #Calculate the reward from the previous state transitioning to the current state
            reward = self.computeReward(self.previousState, state)
            #Perform Q learning update with the reward, previous state, action and current state.
            self.learn(GameStateFeatures(self.previousState), self.previousAction, reward, GameStateFeatures(state))
        #Reset the previous state and action to None
        self.previousState = None
        self.previousAction = None
        #If the number of games played so far is equal to the number of training games
        if self.getEpisodesSoFar() == self.getNumTraining():
            #Print a message saying that training is done
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            #Set the learning and exploration rates to 0
            self.setAlpha(0)
            self.setEpsilon(0)
