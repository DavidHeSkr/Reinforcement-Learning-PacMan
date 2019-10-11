# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from featureExtractors import *

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, numTraining,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):

  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def __init__(self, index, timeForComputing = .1):
    super().__init__(index)
    self.mode = 0
    self.home_point = (0,0)
    #self.index = index
    #self.mode = 0

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    #################################################################
    #################################################################
    #################################################################
    """
    Picks among the actions with the highest Q(s,a).
    """
    """

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
     """
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    #print("features are", features)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):

  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    ##########################################
    #For test reason we only control the red team as the algorithm implemented team
    ##########################################

    #if self.red:
    if gameState.data.agentStates[self.index].numCarrying > 1:
        self.mode = 1
        myPos_ = self.getSuccessor(gameState, action).getAgentState(self.index).getPosition()
        myPos_y_ = myPos_[1]
        home_point_y = myPos_y_
        ####for the team on the left
        if gameState.isRed:
            home_point_x = gameState.data.layout.width/2 -1
        ####for the team on the right
        if not gameState.isRed:
            home_point_x = gameState.data.layout.width/2
        while gameState.data.layout.walls[int(home_point_x)][int(home_point_y)]:
            home_point_y += 1
        self.home_point = (home_point_x, home_point_y)

        #print("my home_point is", self.home_point)
    # if the food has been unloaded
    else:
        self.mode = 0
    #normal state, continue to eat
    ############################LEFT OVER HERE THE DISTANCE TO OPPONENT MAY HAVE PROBLEM
    ####################################################################################
    ####################################################################################
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    upperFoodList = [a for a in foodList if a[1] > gameState.data.layout.height/2]
    lowerFoodList = [a for a in foodList if a[1] <= gameState.data.layout.height/2]
    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ##########
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    #if len(defenders) > 0:
    dists_defenders = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
    dists_invaders = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
    if len(dists_defenders):
        min_defenders = min(dists_defenders)
    else:
        min_defenders = 10000000
    if len(dists_invaders) > 0:
        min_invaders = min(dists_invaders)
    else:
        min_invaders = 100000000
    if min_invaders < min_defenders:
        min_defenders = -min_invaders
    features['numInvaders'] = len(invaders)
    features['defenderDistance'] = 1/(min_defenders)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    if self.mode == 0:
        features['successorScore'] = -len(foodList)#self.getScore(successor)
        # Compute distance to the nearest foo
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
          if self.index == self.getTeam(gameState)[0]:
              minDistance = min([self.getMazeDistance(myPos, food) for food in upperFoodList])
              features['distanceToFood'] = minDistance
          else:
              minDistance = min([self.getMazeDistance(myPos, food) for food in lowerFoodList])
              features['distanceToFood'] = minDistance

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        #print("features in baseline:", features)
        return features
    #returns back to home to unload the food
    else:
        ###################################
        ###################################
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        features['distanceToHome'] = self.getMazeDistance(myPos, self.home_point)
        return features

    #the default team
    """
    else:

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        features['successorScore'] = -len(foodList)#self.getScore(successor)

        # Compute distance to the nearest foo
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
          if self.index == 0:
              minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
              features['distanceToFood'] = minDistance
        return features
    """

  def getWeights(self, gameState, action):
    #if self.red:
    if self.mode == 0:
        #NOT EATEN ENOUGH FOOD
        return {'numInvaders': -1000, 'defenderDistance': -150, 'successorScore': 1000, 'distanceToFood': -10, 'stop': -20, 'reverse': 10}
    else:
        #EATEN ENOUGH FOOD WANT TO GO HOME
        return {'defenderDistance': -700, 'distanceToHome': -30, 'stop': -20, 'reverse': 10}
    #else:
    #    return {'successorScore': 1000, 'distanceToFood': -10}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    ################
    #print("defensive self index", self.index)
    ################
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]

      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



# learningAgents.py
# -----------------
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


from game import Directions, Agent, Actions

import random,util,time

class ValueEstimationAgent(CaptureAgent):

    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, index, timeForComputing = .1, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        super().__init__(index)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    def registerInitialState(self, gameState):
      self.start = gameState.getAgentPosition(self.index)
      CaptureAgent.registerInitialState(self, gameState)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with chooseAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def chooseAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """
    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raiseNotDefined()

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        # episodeRewards : total rewards in a training episode
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, index, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        super().__init__(index)
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions(self.index)
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            # get the socre change by the new move, compared to the last move
            reward = state.getScore() - self.lastState.getScore()
            # pass the reward added to the observeTransition function
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        ValueEstimationAgent.registerInitialState(self, state)
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print ('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))

############################################################################


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - chooseAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "Q6"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "Q6"
        return self.values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "Q6"
        q_values = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if len(q_values):
          return max(q_values)
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "Q6"
        actions = self.getLegalActions(state)
        if not actions:
          return None
        max_q = self.computeValueFromQValues(state)
        best_actions = [action for action in actions if self.getQValue(state, action) == max_q]
        return random.choice(best_actions)

    def chooseAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.values[(state, action)] = (1 - self.alpha) * self.values[(state, action)] + self.alpha * sample


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, index, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        print("**args", **args)
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        args['index'] = index  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def chooseAction(self, state):
        """
        Simply calls the chooseAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.chooseAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, index, extractor='SimpleExtractor'):
        print("extractor is:", extractor);
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, index)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "Q10"

        #actions = self.getLegalActions(state)
        #if not actions:
        #  return None
        #max_q = self.computeValueFromQValues(state)
        #best_actions = [action for action in actions if self.getQValue(state, action) == max_q]
        #return random.choice(best_actions)


        feats = self.getFeatures(state, action)
        print("the feats values are: ", feats)
        return sum([self.weights[feat] * value for feat, value in feats.items()])

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "Q10"

        feats = self.getFeatures(state, action)
        print("feats:", feats)
        diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for feat in feats:
          self.weights[feat] += self.alpha * diff * feats[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "Q10"
            pass

    #get the gameState of the next move
    def getSuccessor(self, gameState, action):
      """
      Finds the next successor which is a grid position (location tuple).
      """
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
      else:
        return successor

    #get the correspond features of next move
    def getFeatures(self, gameState, action):
      ##########################################
      #For test reason we only control the red team as the algorithm implemented team
      ##########################################
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      foodList = self.getFood(successor).asList()
      #print("food list is:", foodList)
      myPos = successor.getAgentState(self.index).getPosition()
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      ##########
      defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      #if len(defenders) > 0:
      dists_defenders = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      dists_invaders = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      #print("the defensers distacne is: ", dists_defenders)
      # the max distance between 2 pacman is 74
      if len(dists_defenders):
          min_defenders = min(dists_defenders)
      else:
          min_defenders = 10000000
      if len(dists_invaders) > 0:
          min_invaders = min(dists_invaders)
      else:
          min_invaders = 100000000
      if min_invaders < min_defenders:
          min_defenders = -min_invaders

      features['numInvaders'] = len(invaders)
      #print("------------min_defenders is: ", min_defenders)
      features['defenderDistance'] = 1/(min_defenders)

      #if action == Directions.STOP: features['stop'] = 1
      #rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      #if action == rev: features['reverse'] = 1

      #if self.mode == 0:
      features['successorScore'] = -len(foodList)#self.getScore(successor)


      # Compute distance to the nearest food
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

      #if action == Directions.STOP: features['stop'] = 1
      #rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      #if action == rev: features['reverse'] = 1
      return features
