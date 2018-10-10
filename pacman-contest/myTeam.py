# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import math
from distanceCalculator import Distancer


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        if self.red:
            middle = (gameState.data.layout.width - 2) / 2
        else:
            middle = (gameState.data.layout.width - 2) + 1 / 2
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                self.boundary.append((middle, i))

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        gameStateCopy = gameState.deepCopy()
        actions = gameStateCopy.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        # values1 = [self.evaluate(gameState, a) for a in actions]
        values = []
        for a in actions:
            # successor = gameState.generateSuccessor(self.index, a)
            value = self.MCTS(gameState, a, 0.01, 3)
            values.append(value)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

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

        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()

        return features

    def getWeights(self, gameState, action):
        return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 0}

    def qLearning(self, gameState, decay, depth):
        new_state = gameState.deepCopy()
        if depth == 0:
            result_list = []
            actions = new_state.getLegalActions(self.index)
            actions.remove(Directions.STOP)

            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            a = random.choice(actions)
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(self.evaluate(next_state, Directions.STOP))
            return max(result_list)

        # Get valid actions
        result_list = []
        actions = new_state.getLegalActions(self.index)
        current_direction = new_state.getAgentState(self.index).configuration.direction
        # The agent should not use the reverse direction during simulation

        reversed_direction = Directions.REVERSE[current_direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)

        # Randomly chooses a valid action
        for a in actions:
            # Compute new state and update depth
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(
                self.evaluate(next_state, Directions.STOP) + decay * self.qLearning(next_state, decay, depth - 1))
        return max(result_list)

    def MCTS(self, gameState, action, discount, depth):
        gameStateCopy = gameState.deepCopy()
        child = gameState.generateSuccessor(self.index, action).deepCopy()
        actions = child.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        if depth > 0:
            a = random.choice(actions)
            value = self.evaluate(gameState, action)

            value = value + discount * self.MCTS(child, a, discount, depth - 1)
            return value

        else:
            return self.evaluate(gameState, action)


class OffensiveReflexAgent(DummyAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        foodList = self.getFood(successor).asList()
        features['foods'] = -len(foodList)  # self.getScore(successor)

        state = successor.getAgentState(self.index)

        distance = []
        pos = state.getPosition()
        opponents = self.getOpponents(gameState)
        for o in opponents:
            opponent = successor.getAgentState(o)
            if not opponent.isPacman:
                opponentPos = opponent.getPosition()
                if opponentPos is not None:
                    distance.append(self.getMazeDistance(pos, opponentPos))

        if len(distance) > 0:
            features['disToOpponent'] = min(distance)

        if gameState.getAgentState(self.index).isPacman:
            corner = self.isCorner(successor, 5)
            # if the state is in a corner
            if corner:
                dis = features['disToOpponent']
                if dis != 0:
                    features['RiskInCorner'] = 1000 / (dis ** 2)
                else:
                    features['RiskInCorner'] = 1

        numOfCarry = successor.getAgentState(self.index).numCarrying
        boundaryMin = 1000000
        for i in range(len(self.boundary)):
            disBoundary = self.getMazeDistance(pos, self.boundary[i])
            if disBoundary < boundaryMin:
                boundaryMin = disBoundary

        features['return'] = boundaryMin * math.sqrt(numOfCarry) - features['disToOpponent']

        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            features['reverse'] = 1

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # add distanceToCapsule feature
        capsuleList = set(gameState.data.capsules) - set(self.getCapsulesYouAreDefending(gameState))
        if len(capsuleList) > 0:
            minCapsuleDistance = 99999
            for c in capsuleList:
                distance = self.getMazeDistance(pos, c)
                if distance < minCapsuleDistance:
                    minCapsuleDistance = distance
            features['distanceToCapsule'] = minCapsuleDistance
        else:
            features['distanceToCapsule'] = 0

        return features

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        foodList = self.getFood(successor).asList()

        if len(foodList) > 2:
            # some one in vision
            if len(visible) > 0:
                for agent in visible:
                    # someone scared in vision
                    if agent.scaredTimer >= 10:
                        # in case the agent is in others boundary
                        if gameState.getAgentState(self.index).isPacman:
                            return {'foods': 100, 'distanceToFood': -2, 'disToOpponent': 0, 'RiskInCorner': 0,
                                    'return': -0.01, 'reverse': -1, 'distanceToCapsule': 0}
                        else:
                            return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 0, 'RiskInCorner': 0,
                                    'return': 0, 'reverse': -1}

                    elif 0 < agent.scaredTimer < 10:
                        # in case the agent is in others boundary
                        if gameState.getAgentState(self.index).isPacman:
                            return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 100, 'RiskInCorner': -1,
                                    'return': -0.5, 'reverse': -1, 'distanceToCapsule': 0}
                        else:
                            return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 0, 'RiskInCorner': 0,
                                    'return': 0, 'reverse': -1}

                    # Visible and not scared
                    else:
                        # in case the agent is in others boundary
                        if gameState.getAgentState(self.index).isPacman:
                            return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 100, 'RiskInCorner': -1,
                                    'return': -0.5, 'reverse': -1, 'distanceToCapsule': -2}
                        else:
                            return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 100, 'RiskInCorner': 0,
                                    'return': 0, 'reverse': -1}

            # no one in vision
            else:
                # in case the agent is in others boundary
                if gameState.getAgentState(self.index).isPacman:
                    return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 100, 'RiskInCorner': -1,
                            'return': -0.5, 'reverse': -1, 'distanceToCapsule': -2}
                else:
                    return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 100, 'RiskInCorner': 0,
                            'return': 0, 'reverse': -1}
        else:
            return {'foods': 0, 'distanceToFood': 0, 'disToOpponent': 100, 'RiskInCorner': -1,
                    'return': -10, 'reverse': -1, 'distanceToCapsule': 0}

    def isCorner(self, gameState, depth):
        if depth > 0:
            stateCopy = gameState.deepCopy()
            legalActions = stateCopy.getLegalActions(self.index)
            currentAction = stateCopy.getAgentState(self.index).configuration.direction
            if Directions.REVERSE[currentAction] in legalActions:
                legalActions.remove(Directions.REVERSE[currentAction])
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)

            # if the legal actions has only STOP and other 1 action. that means this state is a corner
            if len(legalActions) == 0:
                return True
            elif len(legalActions) == 1:
                successor = stateCopy.generateSuccessor(self.index, legalActions[0])
                return self.isCorner(successor, depth - 1)
            else:
                return False
        return False


class DefensiveReflexAgent(DummyAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
    def getCenterPointOfDefensiveFood(self, gameState):
        homeFoods = self.getFoodYouAreDefending(gameState).asList()
        while len(homeFoods) > 1:
            newCenters = []
            while len(homeFoods) > 1:
                x = (homeFoods[0][0] + homeFoods[1][0]) >> 1
                y = (homeFoods[0][1] + homeFoods[1][1]) >> 1
                newCenters.append((x, y))
                homeFoods.remove(homeFoods[0])
                homeFoods.remove(homeFoods[0])
            for i in homeFoods:
                newCenters.append(i)
            homeFoods = newCenters
        return homeFoods[0]

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0
        
        # Computes the distance between defensive agent and food center
        foodCenter = self.getCenterPointOfDefensiveFood(gameState)
        if gameState.hasWall(foodCenter[0], foodCenter[1]):
            foodCenter = self.nearPosInGrid(gameState, foodCenter)

        features['distToFoodCenter'] = self.getMazeDistance(myPos, foodCenter)

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
        successor = self.getSuccessor(gameState,action)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 
                    'reverse': -2, 'distToFoodCenter': 0}
        else:
            return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100,
                    'reverse': -2, 'distToFoodCenter': -1}

    def nearPosInGrid(self, gameState, pos):
        left = (pos[0] - 2, pos[1])
        right = (pos[0] + 2, pos[1])
        up = (pos[0], pos[1] + 2)
        down = (pos[0], pos[1] - 2)
        dirList = [left, right, up, down]

        for dir in dirList:
            if self.inGrid(dir, gameState) and not gameState.hasWall(dir[0], dir[1]):
                return dir
        return random.choice(self.boundary)

    def inGrid(self, pos, gameState):
        if pos[0] < 1:
            return False
        elif pos[0] > gameState.data.layout.width - 1:
            return False
        elif pos[1] < 1:
            return False
        elif pos[1] > gameState.data.layout.height -1:
            return False
        else:
            return True
