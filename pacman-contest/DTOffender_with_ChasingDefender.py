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


#################
# Team creation #
#################
import sys

sys.path.append("teams/Zelda/")

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

class OffensiveReflexAgent(CaptureAgent):
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
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        # set the border of two teams
        if self.red:
            borderX = (gameState.data.layout.width - 2) / 2
        else:
            borderX = (gameState.data.layout.width - 2) / 2 + 1
        self.border = ([(borderX, y) for y in range(1, gameState.data.layout.height - 1)
                        if not gameState.hasWall(borderX, y)])
        # initialize team, opponents, defendingState and a random food
        self.team = self.getTeam(gameState)
        self.opponents = self.getOpponents(gameState)

    def chooseAction(self, gameState):
        # get my position
        x, y = gameState.getAgentState(self.index).getPosition()
        position = (int(x), int(y))
        # get a list of legal actions
        actions = gameState.getLegalActions(self.index)
        # if there are actions in the list, remove stop
        if len(actions) > 0:
            actions.remove(Directions.STOP)
        # get food I can eat
        opponentFood = self.getFood(gameState).asList()
        # add capsules into the food list
        capsules = self.getCapsules(gameState)
        opponentFood += capsules
        # if eatable food more than 2
        if len(opponentFood) > 2:
            # get opponent status
            scaredTime = [gameState.getAgentState(opponent).scaredTimer for opponent in self.opponents]
            ghosts = [opponent for opponent in self.opponents if not gameState.getAgentState(opponent).isPacman]
            ghostsConfig = [gameState.getAgentState(ghost).configuration for ghost in ghosts]
            # if opponents are all defending
            if len(ghosts) == 2:
                ghost1 = ghostsConfig[0]
                ghost2 = ghostsConfig[1]
                # if they are not scared
                if ghost1 is not None and ghost2 is not None and (scaredTime[0] <= 5 or scaredTime[1] <= 5):
                    ghost1Position = ghost1.getPosition()
                    ghost2Position = ghost2.getPosition()
                    # if there are capsules and I am closer to a capsule, eat the closest capsule
                    if len(capsules) > 0:
                        capsuleDistance, closestCapsule = self.getClosest(gameState, position, capsules)
                        if capsuleDistance < self.getMazeDistance(ghost1Position,
                                                                  closestCapsule) and capsuleDistance < self.getMazeDistance(
                                ghost2Position, closestCapsule):
                            bestAction = self.bestAction(gameState, closestCapsule, actions)
                            return bestAction

                    if self.getMazeDistance(position, ghost1Position) <= 5 or self.getMazeDistance(position,
                                                                                                   ghost2Position) <= 5:
                        # if pacman gets too close to opponent ghost, escape
                        if gameState.getAgentState(self.index).isPacman:
                            distance, closestEscape = self.getClosest(gameState, position, self.border)
                            bestAction = self.bestAction(gameState, closestEscape, actions)
                            return bestAction

            # if there is only 1 ghost
            elif len(ghosts) == 1:
                # set up ghost status
                ghost = ghostsConfig[0]
                if ghost is not None and gameState.getAgentState(ghosts[0]).scaredTimer <= 5:
                    ghostPosition = ghost.getPosition()
                    # rush to capsule if I am closer to it
                    if len(capsules) > 0:
                        capsuleDistance, closestCapsule = self.getClosest(gameState, position, capsules)
                        if capsuleDistance < self.getMazeDistance(ghostPosition, closestCapsule):
                            bestAction = self.bestAction(gameState, closestCapsule, actions)
                            return bestAction

                    if self.getMazeDistance(position, ghostPosition) <= 5:
                        # run from ghost if I am pacman
                        if gameState.getAgentState(self.index).isPacman:
                            distance, closestEscape = self.getClosest(gameState, position, self.border)
                            bestAction = self.bestAction(gameState, closestEscape, actions)
                            return bestAction

            distance2Food, closestFood = self.getClosest(gameState, position, opponentFood)
            distance2Escape, closestEscape = self.getClosest(gameState, position, self.border)
            if gameState.getAgentState(self.index).numCarrying > distance2Escape:
                bestAction = self.bestAction(gameState, closestEscape, actions)
                return bestAction
            bestAction = self.bestAction(gameState, closestFood, actions)
            return bestAction
        else:
            distance, closestEscape = self.getClosest(gameState, position, self.border)
            bestAction = self.bestAction(gameState, closestEscape, actions)
            return bestAction

    def getClosest(self, gameState, position, dots):
        distance, closest = min([(self.getMazeDistance(position, dot), dot) for dot in dots])
        return (distance, closest)

    def getFurtherest(self, gameState, position, dots):
        distance, furtherest = max([(self.getMazeDistance(position, dot), dot) for dot in dots])
        return furtherest

    def bestAction(self, gameState, dotPosition, actions):
        distance, bestAction = min(
            [(self.getMazeDistance(self.getSuccessor(gameState, action), dotPosition), action) for action in actions])
        return bestAction

    def isAtBorder(self, gameState):
        position = gameState.getAgentState(self.index).getPosition()
        if position in self.border:
            return True
        else:
            return False

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return pos


class DefensiveReflexAgent(CaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def registerInitialState(self, gameState):

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        if self.red:
            self.middle = (gameState.data.layout.width - 2) / 2
        else:
            self.middle = (gameState.data.layout.width - 2) / 2 + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.middle, i):
                self.boundary.append((self.middle, i))

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        gameStateCopy = gameState.deepCopy()
        actions = gameStateCopy.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        values = [self.evaluate(gameState, a) for a in actions]
        # values = []
        # for a in actions:
        #     value = self.MCTS(gameState, a, 0.1, 2, 1)
        #     values.append(value)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        value = features * weights
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
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

        dists = []
        for e in enemies:
            dists.append(self.getMazeDistance(myPos, e.start.pos))
        features['enemyDistance'] = min(dists)

        for a in invaders:
            if self.getMazeDistance(successor.getAgentState(self.index).getPosition(), a.getPosition()) < 2:
                features['inDangerousZone'] = 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        scaredTime = successor.getAgentState(self.index).scaredTimer
        if scaredTime > 0:
            return {'onDefense': 1000, 'enemyDistance': -120, 'numInvaders': -1000, 'stop': -100,
                    'reverse': -2, 'invaderDistance': -140, 'inDangerousZone': -10000}
        else:
            return {'onDefense': 1000, 'enemyDistance': -120, 'numInvaders': -1000, 'stop': -100,
                    'reverse': -2, 'invaderDistance': -140}
