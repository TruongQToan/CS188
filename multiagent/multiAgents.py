# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, itertools

from game import Agent

def euclideanDistance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def mean(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)

def getScore(p, listOfPositions, distanceFunction):
    def modifiedDistance(p1):
        extra = 0
        extra += 1 if p1[0] < p[0] else 0
        extra += 1 if p1[1] < p[1] else 0
        return distanceFunction(p, p1) + extra

    if len(listOfPositions) == 0:
        return 0
    distances = [modifiedDistance(p1) for p1 in listOfPositions]
    minScore = mean(distances)
    return minScore

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodPos = [(x, y) for x in range(newFood.width) for y in range(newFood.height) if newFood[x][y]]
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')
        # get min distance to Food
        x = getScore(newPos, newFoodPos, util.manhattanDistance)
        # get min distance to Ghost
        y = getScore(newPos, newGhostPos, util.manhattanDistance)
        if y == 0.0:
            return float('-inf')
        # number of food remained
        z = newFood.count()
        # number of ghosts remained
        t = len(newGhostPos)
        a, b, c, d = 1, 100, 1, 1
        score = y / x - 100 * (t + z)
        successorGameState.data.score = score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def isTerminal(self, state, depth):
        return state.isWin() or state.isLose() or self.depth + 1 == depth

    def maxValueState(self, state, depth):
        if self.isTerminal(state, depth):
            return self.evaluationFunction(state), state
        nextActions = state.getLegalActions(0)
        nextStates = [state.generateSuccessor(0, action) for action in nextActions]
        maxScore = float('-inf')
        maxAction = None
        for i, nextState in enumerate(nextStates):
            value = self.minValueState(nextState, depth)
            if maxScore < value:
                maxScore = value
                maxAction = nextActions[i]
        return maxScore, maxAction

    def getNextPacmanState(self, agentIndex, state):
        if state.isWin() or state.isLose() or agentIndex >= self.numAgents:
            yield state
            return
        nextActions = state.getLegalActions(agentIndex)
        for action in nextActions:
            nextState = state.generateSuccessor(agentIndex, action)
            yield from self.getNextPacmanState(agentIndex + 1, nextState)

    def minValueState(self, state, depth):
        if self.isTerminal(state, depth):
            return self.evaluationFunction(state)
        minScore = float('+inf')
        for nextState in self.getNextPacmanState(1, state):
            value, _ = self.maxValueState(nextState, depth + 1)
            if minScore > value:
                minScore = value
        return minScore

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        score, action = self.maxValueState(gameState, 1)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def isTerminal(self, state, depth):
        return state.isWin() or state.isLose() or self.depth + 1 == depth

    def maxValueState(self, state, depth, alpha, beta):
        if self.isTerminal(state, depth):
            value = self.evaluationFunction(state)
            return value, None
        nextActions = state.getLegalActions(0)
        maxScore = float('-inf')
        maxAction = None
        for i, nextAction in enumerate(nextActions):
            nextState = state.generateSuccessor(0, nextAction)
            value = self.minValueState(nextState, 1, depth, alpha, beta)
            if value > beta:
                return value, nextActions[i]
            alpha = max(alpha, value)
            if maxScore < value:
                maxScore = value
                maxAction = nextActions[i]
        return maxScore, maxAction

    def minValueState(self, state, agentId, depth, alpha, beta):
        if self.isTerminal(state, depth):
            value = self.evaluationFunction(state)
            return value
        nextActions = state.getLegalActions(agentId)
        minScore = float('+inf')
        for i, nextAction in enumerate(nextActions):
            nextState = state.generateSuccessor(agentId, nextAction)
            if agentId == self.numAgents - 1:
                value, _ = self.maxValueState(nextState, depth + 1, alpha, beta)
            else:
                value = self.minValueState(nextState, agentId + 1, depth, alpha, beta)
            if value < alpha:
                return value
            beta = min(beta, value)
            if minScore > value:
                minScore = value
        return minScore

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        alpha = float('-inf')
        beta = float('inf')
        score, action = self.maxValueState(gameState, 1, alpha, beta)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
