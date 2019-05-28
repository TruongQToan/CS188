# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    startState = problem.getStartState()
    stack = [startState]
    visiting = []
    visited = []
    paths = []
    while len(stack) > 0:
        state = stack[-1]
        if problem.isGoalState(state):
            visiting.append(state)
            break
        if state in visiting:
            visiting.pop()
            stack.pop()
            visited.append(state)
            continue
        visiting.append(state)
        for succ, _, _ in problem.getSuccessors(state):
            if succ not in visiting and succ not in visited:
                stack.append(succ)
    actions = []
    if len(visiting) > 0:
        for i in range(len(visiting) - 1):
            state = visiting[i]
            for succ, action, _ in problem.getSuccessors(state):
                if succ == visiting[i + 1]:
                    actions.append(action)
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    startState = problem.getStartState()
    parents = {}
    queue = [startState]
    goalState = None
    visited = set()
    while len(queue) > 0:
        state = queue.pop(0)
        if state in visited:
            continue
        visited.add(state)
        #visited.append(state)
        if problem.isGoalState(state):
            goalState = state
            break
        for succ, _, _ in problem.getSuccessors(state):
            if succ not in visited:
                queue.append(succ)
                parents[succ] = state
    path = []
    state = goalState
    while state is not None: 
        path.append(state)
        state = parents[state]
        if state == startState:
            break
    path.reverse()
    path.insert(0, startState)
    actions = []
    if len(path) > 0:
        for i in range(len(path) - 1):
            state = path[i]
            for succ, action, _ in problem.getSuccessors(state):
                if succ == path[i + 1]:
                    actions.append(action)
    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priorities = {}
    parents = {}
    startState = problem.getStartState()
    priorities[startState] = 0
    queue = util.PriorityQueue()
    queue.push(startState, 0)
    goalState = None
    visited = []
    while not queue.isEmpty() > 0:
        state = queue.pop()
        if state in visited:
            continue
        accumCost = priorities[state]
        visited.append(state)
        if problem.isGoalState(state):
            goalState = state
            break
        for succ, _, costStep in problem.getSuccessors(state):
            if succ not in visited:
                priorities[succ] = accumCost + costStep
                queue.push(succ, priorities[succ])
                parents[succ] = state
    path = []
    state = goalState
    while state is not None: 
        path.append(state)
        state = parents[state]
        if state == startState:
            break
    path.reverse()
    path.insert(0, startState)
    actions = []
    if len(path) > 0:
        for i in range(len(path) - 1):
            state = path[i]
            for succ, action, _ in problem.getSuccessors(state):
                if succ == path[i + 1]:
                    actions.append(action)
    return actions
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priorities = {}
    parents = {}
    startState = problem.getStartState()
    priorities[startState] = heuristic(startState, problem)
    queue = util.PriorityQueue()
    queue.push(startState, 0)
    goalState = None
    visited = set()
    while not queue.isEmpty() > 0:
        state = queue.pop()
        if state in visited:
            continue
        accumCost = priorities[state]
        visited.add(state)
        if problem.isGoalState(state):
            goalState = state
            break
        for succ, action, costStep in problem.getSuccessors(state):
            if succ not in visited:
                priorities[succ] = accumCost + costStep + heuristic(succ, problem)
                queue.push(succ, priorities[succ])
                parents[succ] = (state, action)
    actions = []
    state = goalState
    while state is not None: 
        state, action = parents[state]
        actions.append(action)
        if state == startState:
            break
    actions.reverse()
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
