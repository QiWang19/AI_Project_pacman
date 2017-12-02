# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

import pacman

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        oldFoodList = (currentGameState.getFood()).asList()
        totalPts = 0;
        #compare the distance from food to pacman's new pos in increasing order
        #TODO cmp change
        oldFoodList.sort(lambda f1, f2: ManhDistCmp(newPos, f1, f2))
        foodPts = manhattanDistance(newPos, oldFoodList[0])
        if foodPts == 0:
            totalPts = totalPts + 2
        else:
            totalPts = totalPts + 1.0 / foodPts

        ghostsPos = []
        for ghost in newGhostStates:
            ghostsPos.append(ghost.getPosition())
        ghostPts = 0
        if len(ghostsPos) != 0:
            ghostsPos.sort(lambda g1, g2: ManhDistCmp(newPos, g1, g2))
        if manhattanDistance(newPos, ghostsPos[0]) == 0:
            return -999
        else:
            ghostPts = -3.0 / manhattanDistance(newPos, ghostsPos[0])
        totalPts = totalPts + ghostPts

        if action == Directions.STOP:
           totalPts = totalPts - 1
        #successorGameState.setScore(totalPts)

        return totalPts

def ManhDistCmp(pos, p1, p2):
    diff = manhattanDistance(pos, p1) - manhattanDistance(pos, p2)
    if diff < 0:
        return -1
    elif diff > 0:
        return 1
    else:
        return 0

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
countformoves = 0
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        """
        """
        #TODO: hard part
        max(state, 0)
        exp(0,1)
        exp(0,2)
        max(state, 1)
        exp(1,1)
        exp(1,2)
        evaluation
        """
        #https://github.com/lightninglu10/pacman-minimax/blob/master/multiAgents.py

        "*** YOUR CODE HERE ***"

        PACMAN = 0

        def max_agent(state, depth):
            global countformoves
            if state.isWin() or state.isLose():
                #define in pacman.py
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = exp_agent(state.generateSuccessor(PACMAN, action), depth, 1)
                if score > best_score:
                    best_score = score
                    best_action = action
            if depth == 0:
                countformoves = countformoves + 1
                return best_action
            else:
                return best_score

        def exp_agent(state, depth, ghost):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN:  # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1)
                else:
                    score = exp_agent(state.generateSuccessor(ghost, action), depth, next_ghost)
                if score < best_score:
                    best_score = score
            return best_score

        return max_agent(gameState, 0)
        #util.raiseNotDefined()




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        PACMAN = 0
        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return best_action
            else:
                return best_score
        def min_agent(state, depth, ghost, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score
        return  max_agent(gameState, 0, float("-inf"), float("inf"))

        util.raiseNotDefined()

#https://web.uvic.ca/~maryam/AISpring94/Slides/06_ExpectimaxSearch.pdf
#Page9 What probabilities to use  a distribution to assign probabilities to opponent-actions

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

        numAgents = gameState.getNumAgents()
        totalDepth = self.depth * numAgents  # 1 ply means each agent moves one time

        # remove STOP from pacman's action list if it exists in the list
        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # get a list pacman's successor states
        newStates = []
        for action in actions:
            newStates.append(gameState.generateSuccessor(0, action))

        # get a list of values of pacman's successor states
        vals = []
        for nextState in newStates:
            vals.append(self.ExpectimaxValue(nextState, 1, numAgents, totalDepth - 1))

        # find the largest value(s) the pacman can get
        # among all the successor states
        maxVal = max(vals)
        bestIndices = [idx for idx in range(len(vals)) if vals[idx] == maxVal]

        # return the action that will let pacman get
        # the largest value; randomly pick one action if
        # there are multiple actions with the greatest value
        chosenIdx = random.choice(bestIndices)
        global countformoves
        countformoves = countformoves + 1
        #print "count for moves is %s" %countformoves
        return actions[chosenIdx]

    def ExpectimaxValue(self, gameState, agentIdx, numAgents, depth):
        # return the value of current state using evaluationFunction
        # if the current state is a terminal state (win/lose) or if
        # the function hits the specified depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            #print "count for moves is %s" % countformoves
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIdx)
        # remove STOP from actions list if the agent is pacman
        if agentIdx == 0:
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)

        # get a list of successor states
        newStates = []
        for action in actions:
            newStates.append(gameState.generateSuccessor(agentIdx, action))

        # evalue the successor states by recursively calling this function
        # until it is terminal state or depth is 0
        vals = []
        for nextState in newStates:
            vals.append(self.ExpectimaxValue(nextState, (agentIdx + 1) % numAgents, numAgents, depth - 1))

        # if the agent is pacman, return the maximum value;
        # otherwise, return the expectation according to
        # how the ghosts act (assume the ghost has equal chance
        # to choose each action among all the legal actions)
        if agentIdx == 0:
            return max(vals)
        else:
            return sum(vals) / len(actions)

        #util.raiseNotDefined()

#https://github.com/lightninglu10/pacman-minimax/blob/master/multiAgents.py
#score 6/6
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    def closest_dot(cur_pos, food_pos):
        '''
        find mini distance to all the food from curt pacman position
        :param cur_pos:
        :param food_pos:
        :return:
        '''
        food_distances = []
        for food in food_pos:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1



    def closest_ghost(cur_pos, ghosts):
        '''
        curt pacman mini distance to all ghost
        :param cur_pos:
        :param ghosts:
        :return:
        '''
        food_distances = []
        for food in ghosts:
            food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1



    def ghost_stuff(cur_pos, ghost_states, radius, scores):
        '''
        if ghost not in safe circle, decrease score
        :param cur_pos:
        :param ghost_states:
        :param radius:
        :param scores:
        :return:
        '''
        #num_ghosts = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                #num_ghosts += 1
        return scores



    def food_stuff(cur_pos, food_positions):
        '''
        find sum of distance to all the food
        :param cur_pos:
        :param food_positions:
        :return:
        '''
        food_distances = []
        sumDis = 0
        for food in food_positions:
            food_distances.append(util.manhattanDistance(food, cur_pos))
            sumDis = sumDis + util.manhattanDistance(food, cur_pos)
        return sumDis


    def num_food(cur_pos, food):
        return len(food)


    def closest_capsule(cur_pos, caps_pos):
        '''
        mini distance to all pellet
        :param cur_pos:
        :param caps_pos:
        :return:
        '''
        capsule_distances = []
        for caps in caps_pos:
            capsule_distances.append(util.manhattanDistance(caps, cur_pos))
        return min(capsule_distances) if len(capsule_distances) > 0 else 9999999

    def scaredghosts(ghost_states, cur_pos, scores):
        '''
        use scaretime and distance to change score,
        find max score
        :param ghost_states:
        :param cur_pos:
        :param scores:
        :return:
        '''
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 4:
                scoreslist.append(scores + 50)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 3:
                scoreslist.append(scores + 60)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 2:
                scoreslist.append(scores + 70)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 1:
                scoreslist.append(scores + 90)

        return max(scoreslist) if len(scoreslist) > 0 else scores

    def ghostattack(ghost_states, cur_pos, scores):
        '''
        if scared time, decrease the score
        :param ghost_states:
        :param cur_pos:
        :param scores:
        :return:
        '''
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer == 0:
                scoreslist.append(scores - util.manhattanDistance(ghost.getPosition(), cur_pos) - 10)
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def scoreagent(cur_pos, food_pos, ghost_states, caps_pos, score):
        '''
        curt pos --> distance pellet / ghost
        curt pos --> food / ghost
        curt pos --> pellet food
        :param cur_pos:
        :param food_pos:
        :param ghost_states:
        :param caps_pos:
        :param score:
        :return:
        '''
        if closest_capsule(cur_pos, caps_pos) < closest_ghost(cur_pos, ghost_states):
            return score + 40
        if closest_dot(cur_pos, food_pos) < closest_ghost(cur_pos, ghost_states) + 3:
            return score + 20
        if closest_capsule(cur_pos, caps_pos) < closest_dot(cur_pos, food_pos) + 3:
            return score + 30
        else:
            return score

    capsule_pos = currentGameState.getCapsules()
    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()


    score = scoreagent(pacman_pos, food, ghosts, capsule_pos, score)
    score = scaredghosts(ghosts, pacman_pos, score)
    score = ghostattack(ghosts, pacman_pos, score)
    score -= .35 * food_stuff(pacman_pos, food)


    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

