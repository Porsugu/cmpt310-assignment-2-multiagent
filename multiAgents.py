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
import random, util

from game import Agent

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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        foodScore=0
        foodNum=0
        totalMan=0
        pos=successorGameState.getPacmanPosition()
        for f in newFood.asList():
            totalMan+=manhattanDistance(pos,f)
            foodNum+=1
        if(foodNum>0):
            foodScore=totalMan/foodNum
        ghostSocre=0
        ghostNum=0
        totalGhostMan=0
        danger=False
        for g in successorGameState.getGhostPositions():
            ghostNum+=1
            dis=manhattanDistance(pos,g)
            if(dis<2):
                danger=True
                break

        if(danger):
            ghostSocre=float("-inf")
        return successorGameState.getScore()-foodScore+ghostSocre

            
        #return successorGameState.getScore()

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

    def ghostAgent(self,state,depth,index,maxDepth):
        if state.isWin() or state.isLose():
            return state.getScore()
        
        agentNum=state.getNumAgents()
        nextIndex=0
        if index<agentNum-1:
            nextIndex=index+1
        bestScore = float("inf")
        score = bestScore
        actions=state.getLegalActions(index)
        for action in actions:
            if nextIndex == 0: 
                if depth == self.depth - 1:
                    score = self.evaluationFunction(state.generateSuccessor(index, action))
                else:
                    score = self.pacAgent(state.generateSuccessor(index, action), depth + 1,maxDepth)
            else:
                score = self.ghostAgent(state.generateSuccessor(index, action), depth, nextIndex,maxDepth)
            if score < bestScore:
                bestScore = score
            
            if score<bestScore:
                bestScore=score

        return bestScore
    
    def pacAgent(self,state,depth,maxDepth):
        if state.isWin() or state.isLose():
            return state.getScore()
        actions=state.getLegalActions(0)
        bestScore = float("-inf")
        score = bestScore
        bestAction = Directions.STOP

        for action in actions:
            score=self.ghostAgent(state.generateSuccessor(0, action), depth, 1,maxDepth)
            if score > bestScore:
                    bestScore = score
                    bestAction = action

        if depth == 0:
            return bestAction
        else:
            return bestScore

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
        return self.pacAgent(gameState,0,self.depth)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def ghostAgent(self,state,depth,index,maxDepth,alpha,beta):
        if state.isWin() or state.isLose():
            return state.getScore()
        
        agentNum=state.getNumAgents()
        nextIndex=0
        if index<agentNum-1:
            nextIndex=index+1
        bestScore = float("inf")
        score = bestScore
        actions=state.getLegalActions(index)
        for action in actions:
            if nextIndex == 0: 
                if depth == self.depth - 1:
                    score = self.evaluationFunction(state.generateSuccessor(index, action))
                else:
                    score = self.pacAgent(state.generateSuccessor(index, action), depth + 1,maxDepth,alpha,beta)
            else:
                score = self.ghostAgent(state.generateSuccessor(index, action), depth, nextIndex,maxDepth,alpha,beta)
            if score<bestScore:
                bestScore=score
            beta = min(beta, bestScore)
            if bestScore < alpha:
                return bestScore

        return bestScore
    
    def pacAgent(self,state,depth,maxDepth,alpha,beta):
        if state.isWin() or state.isLose():
            return state.getScore()
        actions=state.getLegalActions(0)
        bestScore = float("-inf")
        score = bestScore
        bestAction = Directions.STOP

        for action in actions:
            score=self.ghostAgent(state.generateSuccessor(0, action), depth, 1,maxDepth,alpha,beta)
            if score > bestScore:
                    bestScore = score
                    bestAction = action
            alpha = max(alpha, bestScore)
            if bestScore > beta:
                return bestScore
            
        if depth == 0:
            return bestAction
        else:
            return bestScore

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.pacAgent(gameState,0,self.depth,float("-inf"),float("inf"))
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def ghostAgent(self,state,depth,index,maxDepth):
        if state.isWin() or state.isLose():
            return state.getScore()
        
        agentNum=state.getNumAgents()
        nextIndex=0
        if index<agentNum-1:
            nextIndex=index+1
        bestScore = float("inf")
        score = bestScore
        actions=state.getLegalActions(index)
        prob = 1.0/len(actions)
        sum=0
        for action in actions:
            if nextIndex == 0: 
                if depth == self.depth - 1:
                    sum+=self.evaluationFunction(state.generateSuccessor(index, action))*prob
                    score=sum
                else:
                    score = self.pacAgent(state.generateSuccessor(index, action), depth + 1,maxDepth)
                
            else:
                score = self.ghostAgent(state.generateSuccessor(index, action), depth, nextIndex,maxDepth)

            if score<bestScore:
                bestScore=score

        return bestScore
    
    def pacAgent(self,state,depth,maxDepth):
        if state.isWin() or state.isLose():
            return state.getScore()
        actions=state.getLegalActions(0)
        bestScore = float("-inf")
        score = bestScore
        bestAction = Directions.STOP

        for action in actions:
            score=self.ghostAgent(state.generateSuccessor(0, action), depth, 1,maxDepth)
            if score > bestScore:
                    bestScore = score
                    bestAction = action

        if depth == 0:
            return bestAction
        else:
            return bestScore

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.pacAgent(gameState,0,self.depth)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #Find closest ghost

    def closestFood(pacPos,foodList):
        disQ=util.PriorityQueue()
        for food in foodList:
            dis=manhattanDistance(pacPos,food)
            disQ.push(dis,dis)
        return float("inf") if disQ.isEmpty() else disQ.pop()
    
    def closestGhost(pacPos,ghostList):
        ghostQ=util.PriorityQueue()
        for ghost in ghostList:
            dis=manhattanDistance(pacPos,ghost.getPosition())
            ghostQ.push(dis,dis)
        return float("inf") if ghostQ.isEmpty() else ghostQ.pop()

    def closestCapsule(pacPos,capList):
        capQ=util.PriorityQueue()
        for cap in capList:
            dis=manhattanDistance(pacPos,cap)
            capQ.push(dis,dis)
        return float("inf") if capQ.isEmpty() else capQ.pop()
    
    def baseScore(pacPos,foodList,ghostList,capList,score):
        if closestCapsule(pacPos,capList)<closestGhost(pacPos,ghostList):
            return score*1.5
        if closestFood(pacPos,foodList)<closestCapsule(pacPos,capList):
            return score*1.4
        if closestFood(pacPos,foodList)<closestGhost(pacPos,ghostList):
            return score*1.3
        return score
    
    def scareScore(pacPos,ghostList,score):
        max=score
        for ghost in ghostList:
            s=score
            if ghost.scaredTimer >8:
                dis=manhattanDistance(pacPos,ghost.getPosition())
                if dis<=4:
                    s+=(100-dis*10)
                if(s>max):
                    max=s
        return max
    
    def dodgeScore(pacPos,ghostList,score):
        max=float("-inf")
        for ghost in ghostList:
            s=score
            if ghost.scaredTimer == 0:
                s-=manhattanDistance(ghost.getPosition(), pacPos)
                if(s>max):
                    max=s
        
        return score if max==float("-inf") else max
    
    pacPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood()
    ghostList = currentGameState.getGhostStates()
    capList=currentGameState.getCapsules()
    score=currentGameState.getScore()
    score=baseScore(pacPos,foodList,ghostList,capList,score)
    score=scareScore(pacPos,ghostList,score)
    score=dodgeScore(pacPos,ghostList,score)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
