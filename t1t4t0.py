import random
board = [0 for _ in range(9)]
Q = {}
alpha = 0.001
gamma = 0.9
epsilon = 0.5
nEpisodes = 1000

def showBoard(board):
    #prints each row line by line
    for i in range(0, len(board), 3):
        print(board[i : i + 3])

def availMoves(board):
    availPos = []
    #if position available, adds to list
    for i in range(len(board)):
        if board[i] == 0:
            availPos.append(i)
    #returns list
    return availPos

def updateQ(state, action, reward, nextState):
    #ensure current state exists
    if state not in Q:
        Q[state] = [0.0 for _ in range(9)]
    #ensure next state exists
    if nextState not in Q:
        Q[nextState] = [0.0 for _ in range(9)]
    #update q vals
    bestNextQ = max(Q[nextState])
    Q[state][action] += alpha * (reward + (gamma * bestNextQ) - Q[state][action])

def chooseAction(board, epsilon):
    #current state
    state = tuple(board)
    legalActions = availMoves(board)
    if state not in Q:
        Q[state] = [0.0 for _ in range(9)]
    #exploration
    if random.random() < epsilon:
        #random choice with no correlation to weightage
        return random.choice(legalActions)
    #exploitation
    else:
        #list of q val of every free spot
        qVals = [Q[state][action] for action in legalActions]
        #highest possible q val in current state
        maxQ = max(qVals)
        #list containing index position of every move with max q value
        bestActions = [action for action in legalActions if Q[state][action] == maxQ]
        #randomly picks a move since all have equal weightage
        return random.choice(bestActions)
