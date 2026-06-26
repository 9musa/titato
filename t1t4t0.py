import random
board = [0 for _ in range(9)]
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.5
nEpisodes = 1000

def showBoard(board):
    symbols = {
        1: "X",
        -1: "O",
        0: "."
    }
    #prints each row line by line
    for i in range(0, len(board), 3):
        print(f"{symbols[board[i]]} {symbols[board[i+1]]} {symbols[board[i+2]]}")

def updBoard(board, pos, plyr):
    if board[pos] == 0:
        board[pos] = plyr
    return board

def availMoves(board):
    availPos = []
    #if position available, adds to list
    for i in range(len(board)):
        if board[i] == 0:
            availPos.append(i)
    #returns list
    return availPos

#updates Q dictionary
def updateQ(state, action, reward, nextState, terminal):
    #ensure current state exists
    if state not in Q:
        Q[state] = [0.0 for _ in range(9)]
    #ensure next state exists
    if nextState not in Q:
        Q[nextState] = [0.0 for _ in range(9)]
    #terminal checks
    if not terminal:
        #update q vals
        bestNextQ = max(Q[nextState])
        target = reward + (gamma * bestNextQ)
    else: target = reward
    Q[state][action] += alpha * (target - Q[state][action])

#check for current game state
def getGameState(board):
    winConditions = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6], [0, 3, 6], [1, 4, 7], [2, 5, 8]]
    for a, b, c in winConditions:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None

#returns possible action
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
showBoard(board)

def trainAgent(n):
    epsilon = 0.5
    terminal = [1, 0, -1]
    for _ in range(n):
        board = [0 for _ in range(9)]
        agent = -1
        opp = 1
        while True:
            state = tuple(board)
            aiAction = chooseAction(board, epsilon)
            board = updBoard(board, aiAction, agent)
            result = getGameState(board)
            if result in terminal:
                if result == agent:
                    reward = 1
                elif result == opp:
                    reward = -1
                else: reward = 0
                nextState = tuple(board)
                updateQ(state, aiAction, reward, nextState, True)
                break
            else:
                oppAction = random.choice(availMoves(board))
                board = updBoard(board, oppAction, opp)
                nextState = tuple(board)
                result = getGameState(board)
                if result in terminal:
                    if result == opp:
                        reward = -1
                    elif result == 0:
                        reward = 0
                    else:
                        raise RuntimeError("Impossible game state after opponent move.")
                    updateQ(state, aiAction, reward, nextState, True)
                    break
                else:
                    updateQ(state, aiAction, 0, nextState, False)
        epsilon = max(0.05, epsilon * 0.99999)
    print(f"AI trained with {len(Q)} different board states")
trainAgent(500000)

def play():
    board = [0 for _ in range(9)]
    aiMove = chooseAction(board, 0.0)
    board = updBoard(board, aiMove, -1)
    showBoard(board)
    while getGameState(board) == None:
        move = int(input())
        board = updBoard(board, move, 1)
        showBoard(board)
        if getGameState(board) != None:
            break
        state = tuple(board)
        print(state in Q)
        AIMove = chooseAction(board, 0.0)
        board = updBoard(board, AIMove, -1)
        showBoard(board)
    winner = getGameState(board)
    if winner == -1:
        print("You lose!")
    elif winner == 1:
        print("You win!")
    else:
        print("It's a draw!")
play()
                
