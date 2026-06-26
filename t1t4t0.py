import random
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager
board = [0 for _ in range(9)]
winConditions = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6], [0, 3, 6], [1, 4, 7], [2, 5, 8]]
gameOver = False
gameMode = ""
currentPlayer = 1
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.5
gamesPlayed = 0
wins = 0
draws = 0
losses = 0

def renderBoard(board):
    symbols = {
        1: "X",
        -1: "O",
        0: ""
    }
    for i in range(len(board)):
        buttons[i].config(text=symbols[board[i]])

def updBoard(board, pos, plyr):
    if board[pos] == 0:
        board[pos] = plyr
    return board

def playMove(index):
    global currentPlayer
    global board
    if gameOver: return
    if board[index] != 0: return

    board = updBoard(board, index, currentPlayer)
    renderBoard(board)

    currentPlayer *= -1

    winner = getGameState(board)
    if winner is not None:
        endGame(winner)
        return
    if gameMode == "Agent":
        aiMove = chooseAction(board, 0.0)
        board = updBoard(board, aiMove, -1)
        renderBoard(board)

        winner = getGameState(board)
        if winner is not None:
            endGame(winner)
            return
        currentPlayer *= -1
    
def endGame(winner):
    global gameOver
    gameOver = True
    line = getWinningLine(board)
    if winner == 1:
        gameLabel.config(text="Victory.")
    elif winner == -1:
        gameLabel.config(text="Loss.")
    else: gameLabel.config(text="Stalemate.")
    if line is not None:
        highlightLine(line)

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
    for a, b, c in winConditions:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None

def getWinningLine(board):
    for a, b, c in winConditions:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return (a, b, c)
    return None

def highlightLine(line):
    for i in range(len(board)):
        if i not in line:
            buttons[i].config(foreground="#cfcfcf")

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

def trainAgent(n):
    global gamesPlayed, wins, draws, losses
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
        result = getGameState(board)
        if result == -1: wins += 1
        elif result == 1: losses += 1
        else: draws += 1
        gamesPlayed += 1
        epsilon = max(0.05, epsilon * 0.99999)
    print(f"AI trained with {len(Q)} different board states", gamesPlayed, wins, draws, losses)

def playAgent():
    global gameMode
    gameMode = "Agent"
    showGame()

def playVersus():
    global gameMode
    gameMode = "Versus"
    showGame()

def showGame():
    loadingFrame.pack_forget()
    gameFrame.pack(fill="both", expand=True)
    startGame()

def startGame():
    global board, currentPlayer, gameOver
    board = [0 for _ in range(9)]
    currentPlayer = 1
    gameOver = False
    if gameMode == "Agent":
        aiMove = chooseAction(board, 0.0)
        board = updBoard(board, aiMove, -1)
    renderBoard(board)

def playGame():
    global board
    board = [0 for _ in range(9)]
    aiMove = chooseAction(board, 0.0)
    board = updBoard(board, aiMove, -1)
    renderBoard(board)
    while getGameState(board) == None:
        move = int(input())
        board = updBoard(board, move, 1)
        renderBoard(board)
        if getGameState(board) != None:
            break
        state = tuple(board)
        print(state in Q)
        AIMove = chooseAction(board, 0.0)
        board = updBoard(board, AIMove, -1)
        renderBoard(board)
    winner = getGameState(board)
    if winner == -1:
        print("You lose!")
    elif winner == 1:
        print("You win!")
    else:
        print("It's a draw!")

#GUI
root = tkinter.Tk()
root.title("T1T4T0")
root.geometry("500x600")
root.configure(bg="#000000")
loadingFrame = tkinter.Frame(root, background="#000000")
gameFrame = tkinter.Frame(root, background="#000000")
logo = tkinter.PhotoImage(file="./assets/T1T4T0.png")
logoLabel = tkinter.Label(loadingFrame, image=logo, bg="#000000", borderwidth=0, highlightthickness=0)
logoLabel.pack(pady=20)

statusLabel = tkinter.Label(
    loadingFrame,
    text="Training...",
    background="#000000",
    foreground="#ffffff",
    font=("VT323", 18)
)
statusLabel.pack(pady=10)
gameLabel = tkinter.Label(
    gameFrame,
    text="",
    background="#000000",
    foreground="#ffffff",
    font=("VT323", 30)
)
gameLabel.pack(pady=10)

fontPath = "./assets/VT323-Regular.ttf"
font_manager.fontManager.addfont(fontPath)
vt323 = font_manager.FontProperties(fname=fontPath)
figure = Figure(figsize=(5, 3), dpi=100, facecolor="#000000")
plot = figure.add_subplot(111)
plot.set_facecolor("#000000") 
xData = []
yData = []

plot.set_title("Agent Win Rate", color="#ffffff", fontproperties=vt323)
plot.set_xlabel("Games Played", color="#ffffff", fontproperties=vt323)
plot.set_ylabel("Win Rate", color="#ffffff", fontproperties=vt323)
plot.tick_params(colors="#ffffff")
for label in plot.get_xticklabels():
    label.set_fontproperties(vt323)
for label in plot.get_yticklabels():
    label.set_fontproperties(vt323)
plot.spines["bottom"].set_color("white")
plot.spines["left"].set_color("white")
plot.spines["top"].set_visible(False)
plot.spines["right"].set_visible(False)
canvas = FigureCanvasTkAgg(figure, master=loadingFrame)
canvas.get_tk_widget().pack(pady=10)
line, = plot.plot([], [], color="#ffffff", linewidth=2)

modeFrame = tkinter.Frame(loadingFrame, background="#000000")
modeFrame.pack(pady=15)
pvaButton = tkinter.Button(
    modeFrame,
    text="PLAY AGENT",
    font=("VT323", 16),
    bg="black",
    fg="white",
    activebackground="black",
    activeforeground="white",
    borderwidth=1,
    highlightthickness=0,
    padx=25,
    pady=8,
    cursor="hand2",
    command=playAgent
)
pvpButton = tkinter.Button(
    modeFrame,
    text="PLAY VERSUS",
    font=("VT323", 16),
    bg="black",
    fg="white",
    activebackground="black",
    activeforeground="white",
    borderwidth=1,
    highlightthickness=0,
    padx=25,
    pady=8,
    cursor="hand2",
    command=playVersus
)

def trainChunks(chunkSize, totalGames):
    global gamesPlayed
    print(gamesPlayed, wins, draws, losses)
    if gamesPlayed >= totalGames:
        print("Training complete!")
        statusLabel.config(text="Agent Ready")
        pvaButton.pack(side="left", pady=15, in_=modeFrame)
        pvpButton.pack(side="left", pady=15, in_=modeFrame)
        return
    trainAgent(chunkSize)
    xData.append(gamesPlayed)
    yData.append(wins / gamesPlayed)
    line.set_data(xData, yData)
    plot.relim()
    plot.autoscale_view()
    canvas.draw()
    root.after(1, lambda: trainChunks(chunkSize, totalGames))

boardFrame = tkinter.Frame(gameFrame, bg="#000000", width=300, height=300)
boardFrame.pack(pady=20)
boardFrame.grid_propagate(False)
boardFrame.configure(width=300, height=300)
buttons = []
for i in range(3):
    boardFrame.grid_rowconfigure(i, weight=1)
    boardFrame.grid_columnconfigure(i, weight=1)
for row in range(3):
    for col in range(3):
        index = row * 3 + col
        button = tkinter.Button(
            boardFrame,
            text="",
            background="#000000",
            foreground="#ffffff",
            activebackground="#000000",
            activeforeground="#ffffff",
            borderwidth=1,
            highlightthickness=0,
            cursor="hand2",
            width=2,
            height=2,
            font=("VT323", 40),
            command=lambda i=index: playMove(i)
        )
        button.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
        buttons.append(button)
loadingFrame.pack(fill="both", expand=True)
trainChunks(100, 3000)
root.mainloop()
                
