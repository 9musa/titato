import sys
import os
import random
import tkinter
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager

# initialization
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

# resolve file path conflicts
def resourcePath(relative):
    # unpack assets
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.abspath("."), relative)

# render board in GUI
def renderBoard(board):
    symbols = {
        1: "X",
        -1: "O",
        0: ""
    }
    for i in range(len(board)):
        buttons[i].config(text=symbols[board[i]])

# updates board with move
def updBoard(pos, plyr):
    global board
    if board[pos] == 0:
        board[pos] = plyr

# helper function for agent opening
def agentOpen():
    global board, currentPlayer
    agentMove = chooseAction(board, 0.0)
    updBoard(agentMove, -1)
    renderBoard(board)
    currentPlayer = 1

# takes button inputs and updates board and turn
def playMove(index):
    global currentPlayer
    global board
    if gameOver: return
    if gameMode == "Agent" and currentPlayer != 1: return
    if board[index] != 0: return

    updBoard(index, currentPlayer)
    renderBoard(board)

    currentPlayer *= -1

    winner = getGameState(board)
    if winner is not None:
        endGame(winner)
        return
    if gameMode == "Agent":
        aiMove = chooseAction(board, 0.0)
        updBoard(aiMove, -1)
        renderBoard(board)

        winner = getGameState(board)
        if winner is not None:
            endGame(winner)
            return
        currentPlayer *= -1

# end game routine
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
    pAgainButton.pack(side="left", padx=15, pady=15)

# finds all available moves and returns list
def availMoves(board):
    availPos = []
    #if position available, adds to list
    for i in range(len(board)):
        if board[i] == 0:
            availPos.append(i)
    return availPos

# updates Q dictionary
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

# check for current game state
def getGameState(board):
    for a, b, c in winConditions:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None

# helper function used in end game routine: return winning line
def getWinningLine(board):
    for a, b, c in winConditions:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return (a, b, c)
    return None

# helper function used in end game routine: highlighting
def highlightLine(line):
    for i in range(len(board)):
        if i not in line:
            buttons[i].config(foreground="#cfcfcf")

# returns possible action
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

# train agent function
def trainAgent(n):
    global gamesPlayed, wins, draws, losses, board
    epsilon = 0.5
    terminal = [1, 0, -1]
    #one episode
    for _ in range(n):
        agent = -1
        opp = 1
        #chooses first turn
        firstPlayer = random.choice([agent, opp])
        turn = firstPlayer
        board = [0 for _ in range(9)]
        #initialise
        lastState = None
        lastAgentAction = None
        #game loop
        while True:
            #agent turn
            if turn == agent:
                #saving last state
                lastState = tuple(board)
                lastAgentAction = chooseAction(board, epsilon)
                updBoard(lastAgentAction, agent)
                result = getGameState(board)
                #calculating rewards
                if result in terminal:
                    if result == agent:
                        reward = 1
                        gamesPlayed += 1
                        wins += 1
                    elif result == 0:
                        reward = 0
                        gamesPlayed += 1
                        draws += 1
                    #handling potential bug
                    else: raise RuntimeError("Impossible game state after AI move.")
                    nextState = tuple(board)
                    #guard incase AI has first turn
                    if lastState is not None:
                        updateQ(lastState, lastAgentAction, reward, nextState, True)
                        break
            #opponent turn
            else:
                oppAction = random.choice(availMoves(board))
                updBoard(oppAction, opp)
                result = getGameState(board)
                nextState = tuple(board)
                #calculating reward
                if result in terminal:
                    if result == opp:
                        reward = -1
                        gamesPlayed += 1
                        losses += 1
                    elif result == 0:
                        reward = 0
                        gamesPlayed += 1
                        draws += 1
                    #handling potential bug
                    else: raise RuntimeError("Impossible game state after AI move.")
                    if lastState is not None:
                        updateQ(lastState, lastAgentAction, reward, nextState, True)
                        break
                else:
                    #guard incase opponent first turn
                    if lastState is not None:
                        updateQ(lastState, lastAgentAction, 0, nextState, False)
            #change turns
            turn *= -1
        #epsilon decay
        epsilon = max(0.05, epsilon * 0.99999)
    print(f"AI trained with {len(Q)} different board states. Games played: {gamesPlayed}, Games won: {wins}, Games lost: {losses} Games drawn: {draws}")

# play agent button command
def playAgent():
    global gameMode
    gameMode = "Agent"
    showGame()

# play versus button command
def playVersus():
    global gameMode
    gameMode = "Versus"
    showGame()

# hide graph and start game
def showGame():
    loadingFrame.pack_forget()
    gameFrame.pack(fill="both", expand=True)
    startGame()

# initialize game
def startGame():
    global board, currentPlayer, gameOver
    board = [0 for _ in range(9)]
    currentPlayer = 1
    gameOver = False
    gameLabel.config(text="")
    pAgainButton.pack_forget()
    for button in buttons:
        button.config(
            background="#000000",
            foreground="#ffffff"
        )
    renderBoard(board)
    if gameMode == "Agent":
        agentStarts = random.choice([True, False])
        if agentStarts:
            currentPlayer = -1
            root.after(400, agentOpen)
    renderBoard(board)

# back button
def backToMenu():
    gameFrame.pack_forget()
    loadingFrame.pack(fill="both", expand=True)

# GUI
root = tkinter.Tk()
root.title("T1T4T0")
w, h = 500, 600
x = (root.winfo_screenwidth() - w) // 2
y = (root.winfo_screenheight() - h) // 2
root.geometry(f"{w}x{h}+{x}+{y}")
root.configure(bg="#000000")

# icons
if sys.platform == "win32":
    root.iconbitmap(resourcePath("assets/logo.ico"))
else:
    #potential cross platform
    root.iconphoto(True, tkinter.PhotoImage(file=resourcePath("assets/logo.png")))

loadingFrame = tkinter.Frame(root, background="#000000") # graph frame
gameFrame = tkinter.Frame(root, background="#000000") # game frame
logo = tkinter.PhotoImage(file=resourcePath("assets/T1T4T0.png"))
logoLabel = tkinter.Label(loadingFrame, image=logo, bg="#000000", borderwidth=0, highlightthickness=0)
logoLabel.pack(pady=10)

# graph status
statusLabel = tkinter.Label(
    loadingFrame,
    text="Training...",
    background="#000000",
    foreground="#ffffff",
    font=("VT323", 18)
)
statusLabel.pack(pady=10)
# victory, defeat, stalemate
gameLabel = tkinter.Label(
    gameFrame,
    text="",
    background="#000000",
    foreground="#ffffff",
    font=("VT323", 30)
)
gameLabel.pack(pady=10)
# footer
loadingFooter = tkinter.Label(
    loadingFrame,
    text="© 2026 Benevolence Labs",
    bg="#000000",
    fg="#4a4a4a",
    font=("VT323", 10)
)
loadingFooter.pack(side="bottom", anchor="e", padx=10, pady=8)
# footer
gameFooter = tkinter.Label(
    gameFrame,
    text="© 2026 Benevolence Labs",
    bg="#000000",
    fg="#4a4a4a",
    font=("VT323", 10)
)
gameFooter.pack(side="bottom", anchor="e", padx=10, pady=8)

# font
fontPath = resourcePath("assets/VT323-Regular.ttf")
font_manager.fontManager.addfont(fontPath)
vt323 = font_manager.FontProperties(fname=fontPath)
# graph
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
# progress bar
style = ttk.Style()
style.theme_use("clam")
style.configure(
    "custom.Horizontal.TProgressbar",
    troughcolor = "#000000",
    background = "#ffffff",
    bordercolor = "#ffffff",
    lightcolor="#ffffff",
    darkcolor = "#ffffff"
)
progress = ttk.Progressbar(
    loadingFrame,
    style = "custom.Horizontal.TProgressbar",
    mode = "determinate",
    length = "300",
)
progress.pack()

# graph navigation
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

# graph update function
def trainChunks(chunkSize, totalGames):
    global gamesPlayed
    print(gamesPlayed, wins, draws, losses)
    if gamesPlayed >= totalGames:
        print("Training complete!")
        statusLabel.config(text="Agent Ready")
        progress["value"] = 100
        progress.pack_forget()
        pvaButton.pack(side="left", padx=15, pady=15, in_=modeFrame)
        pvpButton.pack(side="left", padx=15, pady=15, in_=modeFrame)
        return
    trainAgent(chunkSize)
    xData.append(gamesPlayed)
    yData.append(wins / gamesPlayed)
    line.set_data(xData, yData)
    plot.relim()
    plot.autoscale_view()
    canvas.draw()
    progress["value"] = (gamesPlayed / totalGames) * 100
    root.after(1, lambda: trainChunks(chunkSize, totalGames))

# board GUI ini
boardFrame = tkinter.Frame(gameFrame, bg="#000000", width=300, height=300)
boardFrame.pack(pady=20)
boardFrame.grid_propagate(False)
boardFrame.configure(width=300, height=300)
buttonFrame = tkinter.Frame(gameFrame, bg="black")
buttonFrame.pack(pady=15)
# board navigation
pAgainButton = tkinter.Button(
    buttonFrame,
    text="PLAY AGAIN",
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
    command=startGame
)
backButton = tkinter.Button(
    buttonFrame,
    text="BACK",
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
    command=backToMenu
)
backButton.pack(side="left", padx=15, pady=15)

# board GUI
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

# initialize with graph
loadingFrame.pack(fill="both", expand=True)
trainChunks(100, 50000)
root.mainloop()
                
