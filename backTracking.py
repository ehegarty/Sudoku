import itertools

blankBoard = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

board = [
    [0, 0, 2, 0, 0, 0, 0, 6, 0],
    [5, 6, 0, 3, 0, 0, 0, 0, 7],
    [0, 0, 8, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 8],
    [6, 3, 0, 0, 0, 9, 0, 1, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 4, 0, 0],
    [9, 1, 0, 0, 0, 3, 0, 8, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 0],
]


def printBoard(board):
    for _ in range(9):
        print("----", end="")
    print("-")
    for i in range(9):
        for j in range(9):
            print("|", board[i][j], end=" ")
        print("|")
        for _ in range(9):
            print("----", end="")
        print("-")


def isPossible(board, row, col, val):
    # sourcery skip: invert-any-all, use-any, use-next
    for j in range(9):
        if board[row][j] == val:
            return False

    for i in range(9):
        if board[i][col] == val:
            return False

    startRow = (row // 3) * 3
    startCol = (col // 3) * 3

    for i, j in itertools.product(range(3), range(3)):
        if board[startRow + i][startCol + j] == val:
            return False

    return True


def solve():
    for i, j in itertools.product(range(9), range(9)):
        if board[i][j] == 0:
            for val in range(1, 10):
                if isPossible(board, i, j, val):
                    board[i][j] = val
                    solve()

                    board[i][j] = 0
            return

    printBoard(board)


solve()
