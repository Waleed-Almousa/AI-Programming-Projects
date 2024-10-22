import random
import copy
import numpy as np


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        move=[]

        drop_phase = True   # TODO: detect drop phase

        moves_c=0

        for i in range(5):
            for j in range(5):
                if state[i][j] !=' ':
                    moves_c+=1
        
        if(moves_c>=8):
            drop_phase=False

        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                if (i+k<5) and (0<=i+k) and (0<=j+l) and 5>j+l and state[i+k][j+l]==' ':
                                    curr=copy.deepcopy(state)
                                    curr[k+i][j+l]=self.my_piece
                                    curr[i][j]= ' '
                                    move=[(i+k, j+l), (i, j)]
        else:
            next_moves=self.succ(state)
            curr_min=-np.inf
            for moves in next_moves:
                curr=copy.deepcopy(state)
                j=moves[1]
                i=moves[0]
                curr[i][j]=self.my_piece
                min=self.min_value(curr, 0)
                if(curr_min<=min):
                    move=[(i, j)]
                    curr_min=min



            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        # move = []
        # (row, col) = (random.randint(0,4), random.randint(0,4))
        # while not state[row][col] == ' ':
        #     (row, col) = (random.randint(0,4), random.randint(0,4))

        # # ensure the destination (row,col) tuple is at the beginning of the move list
        # move.insert(0, (row, col))
        return move
    
    def succ(self, state):
        succ_move=[]
        for i in range(5):
            for j in range(5): 
                if state[i][j]==' ':
                    succ_move.append((i, j))
        
        return succ_move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == state[row + 3][col+3]:
                    return 1 if state[row][col] == self.my_piece else -1



        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check / diagonal wins
        for row in range(2):
            for col in range(3, 5):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col - 1] == state[row + 2][col - 2] == state[row + 3][col - 3]:
                    return 1 if state[row][col] == self.my_piece else -1


        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col] == state[row][col + 1] == state[row + 1][col + 1]:
                    return 1 if state[row][col] == self.my_piece else -1



        return 0 # no winner yet
    
    def max_value(self, state, depth):
        
        if depth == 0:
            return self.heuristic_game_value(state)
        if self.heuristic_game_value(state) == 1 or self.heuristic_game_value(state) == -1:
            return self.heuristic_game_value(state)
        
        
        alpha = -np.inf
        scsrs = self.succ(state)
        for succ in scsrs:
            if alpha < self.min_value(succ[0], depth - 1):
                alpha = self.min_value(succ[0], depth - 1)
            if alpha >= np.inf:
                break
        return alpha
    
    def min_value(self, state, depth):
        if self.heuristic_game_value(state) == 1 or self.heuristic_game_value(state) == -1:
            return self.heuristic_game_value(state)
        if depth == 0:
            return self.heuristic_game_value(state)

        beta=np.inf
        succs = self.succ(state)
        for succ in succs:
            if self.max_value(succ[0], depth - 1) < beta:
                beta = self.max_value(succ[0], depth - 1)
            if -np.inf >= beta:
                break
        return beta
    
    def heuristic_game_value(self, state):

        if self.game_value(state) != 0:
            return self.game_value(state)
        
        val1 = 1
        val2 = 1
        ind1 = []
        ind2 = []
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    ind1.append([i,j])
                elif state[i][j] == self.opp:
                    ind2.append([i,j])
        ind1.sort()
        ind2.sort()

        # compute opponents heuristic
        for ind in ind2:
            # check if 2 in a row 
            if [ind[0]+1, ind[1]] in ind2:
                # check if 3 in a row
                if [ind[1]+2, ind[1]] in ind2:
                    val2=3
                    break
            # check if close to a box
            elif [ind[0]+1, ind[1]] in ind2 or [ind[0]+1,ind[1]+1] in ind2 or [ind[0]+1,ind[1]-1] in ind2 or [ind[0]-1, ind[1]] in ind2:
                    val2 = 2.5
            # else, only 2 in a row
            else:
                val2=2
            # repeat for similar orientations
            if [ind[0]+1,ind[1]+1] in ind2:
                if [ind[0]+2,ind[1]+2] in ind2 or [ind[0]+1,ind[1]-1] in ind2 or [ind[0]+2,ind[1]] in ind2:
                    val2 = 3
                    break
                else:
                    val2 = 2
            if [ind[0],ind[1]+1] in ind2:
                if [ind[0],ind[1]+2] in ind2:
                        val2 = 3
                        break
                elif [ind[0]+1,ind[1]+1] in ind2 or [ind[0]-1,ind[1]+1] in ind2:
                    val2 = 2.5
                else:
                    val2 = 2
            if [ind[0]+2,ind[1]] in ind2:
                if [ind[0]+1,ind[1]-1] in ind2 or [ind[0]+1,ind[1]+1] in ind2:
                    val2 = 3
                    break
                else:
                    val2 = 2
            if [ind[0]+1,ind[1]-1] in ind2:
                if [ind[0]+2,ind[1]-2] in ind2 or [ind[0]+1,ind[1]+1] in ind2 or [ind[0]+2,ind[1]] in ind2:
                    val2 = 3
                    break
                else:
                    val2 = 2



        # compute players current score/heuristic using same method:
        for ind in ind1:
            # check if 2 in a row 
            if [ind[0]+1, ind[1]] in ind1:
                # check if 3 in a row
                if [ind[1]+2, ind[1]] in ind1:
                    val1=3
                    break
            # check if close to a box
            elif [ind[0]+1, ind[1]] in ind1 or [ind[0]+1,ind[1]+1] in ind1 or [ind[0]+1,ind[1]-1] in ind1 or [ind[0]-1, ind[1]] in ind1:
                    val1 = 2.5
            # else, only 2 in a row
            else:
                val1=2
            # repeat for similar orientations
            if [ind[0]+1,ind[1]+1] in ind1:
                if [ind[0]+2,ind[1]+2] in ind1 or [ind[0]+1,ind[1]-1] in ind1 or [ind[0]+2,ind[1]] in ind1:
                    val1 = 3
                    break
                else:
                    val1 = 2
            if [ind[0],ind[1]+1] in ind1:
                if [ind[0],ind[1]+2] in ind1:
                        val1 = 3
                        break
                elif [ind[0]+1,ind[1]+1] in ind1 or [ind[0]-1,ind[1]+1] in ind1:
                    val1 = 2.5
                else:
                    val1 = 2
            if [ind[0]+2,ind[1]] in ind1:
                if [ind[0]+1,ind[1]-1] in ind1 or [ind[0]+1,ind[1]+1] in ind1:
                    val1 = 3
                    break
                else:
                    val1 = 2
            if [ind[0]+1,ind[1]-1] in ind1:
                if [ind[0]+2,ind[1]-2] in ind1 or [ind[0]+1,ind[1]+1] in ind1 or [ind[0]+2,ind[1]] in ind1:
                    val1 = 3
                    break
                else:
                    val1 = 2


        return (val1/(val2 + val1) - 0.5) * 2

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    print(np.array(0))
    main()
   
    
    
