'''Freecell core code'''
from random import shuffle
from copy import deepcopy
from typing import Iterator, List, Union, Tuple
from queue import PriorityQueue
import itertools

class InvalidCardSuit(Exception):
    '''InvalidCardSuit'''

class InvalidCardValue(Exception):
    '''InvalidCardValue'''

class MoveParserException(Exception):
    '''MoveParserException'''

class MoveValidationException(Exception):
    '''MoveValidationException'''

class Card:
    '''Card Implementation'''
    suits = ['spade', 'heart', 'club', 'diamond']
    values = range(1,14)
    RED = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self, value:int, suit:str) -> None:
        if suit not in self.suits:
            raise InvalidCardSuit(f'Invalid Suit: {suit}')
        if value not in self.values:
            raise InvalidCardValue(f'Invalid Value: {value}')
        self.suit = suit
        self.value = value
        if self.suit in ['heart', 'diamond']:
            self.color = 'red'
        else:
            self.color = 'black'

    def __repr__(self) -> str:
        lookup = {
            1: 'A',
            10: 'T',
            11: 'J',
            12: 'Q',
            13: 'K'
        }
        suit_lookup = {
            'spade': '♠',
            'heart' : '♥',
            'club': '♣',
            'diamond': '♦'
        }
        num = lookup.get(self.value, self.value)

        # return f'{num}{self.suit[0].upper()}'.rjust(3)
        if self.color == 'red':
            return f' {num}{self.RED}{suit_lookup[self.suit]}{self.ENDC}'.rjust(3)
        return f'{num}{suit_lookup[self.suit]}'.rjust(3)

    @classmethod
    def from_string(cls, string_representation:str):
        num_lookup = {
            'A': 1,
            'T': 10,
            'J': 11,
            'Q': 12,
            'K': 13
        }
        num = num_lookup.get(string_representation[0]) or int(string_representation[0])
        suit_lookup = {
            'S': 'spade',
            'H': 'heart',
            'C': 'club',
            'D': 'diamond'
        }
        suit = suit_lookup[string_representation[1]]
        return cls(num, suit)

class Foundation:
    '''A place where you can deposit cards by suit starting with aces'''
    def __init__(self, suit:str) -> None:
        self.suit = suit
        self.stack = []

    def __repr__(self) -> str:
        '''String representation of a foundation'''
        card = self.peek()
        if not card:
            return '[ ]'
        return card.__repr__()

    def peek(self) -> Card:
        '''Return the card at the top of the stack or None'''
        if self.stack:
            return self.stack[-1]
        return None

    def check_withdraw(self) -> bool:
        '''Returns true if you can withdraw a card'''
        return bool(self.stack)

    def check_deposit(self, card:Card) -> bool:
        '''Returns true if you can deposit a card'''
        if card.suit == self.suit:
            if len(self.stack) == 0:
                if card.value == 1:
                    return True
                else:
                    return False
            if card.value == self.peek().value + 1:
                return True
        return False

    def deposit(self, card:Card) -> None:
        '''Adds card to foundation'''
        self.stack.append(card)

    def withdraw(self) -> Card:
        '''Withdraws a card from the top of the foundation stack'''
        return self.stack.pop()

class Move:
    '''Definition of a move'''
    # add precendence and sort moves by precedence?
    def __init__(self, from_location:str, from_index:int, to_location:str, to_index:int, move_depth:int=1) -> None:
        self.from_location = from_location
        self.from_index = from_index
        self.to_location = to_location
        self.to_index = to_index
        self.move_depth = move_depth
        precedence_lookup = {
            'board': {
                'board': 5,
                'foundation': 1,
                'free_space': 7
            },
            'free_space': {
                'board': 3,
                'foundation': 2,
            },
            'foundation': {
                'board': 10
            }
        }
        self.priority = precedence_lookup[self.from_location][self.to_location]

    def __repr__(self) -> str:
        return f'Move from {self.from_location} {self.from_index} to {self.to_location} {self.to_index} depth {self.move_depth}'


class FreeCell:
    '''Freecell game code'''
    n_free_cells = 4
    suits = ['spade', 'heart', 'club', 'diamond']
    parent = None

    def __init__(self, board:List[List[Card]]=None, foundations:List[Foundation]=None, free_spaces:List[Card]=None) -> None:
        self.board = board
        self.foundations = foundations or [Foundation(suit) for suit in self.suits]
        self.free_spaces = free_spaces or [None] * self.n_free_cells
        # if not board:
        #     deck = [] # unnecessary state
        #     for suit in ['spade', 'heart', 'club', 'diamond']:
        #         for value in range(1,14):
        #             deck.append(Card(value, suit))
        #     shuffle(deck)

        #     for i, card in enumerate(deck):
                # self.board[i % 8].append(card)

    def get_moves(self) -> Iterator[Move]:
        '''Get all moves given the current game state'''
        all_moves = []
        # get all free space and foundation moves
        empty_column_indices = []
        for column_index, column in enumerate(self.board):
            if column:
                card = column[-1]

                # see if you can move anything into the foundation
                for foundation_index, foundation in enumerate(self.foundations):
                    if foundation.check_deposit(card):
                        new_move = Move('board', column_index, 'foundation', foundation_index)
                        all_moves.append(new_move)
                # you can add any cards to the free space
                if None in self.free_spaces:
                    # print(self.free_spaces)
                    new_move = Move('board', column_index, 'free_space', self.free_spaces.index(None))
                    all_moves.append(new_move)
            else:
                empty_column_indices.append(column_index)

        # moving between columns
        # move to free column
        # move to other column
        # 1) move one card
        # 2) move stack
        # 3) move substack
        for column_index, column in enumerate(self.board):
            # here is where we determine all the combinations of stacks we can move
            previous_card = None
            for card_index, card in enumerate(reversed(column)):
                move_depth = card_index+1
                if move_depth > self.n_movable_cards():
                    break

                if previous_card and (card.color == previous_card.color or card.value != previous_card.value + 1):
                    break

                for to_col_index, to_column in enumerate(self.board):
                    # don't move cards to same location
                    # TODO: prevent cycles somehow?
                    if column_index == to_col_index:
                        continue

                    # if a column has cards check if you can move the card to it
                    if to_column:
                        to_col_card = to_column[-1]
                        if card.color != to_col_card.color and card.value == to_col_card.value - 1:
                            all_moves.append(Move('board', column_index, 'board', to_col_index, move_depth=move_depth))
                    else:
                        # column is empty
                        all_moves.append(Move('board', column_index, 'board', to_col_index, move_depth=move_depth))
                previous_card = card
                # end of column loop

        for free_space_index, card in enumerate(self.free_spaces):
            if card is None:
                continue
            for to_col_index, to_column in enumerate(self.board):
                if to_column:
                    to_col_card = to_column[-1]
                    if card.color != to_col_card.color and card.value == to_col_card.value - 1:
                        all_moves.append(Move('free_space', free_space_index, 'board', to_col_index))
                else:
                    all_moves.append(Move('free_space', free_space_index, 'board', to_col_index))

        return sorted(all_moves, key=lambda m: (-m.move_depth, m.priority))

    def execute_move(self, move:Move) -> None:
        if move.from_location == 'board' and move.to_location == 'board':
            self.board[move.to_index].extend(self.board[move.from_index][-move.move_depth:])
            self.board[move.from_index] = self.board[move.from_index][:-move.move_depth:]
        elif move.from_location == 'board' and move.to_location == 'free_space':
            self.free_spaces[move.to_index] = self.board[move.from_index][-move.move_depth]
            self.board[move.from_index] = self.board[move.from_index][:-move.move_depth:]
        elif move.from_location == 'board' and move.to_location == 'foundation':
            self.foundations[move.to_index].deposit(self.board[move.from_index][-move.move_depth])
            self.board[move.from_index] = self.board[move.from_index][:-move.move_depth:]
        elif move.from_location == 'free_space' and move.to_location == 'board':
            self.board[move.to_index].append(self.free_spaces[move.from_index])
            self.free_spaces[move.from_index] = None
        elif move.from_location == 'free_space' and move.to_location == 'foundation':
            self.foundations[move.to_index].deposit(self.free_spaces[move.from_index][-move.move_depth])
            self.free_spaces[move.from_index] = None
        elif move.from_location == 'foundation' and move.to_location == 'board':
            self.board[move.from_index][-move.move_depth] = self.foundations[move.from_index].withdraw()
        elif move.from_location == 'foundation' and move.to_location == 'free_space':
            self.free_spaces[move.to_index] = self.foundations[move.from_index].withdraw()
        else:
            raise Exception(f'Invalid Move Type {move.move_type}')

    def check_win(self) -> bool:
        '''Returns true if you have won the game'''
        return all(foundation.peek() and foundation.peek().value == 13 for foundation in self.foundations)

    def solve(self) -> bool:
        '''Solve the game'''
        visited = set()
        # counter is used to break ties in priority queue
        counter = itertools.count()
        pqueue = PriorityQueue()
        pqueue.put((1, 1, self.copy()))
        count = 0
        while pqueue:
            _, _, state = pqueue.get()
            state_hash = state.generate_hash()

            if state.check_win():
                state.show()
                solution = []
                parent = state.parent
                while parent:
                    solution.append(parent)
                    parent = parent.parent
                return list(reversed(solution))

            if state_hash in visited:
                continue

            count += 1
            if count % 1000 == 0:
                print(count)
                state.show()

            visited.add(state_hash)
            for move in state.get_moves():
                clone = state.copy()
                clone.parent = state
                clone.execute_move(move)
                clone_hash = clone.generate_hash()
                if clone_hash not in visited:
                    pqueue.put((-clone.score(), -next(counter), clone))
        return False

    def n_movable_cards(self) -> int:
        '''Returns the number of moveable cards given a board'''
        num_free_columns = sum(map(lambda col: len(col)==0, self.board))
        num_free_spaces = self.free_spaces.count(None)
        return (2**num_free_columns) * (num_free_spaces + 1)

    def generate_hash(self) -> str:
        # return hashlib.sha256((str(self.free_spaces) + str(self.foundations) + str(self.board)).encode('utf-8')).hexdigest()
        return hash(str(self.free_spaces) + str(self.foundations) + str(self.board))

    def show(self) -> None:
        '''Print the board'''
        print(''.join('[ ]' if space is None else str(space) for space in self.free_spaces) + '|' + ''.join(str(f) for f in self.foundations))
        print('='*25)
        max_len = 0
        for column in self.board:
            max_len = max(max_len, len(column))
        for row_index in range(max_len):
            row = []
            for column in self.board:
                if len(column) - 1 < row_index:
                    row.append(' ' * 3)
                else:
                    row.append(str(column[row_index]))
            print(''.join(row))

    # def __deepcopy__(self, memodict={}):
    #     copy_object = FreeCell()
    #     copy_object.board = self.board.copy()
    #     copy_object.foundations = self.foundations.copy()
    #     copy_object.free_spaces = self.free_spaces.copy()
    #     return copy_object

    def copy(self):
        # return deepcopy(self)
        # return FreeCell(board=ujson.loads(ujson.dumps(self.board)), foundations=ujson.loads(ujson.dumps(self.foundations)), free_spaces=ujson.loads(ujson.dumps(self.free_spaces)))
        return FreeCell(board=deepcopy(self.board), foundations=deepcopy(self.foundations), free_spaces=deepcopy(self.free_spaces))

    def parse_move(self, move_string:str) -> Move:
        '''Parses a move from command line input'''
        if len(move_string) != 2:
            raise MoveParserException(f'Move {move_string} not recognized')

        # array unpacking works with strings
        from_char, to_char = move_string

        board_lookup = {
            'a': 0,
            's': 1,
            'd': 2,
            'f': 3,
            'j': 4,
            'k': 5,
            'l': 6,
            ';': 7,
        }

        free_space_lookup = {
            'q': 0,
            'w': 1,
            'e': 2,
            'r': 3
        }

        foundation_lookup = {
            'u': 0,
            'i': 1,
            'o': 2,
            'p': 3
        }

        if from_char in board_lookup:
            from_location = 'board'
            from_index = board_lookup[from_char]
        elif from_char in free_space_lookup:
            from_location = 'free_space'
            from_index = free_space_lookup[from_char]
        elif from_char in foundation_lookup:
            from_location = 'foundation'
            from_index = foundation_lookup[from_char]
        else:
            raise MoveParserException(f'To character {from_char} not recognized')

        if to_char in board_lookup:
            to_location = 'board'
            to_index = board_lookup[to_char]
        elif to_char in free_space_lookup:
            to_location = 'free_space'
            to_index = free_space_lookup[to_char]
        elif to_char in foundation_lookup:
            to_location = 'foundation'
            to_index = foundation_lookup[to_char]
        else:
            raise MoveParserException(f'To character {to_char} not recognized')

        move_depth = 1

        if from_location == 'board' and to_location == 'board':
            try:
                raw = input('How many cards would you like to move?')
                move_depth = int(raw)
            except:
                raise MoveParserException(f'Could not parse move depth {raw}')

        return Move(from_location, from_index, to_location, to_index, move_depth=move_depth)

    def check_move(self, move:Move, give_reason:bool=False) -> Union[bool, Tuple[bool, str]]:
        '''Checks if the move is valid, returns true if it is'''
        is_valid = False
        reason = ''
        # Cases:
        # Case1: Board to Board - partial
        # Case4: Free to Board - covered
        # Case7: Foundation to Board -
        # Case2: Board to Frees - covered
        # Case5: Free to Free (not needed) - covered
        # Case8: Foundation to Free - covered
        # Case3: Board to foundation - covered
        # Case6: Free to Foundation - covered
        # Foundation to free

        if move.from_location == move.to_location and move.from_index == move.to_index:
            reason = 'Move from and to location are the same'
        elif move.move_depth > self.n_movable_cards():
            reason = f'Trying to move {move.move_depth} cards but you can only move {self.n_movable_cards()} cards'
        # Case 1
        elif move.from_location == 'board' and move.to_location == 'board':
            card1 = self.board[move.from_index][-move.move_depth]
            card2 = self.board[move.to_index][-1]
            if card1.color != card2.color and card1.value == card2.value - 1:
                is_valid = True
            else:
                reason = f'Cannot move {card1} on top of {card2}'
        elif move.to_location == 'free_space':
            if move.from_location == 'free_space':
                reason = 'Useless Move'
            elif self.free_spaces[move.to_index]:
                reason = 'Free space occupied'
            else:
                is_valid = True
        # Case 2
        elif move.to_location == 'foundation':
            if move.from_location == 'board':
                card = self.board[move.from_index][-1]
            elif move.from_location == 'free_space':
                card = self.free_spaces[move.from_index]
            else:
                reason = 'Cannot move between foundations'
            if not reason:
                if self.foundations[move.to_index].check_deposit(card):
                    is_valid = True
                else:
                    reason = 'Cannot deposit card into foundation'

        elif move.from_location == 'free_space' and move.to_location == 'board':
            card1 = self.free_spaces[move.from_index]
            if self.board[move.to_index] or card1.color == self.board[move.to_index][-1].color or card1.value != self.board[move.to_index][-1].value - 1:
                reason = f'Cannot place {str(card1)} on top of {str(self.board[move.to_index][-1])}'
            else:
                is_valid = True

        elif move.from_location == 'foundation' and move.to_location == 'board':
            card1 = self.foundations[move.from_index].peek()
            card2 = self.board[move.to_index][-1]
            if self.board[move.to_index] or card1.color == self.board[move.to_index][-1].color or card1.value != self.board[move.to_index][-1].value - 1:
                reason = 'Cannot Place card from foundation on board'
            else:
                is_valid = True
        else:
            reason = f'Cannot move from {move.from_index} to {move.to_index}'

        if give_reason:
            return is_valid, reason
        return is_valid

    def score(self) -> int:
        '''Generate a score for the current game state'''
        score = 0
        for foundation in self.foundations:
            top_card = foundation.peek()
            if top_card:
                score += 100*top_card.value

        for free_space in self.free_spaces:
            if free_space is None:
                score += 10

        for column in self.board:
            if column and column[0].value == 13:
                score += 5
        return score

    def play(self) -> None:
        '''Play a command line version of the game'''
        n_moves = 0
        while not self.check_win():
            for move in self.get_moves():
                print(move)
            self.show()
            move_str = input('Execute Move: ')
            try:
                move = self.parse_move(move_str)
            except MoveParserException as error:
                print(str(error))
                continue

            valid_move, reason = self.check_move(move, give_reason=True)

            if not valid_move:
                print(reason)
                continue

            n_moves += 1
            self.execute_move(move)

        print(f'Congrats you won in {n_moves} moves!')

if __name__ == "__main__":
    with open('boards/board1.txt') as fh:
        board_data = fh.read()
    board = [[] for _ in range(8)]
    cards = set()
    for line in board_data.splitlines():
        for column_index, card_str in enumerate(line.split(' ')):
            board[column_index].append(Card.from_string(card_str))
            cards.add(str(card_str))
    if len(cards) != 52:
        raise Exception(f'Bad deck size: {len(cards)}')

    game = FreeCell(board=board)

    # game.play()
    game.show()
    solution = game.solve()
    print('SOLUTION', len(solution))
    for state in solution:
        state.show()
        print('*************************')
        _ = input()
