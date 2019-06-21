# -*- coding: utf-8 -*-
import torch
from collections import defaultdict

class ChessMan(object):

	def __init__(self, player, init_pos):
		self.player = player
		self.pos = init_pos
		self.chess_type = None
		self.name = None
		self.allowed_moves = None
		self.pos_range = (0,0,8,9)

	def state_dict(self):
		return {'player':self.player, 'chess_type':self.chess_type, 'pos':self.pos}

	def __repr__(self):
		return '%d_%d %s %s' % (self.pos[0],  self.pos[1], self.player, self.chess_type)

	def can_move_to(self, board, pos):
		if pos[0] < self.pos_range[0] or pos[1] < self.pos_range[1]: return False
		if pos[0] > self.pos_range[2] or pos[1] > self.pos_range[3]: return False
		move_x = pos[0] - self.pos[0]
		move_y = pos[1] - self.pos[1]
		if self.allowed_moves is None: return True
		for mv in self.allowed_moves:
			if move_x == mv[0] and move_y == mv[1]: return True
		return False

	def clone(self):
		chess = self.__class__(self.player, self.pos)
		chess.chess_type = self.chess_type
		chess.name = self.name
		chess.allowed_moves = self.allowed_moves
		chess.pos_range = self.pos_range
		return chess

def move_direction(init_pos, dest_pos):
	# no move
	if dest_pos[0] == init_pos[0] and dest_pos[1] == init_pos[1]:
		return None
	if dest_pos[0] == init_pos[0]:
		return (0,1) if dest_pos[1] > init_pos[1] else (0,-1)
	if dest_pos[1] == init_pos[1]:
		return (1,0) if dest_pos[0] > init_pos[0] else (-1,0)
	# move is not straight
	return None

def count_chess_on_path(board, init_pos, dest_pos, move_dir):
	x, y = init_pos[0] + move_dir[0], init_pos[1] + move_dir[1]
	count = 0
	while not (x == dest_pos[0] and y == dest_pos[1]):
		if board.get((x, y)) is not None: count += 1
		x, y = x + move_dir[0], y + move_dir[1]
	return count

class King(ChessMan):

	def __init__(self, player, init_pos):
		super(King, self).__init__(player, init_pos)
		self.chess_type='king'
		self.name='帅', 
		self.allowed_moves=((-1,0),(1,0),(0,-1),(0,1))
		self.pos_range=(3,0,5,2)

	def can_move_to(self, board, dest_pos):
		if super(King, self).can_move_to(board, dest_pos):
			return True
		move_dir = move_direction(self.pos, dest_pos)
		if move_dir is None: return False
		if move_dir[0] == 0 and move_dir[1] == 1:
			chess_in_middle = count_chess_on_path(board, self.pos, dest_pos, move_dir)
			if chess_in_middle == 0:
				dest = board.get(dest_pos)
				if dest is not None:
					return dest.chess_type == self.chess_type
		return False

class Rock(ChessMan):

	def __init__(self, player, init_pos):
		super(Rock, self).__init__(player, init_pos)
		self.chess_type='rock'
		self.name='车', 

	def can_move_to(self, board, dest_pos):
		if not super(Rock, self).can_move_to(board, dest_pos): return False
		move_dir = move_direction(self.pos, dest_pos)
		if move_dir is None: return False
		chess_in_middle = count_chess_on_path(board, self.pos, dest_pos, move_dir)
		if chess_in_middle > 0: return False
		dest = board.get(dest_pos)
		if dest is None: return True
		return dest.player != self.player

class Cannon(ChessMan):

	def __init__(self, player, init_pos):
		super(Cannon, self).__init__(player, init_pos)
		self.chess_type='cannon'
		self.name='炮', 

	def can_move_to(self, board, dest_pos):
		if not super(Cannon, self).can_move_to(board, dest_pos): return False
		move_dir = move_direction(self.pos, dest_pos)
		if move_dir is None: return False
		chess_in_middle = count_chess_on_path(board, self.pos, dest_pos, move_dir)
		dest = board.get(dest_pos)
		if chess_in_middle == 0:
			return dest is None
		elif chess_in_middle == 1:
			if dest is None: return False
			return dest.player != self.player
		return False

class Knight(ChessMan):

	def __init__(self, player, init_pos):
		super(Knight, self).__init__(player, init_pos)
		self.chess_type='knight'
		self.name='马', 
		self.allowed_moves=((-2,-1),(-2,1),(-1,-2),(-1,2),(2,-1),(2,1),(1,-2),(1,2))

	def can_move_to(self, board, dest_pos):
		if not super(Knight, self).can_move_to(board, dest_pos): return False
		move_x = dest_pos[0] - self.pos[0]
		move_y = dest_pos[1] - self.pos[1]
		if move_x == -2:
			return board.get((self.pos[0] - 1, self.pos[1])) is None
		if move_x == 2:
			return board.get((self.pos[0] + 1, self.pos[1])) is None
		if move_y == -2:
			return board.get((self.pos[0], self.pos[1] - 1)) is None
		if move_y == 2:
			return board.get((self.pos[0], self.pos[1] + 1)) is None
		return False

class Guard(ChessMan):

	def __init__(self, player, init_pos):
		super(Guard, self).__init__(player, init_pos)
		self.chess_type='guard'
		self.name='士', 
		self.allowed_moves=((-1,-1),(-1,1),(1,-1),(1,1))
		self.pos_range=(3,0,5,2)

class Bishop(ChessMan):

	def __init__(self, player, init_pos):
		super(Bishop, self).__init__(player, init_pos)
		self.chess_type='bishop'
		self.name='象', 
		self.allowed_moves=((-2,-2),(-2,2),(2,-2),(2,2))
		self.pos_range=(0,0,8,4)

	def can_move_to(self, board, dest_pos):
		if not super(Bishop, self).can_move_to(board, dest_pos): return False
		move_x = dest_pos[0] - self.pos[0]
		move_y = dest_pos[1] - self.pos[1]
		move_x /= 2
		move_y /= 2
		return board.get((self.pos[0] + move_x, self.pos[1] + move_y)) is None

class Pawn(ChessMan):

	def __init__(self, player, init_pos):
		super(Pawn, self).__init__(player, init_pos)
		self.chess_type='pawn'
		self.name='兵', 
		self.allowed_moves=((0,1),(-1,0),(1,0))
		self.pos_range=(0,3,8,9)

	def can_move_to(self, board, dest_pos):
		if not super(Pawn, self).can_move_to(board, dest_pos): return False
		return dest_pos[1] >= 5 or self.pos[0] == dest_pos[0]

chess_types = ('king', 'rock', 'knight', 'cannon', 'guard', 'bishop', 'pawn')

def get_all_chess_man(player):
	chess_man = {}
	chess_man['king'] = King(player, (4,0))
	for i in [1,2]:
		x = (i - 1) * 8
		chess_man['rock%d' % i] = Rock(player, (x,0))
	for i in [1,2]:
		x = 1 + (i - 1) * 6
		chess_man['knight%d' % i] = Knight(player, (x,0))
	for i in [1,2]:
		x = 1 + (i - 1) * 6
		chess_man['cannon%d' % i] = Cannon(player, (x,2))
	for i in [1,2]:
		x = 3 + (i - 1) * 2
		chess_man['guard%d' % i] = Guard(player, (x,0))
	for i in [1,2]:
		x = 2 + (i - 1) * 4
		chess_man['bishop%d' % i] = Bishop(player, (x,0))
	for i in [1,2,3,4,5]:
		x = (i - 1) * 2
		chess_man['pawn%d' % i] = Pawn(player, (x,3))
	return chess_man

def _action_list_per_pos():
	a_list = []
	chess_set = set()
	for x in range(8):
		a_list.append(('rock', (x+1,0)))
		a_list.append(('rock', (-x-1,0)))
		a_list.append(('cannon', (x+1,0)))
		a_list.append(('cannon', (-x-1,0)))
	for y in range(9):
		a_list.append(('rock', (0,y+1)))
		a_list.append(('rock', (0,-y-1)))
		a_list.append(('cannon', (0,y+1)))
		a_list.append(('cannon', (0,-y-1)))
	for y in range(5, 10):
		a_list.append(('king', (0,y)))
	chesses = get_all_chess_man(None)
	chess_set.add('rock')
	chess_set.add('cannon')
	for name in chesses:
		chess = chesses[name]
		if chess.chess_type in chess_set: continue
		chess_set.add(chess.chess_type)
		for mv in chess.allowed_moves:
			a_list.append((chess.chess_type, mv))
	return a_list

# Each action is defined as an hash-able object
# This is the set of all possible actions per position
action_list = _action_list_per_pos()
action_indexes = {action: i for i, action in enumerate(action_list)}
actions_for_chess = defaultdict(list)
for action in action_list:
	actions_for_chess[action[0]].append(action)

def get_action_index(pos, action):
	num_action = len(action_list)
	x, y = pos
	return action_indexes[action] * 9 * 10 + x * 10 + y

def chess_types():
	return ['king', 'rock', 'knight', 'cannon', 'guard', 'bishop', 'pawn']

chess_id_map = None

def _get_chess_id_map():
	c_types = chess_types()
	idx = 0
	_chess_id_map = {}
	players = ['Red', 'Black']
	for p in players:
		for t in c_types:
			idx += 1
			_chess_id_map[(p, t)] = idx
	return _chess_id_map

chess_id_map = _get_chess_id_map()

class ChessBoard(object):

	def __init__(self, empty_board=False):
		self.board = {}
		self.wins = None
		if empty_board: return
		chesses = get_all_chess_man('Red')
		for chess in chesses.values():
			pos = chess.pos
			self.board[pos] = chess
		chesses = get_all_chess_man('Black')
		for chess in chesses.values():
			pos = (8 - chess.pos[0], 9 - chess.pos[1])
			chess.pos = pos
			self.board[pos] = chess

	def get(self, pos):
		return self.board.get(pos)

	# This function returns a list of transformation names
	def transformation_names(self):
		return ['identity', 'h_flip']

	# This function returns a transformantion given its name
	def transform(self, transformation_name):
		if transformation_name == 'h_flip':
			self.h_flip()
		if transformation_name == 'identity':
			return self
		raise KeyError(transformation_name)

	def clone(self):
		new_board = ChessBoard(empty_board=True)
		new_board.wins = self.wins
		for pos in self.board:
			new_board.board[pos] = self.board[pos].clone()
		return new_board

	def h_flip(self):
		old_board = self.board
		self.board = {}
		for pos in old_board:
			chess = old_board[pos]
			new_pos = (8 - pos[0], pos[1])
			chess.pos = new_pos
			self.board[new_pos] = chess

	def _rotate(self):
		old_board = self.board
		self.board = {}
		for pos in old_board:
			chess = old_board[pos]
			new_pos = (8 - pos[0], 9 - pos[1])
			chess.pos = new_pos
			chess.player = 'Red' if chess.player == 'Black' else 'Black'
			self.board[new_pos] = chess
		if self.wins is not None:
			self.wins = not self.wins

	# Returns a new Board with players switched
	def switch_players(self):
		self._rotate()

	def take_action(self, chess_pos, action):
		assert self.wins is None
		data = self._is_action_valid(chess_pos, action)
		if data is None: return False, None
		new_pos, chess, chess_to_remove = data
		del self.board[chess_pos]
		self.board[new_pos] = chess
		chess.pos = new_pos

		if chess_to_remove is not None and chess_to_remove.chess_type == 'king':
			self.wins = (chess_to_remove.player == 'Black')
		return True, chess_to_remove

	# Returns another Board object given an action
	def get_next_board(self, chess_pos, action):
		if isinstance(action, int):
			action = action_list[action]
		if self._is_action_valid(chess_pos, action) is None:
			return None, None

		new_board = self.clone()
		_, chess_removed = new_board.take_action(chess_pos, action)
		return new_board, chess_removed

	def _is_action_valid(self, chess_pos, action):
		chess = self.get(chess_pos)
		if chess is None or chess.player != 'Red' or chess.chess_type != action[0]: 
			return None
		new_pos = (chess_pos[0] + action[1][0], chess_pos[1] + action[1][1])
		if not chess.can_move_to(self, new_pos): 
			return None
		chess_to_remove = self.board.get(new_pos, None)
		if chess_to_remove is not None and chess_to_remove.player == chess.player:
			return None
		return (new_pos, chess, chess_to_remove)

	# Returns True if the game has terminated
	def is_terminated(self):
		return self.wins is not None

	# Return the index of the winning player of the game
	def is_winner(self):
		return self.wins

	# A vectorized representaiton of the current board.
	def nn_board_repr(self):
		repr = torch.zeros(9, 10).long()
		for pos, chess in self.board.items():
			x, y = pos
			repr[x, y] = chess_id_map.get((chess.player, chess.chess_type))
			assert repr[x, y] != 0

		x_repr = (torch.arange(9).float() - 4).unsqueeze(1).expand(9, 10).unsqueeze(0)
		y_repr = (4 - torch.arange(10).float()).unsqueeze(0).expand(9, 10).unsqueeze(0)
		c_repr = torch.zeros(1 + len(chess_id_map), 9, 10).scatter(0, repr.unsqueeze(0), 1)
		return torch.cat([x_repr, y_repr, c_repr], dim=0)

	# Returns a list of actions corresponding to the neural network output,
	# followed by a list of probabilities corresponding to each action
	# Actions can be defined as arbitary type, consistant with @next_board
	# def nn_actions(self, nn_actions_repr):
	# 	size_a, size_x, size_y = nn_actions_repr.size()
	# 	actions = []
	# 	policy = []
	# 	a_list = self.action_list()
	# 	for x in range(size_x):
	# 		for y in range(size_y):
	# 			for a in range(size_a):
	# 				pos = (x, y)
	# 				action = a_list[a]
	# 				if self._is_action_valid(pos, action) is not None:
	# 					actions.append((pos, a))
	# 					policy.append(nn_actions_repr[a, x, y])
	# 	return actions, policy

	# Returns the list of valid moves, with move-ids corresponding to the neural network output
	# Move-id is the position of an action in the linearized neural network output.
	def nn_valid_actions(self):
		# size_x, size_y, size_a = 9, 10, len(action_list)
		actions = []
		# t = torch.zeros(size_x, size_y, size_a).byte()
		for pos in sorted(self.board.keys()):
			chess = self.board[pos]
			assert pos == chess.pos
			x, y = pos
			for action in actions_for_chess[chess.chess_type]:
				if self._is_action_valid(pos, action) is not None:
					actions.append((pos, action))
					# t[x, y, action_indexes[action]] = 1
		return actions #, t

	# return a json represetation
	def state_dict(self):
		import json
		obj = {}
		for pos in self.board:
			x, y = pos
			chess = self.board[pos]
			obj['%d_%d' % (x, y)] = (x, y, chess.player, chess.chess_type)
		return obj

	def load_state_dict(self, state_dict):
		self.board = {}
		self.wins = None
		for chess_data in state_dict.values():
			pos = (chess_data[0], chess_data[1])
			player = chess_data[2]
			chess_type = chess_data[3]
			chess = None
			if chess_type == 'king': chess = King(player, pos)
			elif chess_type == 'rock': chess = Rock(player, pos)
			elif chess_type == 'knight': chess = Knight(player, pos)
			elif chess_type == 'cannon': chess = Cannon(player, pos)
			elif chess_type == 'guard': chess = Guard(player, pos)
			elif chess_type == 'bishop': chess = Bishop(player, pos)
			elif chess_type == 'pawn': chess = Pawn(player, pos)
			self.board[pos] = chess


if __name__ == '__main__':
	# board = ChessBoard()
	# # print '\n'.join([str(a) for a in actions])
	# # print board.nn_board_repr()
	# # print len(actions)
	# policy = torch.rand(10,9,len(action_list))
	# policy = policy / policy.sum()
	# actions, policy = board.nn_actions(policy)

	# print(len(action_list))

	# b = ChessBoard()
	# repr = b.nn_board_repr()
	# for i in range(4, 17):
	# 	print(repr[:,:,i])

	# for i, a in enumerate(action_list):
	# 	a2 = action_index(i)
	# 	assert a == a2, (a, a2)

	# b = ChessBoard()
	# actions = b.nn_valid_actions()
	# t = t.view(-1)
	# for pos, action in actions:
	# 	i = get_action_index(pos, action)
	# 	assert t[i] == 1
	# 	t[i] = 0
	# assert t.sum() == 0
	
	b = ChessBoard()
	repr = b.nn_board_repr()
	for i in range(repr.size(0)):
		print(repr[i])



