import torch, numpy as np
from collections import namedtuple
from . import ccboard, model as model_lib

class Edge(object):

	def __init__(self, node, a, p):
		# see page 8 of the nature paper
		# $node.s$ and $self.node$ are of type ChessBoard
		# With $a$, we can call node.s.get_next_board(*a)
		self.node = node
		self.a = a # (pos, action)
		self.p = p
		self.n = 0
		self.sum_v = 0
		self.count_v = 0
		self.next_node = None
		self.event = None

	# def is_leaf(self):
	# 	return self.node is None

	def q(self):
		if self.count_v == 0: return 0
		return self.sum_v / self.count_v

	def u(self, c_puct):
		# sum_n = 1
		# for e in self.prev_node.next_edges.values():
		# 	sum_n += e.n
		# import math
		# return c_puct * self.p * math.sqrt(sum_n) / (1 + self.n)
		return c_puct * self.p / (1 + self.n)


class Node(object):

	def __init__(self, s, edge=None):
		self.s = s # board
		self.edge = edge # incoming edge
		self.next_edges = None

	def expand(self, p, actions, v):
		if self.s.is_terminated():
			assert self.s.is_winner() == False
		elif p is not None:
			self.next_edges = {}
			for a1, p1 in zip(actions, p):
				self.next_edges[a1] = Edge(self, a1, p1)
		self._propagate_w_backward(v)

	def _propagate_w_backward(self, v):
		v = -v #
		if self.edge is not None:
			self.edge.sum_v += v
			self.edge.count_v += 1
			self.edge.node._propagate_w_backward(v)


class Tree(object):

	def __init__(self, args, device, model, root_node):
		self.args = args
		self.device = device
		self.model = model
		self.root_node = root_node

	def search(self, tau): # tau: temperature
		batch = []
		if self.root_node.next_edges is None:
			batch.append(self.root_node)
		for i in range(self.args.search_simulations):
			to_expand = self._search_node(self.root_node)
			if to_expand is not None and not to_expand.s.is_terminated():
				batch.append(to_expand)
			if len(batch) == self.args.search_batch_size or to_expand is None:
				if len(batch) == 0: break
				self._expand_nodes(batch, self.device)
				batch = []
		return self._get_policy(tau)

	def _get_policy(self, tau):
		# self._print_tree(self.root_node, '')
		visits = []
		actions = []
		for action, edge in self.root_node.next_edges.items():
			visits.append(edge.n)
			actions.append(action)
		# apply tau
		action_probs = torch.Tensor(visits)
		if tau == 0:
			action_probs = (action_probs == action_probs.max()).type_as(action_probs)
		else:
			action_probs = action_probs.pow(1 / tau)
		action_probs = action_probs / action_probs.sum()
		return action_probs, actions

	def _expand_nodes(self, batch, device):
		boards = [node.s for node in batch]
		x = torch.stack([b.nn_board_repr() for b in boards])
		p, v = self.model(x.to(device))
		valid_actions, valid_p, _ = model_lib.get_valid_p(device, p, boards)
		v = v.cpu()
		for i, node in enumerate(batch):
			p1 = valid_p[i].cpu().tolist()
			node.expand(p1, valid_actions[i], v[i].item())

	# Returns a node to expand, None if no node to expand
	def _search_node(self, node):
		if node.next_edges is None: # the node has not been expanded
			return None
		if node.s.is_terminated(): # terminated node should not be searched
			return None
		next_edges = list(node.next_edges.values())
		next_edges.sort(key=lambda e: e.q() + e.u(self.args.c_puct), reverse=True)
		for edge in next_edges:
			if edge.next_node is None:
				s, event = node.s.get_next_board(*edge.a)
				s.switch_players()
				edge.next_node = Node(s, edge)
				edge.event = event
				edge.n += 1
				if s.is_terminated():
					assert not s.is_winner()
					edge.n = 1E4 # a winning edge
					edge.next_node.expand(None, None, -1)
				return edge.next_node
			to_expand = self._search_node(edge.next_node)
			if to_expand is not None:
				edge.n += 1
				return to_expand
		return None

	def print(self, depth):
		self._print_tree(self.root_node, '', depth)

	def _print_tree(self, node, indent, depth):
		if depth == 0: return
		import json
		# print(indent + 'board: ' + json.dumps(node.s.state_dict()))
		print(indent + 'next_edges:' + ('None' if node.next_edges is None else str(len(node.next_edges))) + ' is_terminated:' + str(node.s.is_terminated()))
		if node.next_edges is None: return
		indent += ' '
		for a in node.next_edges:
			(pos, action) = a
			edge = node.next_edges[a]
			if edge.next_node is None:
				assert edge.n == 0
				continue
			print(indent + str(pos) + ' ' + str(action) + ' -> ' + ' p:%.3f n:%d sum_v:%.3f count_v:%d' % (edge.p, edge.n, edge.sum_v, edge.count_v))
			self._print_tree(edge.next_node, indent + ' ', depth - 1)


def default_tau_func(step):
	return 1 if step < 10 else 0

# returns a normalized policy
def _mcts_policy(args, device, model, node, tau, debug):
	tree = Tree(args, device, model, node)
	action_probs, actions = tree.search(tau)
	if debug: tree.print(depth=1)
	return action_probs, actions

def _next_action(board, policy, actions, noise_ratio=0.25, dirichlet=0.03):
	n = len(actions)
	policy = policy.numpy()
	if noise_ratio > 0:
		dirichlet = np.random.dirichlet(dirichlet * np.ones(n))
		policy = (1 - noise_ratio) * policy + noise_ratio * dirichlet
	policy /= np.sum(policy)
	index = np.random.choice(range(len(actions)), size=1, p=policy)
	return actions[index[0]]

class Experience:
	def __init__(self, s, a, p, v):
		self.s = s
		self.a = a
		self.p = p
		self.v = v
		self.events = []
	def state_dict(self):
		return {'s':self.s, 'a':self.a, 'p':self.p, 'v':self.v, 
				'events':[e.state_dict() for e in self.events]}

def next_action_for_evaluation(args, device, model, board):
	policy, actions = _mcts_policy(args, device, model, Node(board), 0, True)
	return _next_action(board, policy, actions, noise_ratio=0)

def exploration(args, device, models, resign_value=None, tau_func=default_tau_func):
	history = []
	cur_node = Node(ccboard.ChessBoard())
	winner = None
	for step in range(args.max_game_steps):
		policy, actions = _mcts_policy(args, device, models[step % 2], cur_node, tau_func(step), False)
		pos, action = _next_action(cur_node.s, policy, actions, args.policy_noise_ratio)
		history.append( Experience(cur_node.s, (pos, action), policy.tolist(), 0) )
		edge = cur_node.next_edges[(pos, action)]
		next_node = edge.next_node
		if next_node is None: # a node is randomly chosen that has not been searched
			s, edge.event = cur_node.s.get_next_board(pos, action)
			s.switch_players()
			next_node = Node(s, None)
		history[-1].events.append(edge.event)
		if next_node.s.is_terminated():
			assert not next_node.s.is_winner()
			winner = step % 2
			break
		if resign_value is not None and v < resign_value:
			winner = (step + 1) % 2 # current player losses 
			break
		cur_node = next_node
		cur_node.edge = None # make it the root

	# fill scores
	if winner is not None:
		assert winner == 1 - len(history) % 2
		for i, h in enumerate(reversed(history)):
			h.v = 1 if i % 2 == 0 else -1
	events = [n.events[0] if len(n.events) > 0 else None for h in history]
	for i, e in enumerate(events):
		if e == None: continue
		for j in range(1, 5): # 5 is a parameter
			if i - j < 0: continue
			history[i - j].events.append(e) 
	return history, winner

	
