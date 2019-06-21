import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from . import ccboard, misc

# Convolution
def conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
	return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
					 stride=stride, padding=padding, bias=bias)

# conv + batch_nor + [ relu ]
def convBlock(in_channels, out_channels, non_linearity, kernel_size=3, padding=1):
	elems = [ conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding), 
			nn.BatchNorm2d(out_channels) ]
	if non_linearity is not None:
		elems.append( non_linearity )
	return nn.Sequential(*elems)


# residual block
class ResidualBlock(nn.Module):

	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = convBlock(channels, channels, self.relu)
		self.conv2 = convBlock(channels, channels, None)

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.conv2(x)
		x += residual
		x = self.relu(x)
		return x

# purpose block
class PurposeBlock(nn.Module):

	def __init__(self, channels, purpose_channels):
		super(PurposeBlock, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.importance = nn.Parameter(torch.Tensor(purpose_channels)).unsqueeze(-1).unsqueeze(-1)
		self.conv1 = convBlock(channels, purpose_channels, self.sigmoid)
		self.conv2 = convBlock(purpose_channels, channels, None)

	def forward(self, x):
		residual = x
		event = self.conv1(x)
		purpose = event * self.importance
		x = self.conv2(purpose)
		x += residual
		import torch.nn as nn

		x = nn.ReLU()(x)
		return event, x

class View(nn.Module):

	def __init__(self, *size):
		super(View, self).__init__()
		self.size = size

	def forward(self, x):
		return x.view(*self.size)

class AlphaGoModel(misc.CompositeModel):

	def externalSubModules(self): return []

	# in_channels = 17
	# out_channels = 96
	def __init__(self, in_channels, out_channels, hidden_channels, residual_blocks, board_size=(9,10)):
		super(AlphaGoModel, self).__init__()
		# input conv
		self.relu = nn.ReLU(inplace=True)
		self.conv_in = convBlock(in_channels, hidden_channels, self.relu, 1, 0)
		# residual blocks
		blocks = []
		for i in range(residual_blocks):
			blocks.append(ResidualBlock(hidden_channels))
		self.residual_blocks = nn.Sequential(*blocks)
		# purpose block
		self.purpose_block = PurposeBlock(hidden_channels, 7)
		# pilicy head
		self.policy_head = nn.Sequential( convBlock(hidden_channels, hidden_channels, self.relu),
				conv(hidden_channels, out_channels, bias=True) )
		# value head
		self.value_head = nn.Sequential( convBlock(hidden_channels, 1, self.relu, kernel_size=1, padding=0),
				View(-1, board_size[0] * board_size[1]),
				nn.Linear(board_size[0] * board_size[1], 64),
				self.relu,
				nn.Linear(64, 1),
				nn.Tanh() )

	def forward(self, x):
		x = self.conv_in(x)
		x = self.residual_blocks(x)
		event, px = self.purpose_block(x)
		x += px
		p = self.policy_head(x)
		# p_size = p.size()
		# p = p.view(p_size[0], -1)
		# p = F.softmax(p, dim=1)
		v = self.value_head(x)
		return p, v, event

	# def loss(self, p, v, target_p, target_v):
	# 	batch_size = p.size(0)
	# 	p = p.view(batch_size, -1).log()
	# 	target_p = target_p.view(batch_size, -1)
	# 	return (((v - target_v) ** 2).sum() - (p * target_p).sum()) / batch_size

	def loss(self, valid_p, v, target_p, target_v, event, target_event):
		batch_size = len(valid_p)
		mse_loss = F.mse_loss(v, target_v) + 0.1 * F.mse_loss(event, target_event) 
		ce_loss = 0
		for i in range(batch_size):
			ce_loss1 = - (target_p[i] * (valid_p[i] + 1E-8).log()).sum()
			assert ce_loss1.item() == ce_loss1.item(), (target_p[i], valid_p[i])
			ce_loss += ce_loss1
		return mse_loss, ce_loss


def get_valid_p(device, p, boards):
	valid_p = []
	valid_log_p = []
	valid_actions = []
	for i, b in enumerate(boards):
		valid_actions1 = b.nn_valid_actions()
		valid_actions.append(valid_actions1)
		action_indexes = [ccboard.get_action_index(*a) for a in valid_actions1]
		action_indexes = torch.Tensor(action_indexes).long().to(device)
		p1 = p[i].view(-1).gather(0, action_indexes)
		valid_log_p.append(p1)
		p1 = F.softmax(p1, dim=0)
		valid_p.append(p1)
	return valid_actions, valid_p, valid_log_p

def get_target_event(events_batch, device):
	target_event = torch.zeros(len(events_batch), 7, 9, 10)
	for b, events in enumerate(events_batch):
		for e in events:
			x, y = e.pos
			chess_type = ccboard.chess_types.index(e.chess_type)
			target_event[b, chess_type, x, y] = 1
	return target_event

def train_loss(args, model, data, device):
	boards = [d.s for d in data] # d: [s, p, a, v=+-1]
	x = torch.stack([b.nn_board_repr() for b in boards])
	p, v, event = model.forward(x.to(device))
	_, valid_p, _ = get_valid_p(device, p, boards)
	target_p = [torch.Tensor(d.p).to(device) for d in data]
	target_v = torch.Tensor([d.v for d in data]).unsqueeze(-1).to(device)
	target_event = get_target_event([d.events for d in data], device)
	mse_loss, ce_loss = model.loss(valid_p, v, target_p, target_v, event, target_event)
	info = {'mse_loss': mse_loss.item(), 'ce_loss': ce_loss.item()}
	return mse_loss + ce_loss, info


if __name__ == '__main__':
	from .ccboard import ChessBoard
	model = AlphaGoModel(17, 96, 16, 5)
	x = ChessBoard().nn_board_repr()
	y = model(x.unsqueeze(0))
