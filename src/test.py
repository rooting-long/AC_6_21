

def test1():
	import torch
	from . import train, ccboard, MCTS, misc, model as model_lib
	args = train.get_args([])
	device = train.get_device(args.device, 0)
	model, _ = train.load_model(args, device, None, 'cur_model_')
	assert model is not None, 'Model file not found'

	# board = ccboard.ChessBoard()
	# chess_pos, action = MCTS.next_action_for_evaluation(args, device, model, board)

	for name, param in model.named_parameters():
		print('{:40s}{:20s}{:.4f}'.format(name, str(list(param.data.size())), param.data.max().item()))
		# if param.data.dim() == 1:
		# 	print(param.data)

	def train_data(it, batch_size):
		s = ccboard.ChessBoard()
		actions = s.nn_valid_actions()
		p = [1 / len(actions)] * len(actions)
		v = 0.5
		d = MCTS.Experience(s, None, p, v)
		return [d]

	optimizer = torch.optim.Adam(model.parameters())
	misc._run_train(args, device, model, optimizer, model_lib.train_loss, train_data, args.iters_per_epochs, None)

def test2():
	import torch
	p1 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
		0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
	p2 = [0.3221, 0.0144, 0.0050, 0.0048, 0.0011, 0.0119, 0.0024, 0.0013, 0.0022,
		0.0005, 0.0021, 0.0623, 0.0016, 0.0228, 0.0000, 0.0000, 0.0000, 0.0023,
		0.0157, 0.0016, 0.0341, 0.0055, 0.0021, 0.0137, 0.0346, 0.1546, 0.1071,
		0.0240, 0.0378, 0.0196, 0.0142, 0.0785]
	p1 = torch.Tensor(p1).cuda()
	p2 = torch.Tensor(p2).cuda()
	print(p1 * (p2 + 1E-8).log())

test2()