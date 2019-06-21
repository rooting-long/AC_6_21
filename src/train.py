# -*- coding: utf-8 -*-

import torch, os, json, time, numpy
from . import misc, ccboard, model as model_lib

# exploration

def create_model(args):
	model = model_lib.AlphaGoModel(args.in_channels, args.out_channels, args.hidden_channels, args.residual_blocks)
	model.init_weights()
	return model

def try_reload_model(args, device, lock, model_path_base, prev_path):
	if lock is not None: lock.acquire()
	relevant_model_dirs = misc.get_relevant_model_dirs(model_path_base, None)
	assert len(relevant_model_dirs) <= 1, 'Error: Multiple models files:' + str(relevant_model_dirs)
	model = None
	model_path = None
	if len(relevant_model_dirs) == 1:
		model_path = relevant_model_dirs[0]
		if model_path != prev_path:
			model = create_model(args)
			misc._load_model(model_path, model)
			model = model.to(device)
	if lock is not None: lock.release()
	return model, model_path

def get_device(device_name, proc_id):
	misc.set_seed(proc_id)
	return torch.device(device_name if torch.cuda.is_available() else "cpu")

def explore_func(args, device_name, proc_id, queue, lock):
	device = get_device(device_name, proc_id)
	from .MCTS import exploration
	model_path_base = os.path.join(args.experiment_directory, 'best_model_')
	cur_model = None
	cur_model_path = None
	iters = 0
	while True:
		iters += 1
		model, model_path = try_reload_model(args, device, lock, model_path_base, cur_model_path)
		if model is not None: # new model is loaded
			if cur_model is None:
				print('Exploration process #%d is working on %s' % (proc_id, str(device)))
			cur_model = model
			cur_model_path = model_path
		if cur_model is None: # no model is ever available to load
			print('Exploration #%d: best model file not found' % proc_id)
			time.sleep(1) # try again in 1 second
			continue
		history, winner = exploration(args, device, [cur_model, cur_model]) # self play
		if winner is not None:
			queue.put(history)

# evaluation

def compare_models(args, device, best_model, next_model, iters):
	from .MCTS import exploration
	eval_path = os.path.join(args.experiment_directory, 'eval')
	if not os.path.isdir(eval_path):
		os.mkdir(eval_path)
	assert args.evaluation_games % 2 == 0
	wins = 0
	losses = 0
	ties = 0
	while wins + losses < args.evaluation_games:
		for first in range(2):
			models = [next_model, best_model] if first else [best_model, next_model]
			history, winner = exploration(args, device, models)
			if winner is None: ties += 1
			elif winner == first: wins += 1
			else: losses += 1
			for h in history: h.s = h.s.state_dict()
			history = [h.state_dict() for h in history]
			with open(os.path.join(eval_path, '%d.actions.json' % iters), 'w') as fp:
				json.dump(history, fp, indent=4, sort_keys=True, ensure_ascii=False)
			iters += 1
			old_file = os.path.join(eval_path, '%d.actions.json' % (iters - args.eval_save_files))
			if os.path.isfile(old_file):
				os.unlink(old_file)
	print('wins:%d losses:%d ' % (wins, losses))
	return iters, wins / (wins + losses)

# def clone_model(args, device, lock, model):
# 	lock.acquire()
# 	state_dict = model.state_dict()
# 	model = create_model(args)
# 	model.load_state_dict(state_dict)
# 	lock.release()
# 	return model.to(device)

def load_model(args, device, lock, file_prefix):
	model_path_base = os.path.join(args.experiment_directory, file_prefix)
	return try_reload_model(args, device, lock, model_path_base, None)

def save_model(args, lock, model, file_prefix, prev_model_path=None):
	if not os.path.isdir(args.experiment_directory):
		os.mkdir(args.experiment_directory)
	model_path = os.path.join(args.experiment_directory, file_prefix + misc.datetimestr())
	lock.acquire()
	misc._save_model(model, model_path, prev_model_path)
	lock.release()
	return model_path

def eval_func(args, device, lock, eval_lock):
	while True:
		best_model, best_model_path = load_model(args, device, lock, 'best_model_')
		if best_model is not None: break
		print('Eval: best model file not found')
		time.sleep(1)
	iters = 0
	win = -1
	while True:
		while True:
			cur_model, _ = load_model(args, device, eval_lock, 'cur_model_')
			if cur_model is not None: break
			print('Eval: cur model file not found')
			time.sleep(1)
		print('Eval #%d is working on %s, previous win: %.2f' % (iters, str(device), win))		
		iters, win = compare_models(args, device, best_model, cur_model, iters)
		if win > 0.55:
			best_model_path = save_model(args, lock, cur_model, 'best_model_', best_model_path)
			best_model = cur_model

def start_eval_proc(args, device, ctx, lock, eval_lock):
	args = (args, device, lock, eval_lock)
	evaluation_proc = ctx.Process(target=eval_func, args=args)
	evaluation_proc.daemon = True
	evaluation_proc.start()

# train

# def anneal_lr(optimizer, logger):
# 	optim_state = optimizer.state_dict()
# 	optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args().learning_anneal
# 	optimizer.load_state_dict(optim_state)
# 	logger.info('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

def train_func(args, ctx, queue, lock):
	device = get_device(args.device, 0)
	eval_lock = ctx.Lock()
	model, best_model_path = load_model(args, device, lock, 'best_model_')
	if model is None:
		model = create_model(args)
		model = model.to(device)
		save_model(args, lock, model, 'best_model_')
	model, cur_model_path = load_model(args, device, eval_lock, 'cur_model_')
	if model is None:
		model = create_model(args)
		model = model.to(device)
		cur_model_path = save_model(args, eval_lock, model, 'cur_model_')

	recent_replay_buffer = []
	replay_buffer = []
	total_produced = 0
	total_trained = 0

	def repopulate_buffers():
		nonlocal total_produced
		while not queue.empty():
			history = queue.get()
			total_produced += len(history)
			recent_replay_buffer.extend(history)
		while len(recent_replay_buffer) > args.recent_replay_buffer_size:
			replay_buffer.append(recent_replay_buffer.pop(0))
		while len(replay_buffer) > args.replay_buffer_size:
			replay_buffer.pop(0)

	def train_data(it, batch_size):
		nonlocal total_trained
		repopulate_buffers()
		recent_batch_size = batch_size // 2
		batch_size -= recent_batch_size
		if len(replay_buffer) < batch_size:
			recent_batch_size += (batch_size - len(replay_buffer))
			batch_size = len(replay_buffer)
		if len(recent_replay_buffer) < recent_batch_size:
			recent_batch_size = len(recent_replay_buffer)
		if recent_batch_size + batch_size == 0:
			return []
		recent_batch = numpy.random.choice(recent_replay_buffer, recent_batch_size)
		if batch_size == 0:
			total_trained += len(recent_batch)
			return recent_batch
		batch = list(recent_batch) + list(numpy.random.choice(replay_buffer, batch_size))
		total_trained += len(batch)
		return batch

	while True:
		repopulate_buffers()
		if len(recent_replay_buffer) > 0:
			break
		# print('len(recent_replay_buffer)', len(recent_replay_buffer))
		time.sleep(10)
	print('Training process is working on %s' % str(device))
	start_eval_proc(args, device, ctx, lock, eval_lock)

	optimizer = torch.optim.Adam(model.parameters())
	train_plot = misc.InfoPlot(os.path.join(args.experiment_directory, 'plot_train'))
	for epoch in range(args.num_train_epochs):
		cur_model_path = save_model(args, eval_lock, model, 'cur_model_', cur_model_path)
		print('Epoch %d - %s (%d/%d)' % (epoch + 1, misc.datetimestr(), total_trained, total_produced))
		misc._run_train(args, device, model, optimizer, model_lib.train_loss, train_data, args.iters_per_epochs, train_plot)

# multi-processing

def start_processes(args, train_func, explore_func):
	import multiprocessing
	ctx = multiprocessing.get_context('spawn')
	queue = ctx.Queue()
	lock = ctx.Lock()
	proc_id = 0
	for device in args.exploration_devices:
		for i in range(args.exploration_processes):
			proc_id += 1
			explore_process = ctx.Process(target=explore_func, args=(args, device, proc_id, queue, lock))
			explore_process.daemon = True
			explore_process.start()
	train_func(args, ctx, queue, lock)

# main

def add_model_arguments(ap):
	ap.add_argument('--in_channels', default=17, type=int, help='Types of chess (*2 players) (+1 empty) per position.')
	ap.add_argument('--out_channels', default=96, type=int, help='Number of actions per position')
	ap.add_argument('--hidden_channels', default=128, type=int, help='Number of hidden channels')
	ap.add_argument('--residual_blocks', default=10, type=int, help='Number of residual blocks')
	# ap.add_argument('--board_width', default=9, type=int, help='Width of the game board')
	# ap.add_argument('--board_height', default=10, type=int, help='Height of the game board')

def add_train_arguments(ap):
	# reinforcement
	ap.add_argument("--num_train_epochs", default=1000000, type=int, help="Total number of training epochs to perform.")
	ap.add_argument("--plot_steps", type=int, default=100)
	ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
	ap.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
	ap.add_argument('--search_batch_size', default=32, type=int, help='Batch size for evaluation in search')
	ap.add_argument('--search_simulations', default=256, type=int, help='Number of simulations in search')
	ap.add_argument('--max_game_steps', default=128, type=int, help='Maximum number of steps in each game')
	ap.add_argument('--policy_noise_ratio', default=0.25, type=float, help='Ratio of noise added into the policy in exploration')
	ap.add_argument('--recent_replay_buffer_size', default=3200, type=int, help='The size of the replay buffer.')
	ap.add_argument('--replay_buffer_size', default=500000, type=int, help='The size of the replay buffer.')
	ap.add_argument('--evaluation_games', default=10, type=int, help='Number of games used in evaluation.')
	ap.add_argument('--c_puct', default=1., type=float, help='A constant determining the level of exploration.')
	# eval
	ap.add_argument("--test_batch_size", type=int, default=32)
	ap.add_argument("--eval_save_files", type=int, default=20)
	ap.add_argument("--iters_per_epochs", type=int, default=1000)
	# dirs
	ap.add_argument("--experiment_directory", default='_test', type=str)
	# train
	ap.add_argument('--exploration_devices', default='cuda:1,cuda:2', type=str, help='The GPU ids for the GPUs used for exploration.')
	ap.add_argument('--exploration_processes', default=5, type=int, help='The number of exploration threads per gpu.')
	# ap.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
	# ap.add_argument('--anneal_interval', default=100, type=int, help='Epochs between annealing is applied')
	# ap.add_argument('--learning_anneal', default=1.01, type=float, help='Annealing applied to learning rate')
	# ap.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
	# ap.add_argument('--train_mini_batch', default=128, type=int, help='Mini-batch size in training')
	# ap.add_argument('--momentum', default=0.9, type=float, help='momentum')
	# ap.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
	# ap.add_argument('--logdir', default='log', type=str, help='Log folder')
	# ap.add_argument('--continue_from', type=str, help='Continue from last training epoch')
	# ap.add_argument('--plot', action='store_true', help='Plot training progress in terms of accuracies')
	# parse
	return ap.parse_args([])

def get_args(from_command_line=True):
	from .misc import get_argument_parser
	ap = get_argument_parser()
	add_model_arguments(ap)
	add_train_arguments(ap)
	if from_command_line:
		ap = ap.parse_args()
	else:
		ap = ap.parse_args([])
	ap.exploration_devices = ap.exploration_devices.split(",")
	return ap

if __name__ == '__main__':
	args = get_args()
	start_processes(args, train_func, explore_func)
