

import torch, torch.nn as nn
import os, random, numpy, time, sys, datetime, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NEG_INF = -1e6

def set_seed(seed):
	seed += int(time.time() * 1000) % (2**32)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	numpy.random.seed(seed)

def datetimestr():
	def get_text(value):
		if value < 10:
			return '0%d' % value
		else:
			return '%d' % value
	now = datetime.datetime.now()
	return get_text(now.year) + get_text(now.month) + get_text(now.day) + '_' + \
			get_text(now.hour) + get_text(now.minute) + get_text(now.second)


def format_time(seconds, with_ms=False):
	days = int(seconds / 3600/24)
	seconds = seconds - days*3600*24
	hours = int(seconds / 3600)
	seconds = seconds - hours*3600
	minutes = int(seconds / 60)
	seconds = seconds - minutes*60
	secondsf = int(seconds)
	seconds = seconds - secondsf
	millis = int(seconds*1000)

	f = ''
	if days > 0:
		f += str(days) + 'D'
	if hours > 0:
		f += str(hours) + ':'
	f += str(minutes) + '.' + str(secondsf)
	if with_ms and millis > 0:
		f += '_' + str(millis)
	return f


# progress_bar

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30
last_time = time.time()
begin_time = last_time
last_len = -1
last_step = -1
def progress_bar(current, total, msg=None):
	global last_time, begin_time, last_len, last_step
	if last_step == current: return
	last_step = current
	if current == 0:
		begin_time = time.time()  # Reset for new bar.
		last_len = -1

	cur_len = int(TOTAL_BAR_LENGTH*current/total)
	rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
	cur_time = time.time()
	if last_len == cur_len and current < total - 1 and cur_time - last_time < 1 and msg is None:
		return

	sys.stderr.write(' [')
	for i in range(cur_len):
		sys.stderr.write('=')
	sys.stderr.write('>')
	for i in range(rest_len):
		sys.stderr.write('.')
	sys.stderr.write(']')

	last_time = cur_time
	tot_time = cur_time - begin_time
	last_len = cur_len

	L = []
	est_time = tot_time / (current + 1) * total
	L.append(' Time:%s/Est:%s' % (format_time(tot_time), format_time(est_time)))
	if msg:
		L.append(' ' + msg)

	msg = ''.join(L)
	sys.stderr.write(msg)
	blank_count = term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3
	for i in range(blank_count):
		sys.stderr.write(' ')

	# Go back to the center of the bar.
	info = ' %d/%d ' % (current+1, total)
	for i in range(blank_count + len(msg) + int(TOTAL_BAR_LENGTH)//2 + len(info)//2):
		sys.stderr.write('\b')
	sys.stderr.write(info)

	if current < total - 1:
		sys.stderr.write('\r')
	else:
		sys.stderr.write('\n')
	sys.stderr.flush()



# plot

# 平滑多条数据
def smooth2d(data0, step=4):
	data = torch.zeros(data0.size())
	error = torch.ones(data0.size())
	size = data0.size(1)
	for i in range(size):
		from1 = i - step / 2
		to1 = from1 + step
		if from1 < 0: from1 = 0
		if to1 > size: to1 = size
		data1 = data0[:, from1:to1]
		data[:,i] = data1.mean(1)
		error[:,i] = data1.std(1)
	return data, error

# 平滑1条数据
def smooth1d(data0, step=3):
	data = torch.zeros(data0.size())
	error = torch.ones(data0.size())
	size = data0.size(0)
	for i in range(size):
		from1 = i - step / 2
		to1 = from1 + step
		if from1 < 0: from1 = 0
		if to1 > size: to1 = size
		data1 = data0[from1:to1]
		data[i] = data1.mean()
		error[i] = data1.std()
	return data, error

def plot(x_data, y_data, y_err=None, legends=None, title=None, xlabel=None, ylabel=None, filename=None):
	with plt.style.context(('fivethirtyeight')):
		#plt.style.use('fivethirtyeight')
		fig, ax = plt.subplots(squeeze=True)
		for i in range(y_data.size(0)):
			ax.plot(x_data.numpy(), y_data[i].numpy(), label=legends[i] if legends else '?')
			if (not y_err is None) and (not y_err[i] is None):
				ax.fill_between(x_data.numpy(), (y_data[i]-y_err[i]).numpy(), (y_data[i]+y_err[i]).numpy(), alpha=0.3)
		if legends: ax.legend(loc=2)
		if title: ax.set_title(title)
		if xlabel: ax.set_xlabel(xlabel)
		if ylabel: ax.set_ylabel(ylabel)
		#plt.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.12)
		fig.tight_layout()
	if filename:
		plt.savefig(filename)
		plt.close()
	else:
		plt.show()
		return plt
		# plt.close()
		# plt.show(block=False)

def _data_stat(arr):
	arr = torch.Tensor(arr)
	return {'mean':arr.mean().item(), 
			'std':arr.std().item(),
			'max':arr.max().item(), 
			'min':arr.min().item(), 
			'median':arr.median().item() }

def _pdf_plot(path, info_list, y_key):
	x_data = []
	y_data = []
	legends = None
	is_list = None
	for i, info in enumerate(info_list):
		if y_key not in info: continue
		if info[y_key] is None: continue
		x_data.append(i)
		if is_list is None:
			is_list = isinstance(info[y_key], list)
		else:
			assert is_list == isinstance(info[y_key], list)
		if is_list:
			stat = _data_stat(info[y_key])
			if legends is None:
				legends = sorted(stat.keys())
			y_data.append([stat[l] for l in legends])
		else:
			y_data.append(info[y_key])
	if is_list:
		y_data = torch.Tensor(y_data).transpose(0, 1)
	else:
		y_data = torch.Tensor(y_data).unsqueeze_(0)
	x_data = torch.Tensor(x_data)
	plot(x_data, y_data, y_err=None, legends=legends, title=None, xlabel=None, ylabel=y_key, filename=path)

class InfoPlot(object):
	def __init__(self, path, max_data_plot=200):
		self.path = path
		self.info_list = []
		self.max_data_plot = max_data_plot
		self.keys = set()
	def add(self, info):
		info = InfoPlot.reduce_info(info)
		for key, value in info.items():
			assert isinstance(key, str)
			if isinstance(value, (int, float)): continue
			assert isinstance(value, (tuple, list)), type(value).__name__
			for v in value:
				assert isinstance(v, (int, float)), type(v).__name__
		self.info_list.append(info)
		keys = info.keys()
		self.keys.update(info.keys())
		return info
	@staticmethod
	def reduce_info(info):
		info2 = {}
		to_remove = set()
		for key, value in info.items():
			key2 = key.split('_')
			if len(key2) == 2 and key2[1] == 'correct' and (key2[0] + '_count') in info and (key2[0] + '_acc') not in info:
				key3 = key2[0] + '_acc'
				count = info[key2[0] + '_count']
				info2[key3] = 0 if count == 0 else value / count
				to_remove.add(key2[0] + '_count')
			else:
				info2[key] = value
		for key in to_remove:
			del info2[key]
		return info2
	@staticmethod
	def list_avg(info_list):
		assert isinstance(info_list, list)
		if len(info_list) == 0: return
		info = {}
		info_len = {}
		for info1 in info_list:
			for key, value in info1.items():
				if value is None: continue
				if key not in info:
					info[key] = value
					info_len[key] = 1
				else:
					if isinstance(value, list):
						if len(value) > len(info[key]):
							info[key] = value
						elif len(value) == len(info[key]):
							for i, v in enumerate(value):
								info[key][i] += v
					else:
						assert isinstance(value, (int, float))
						info[key] += value
					info_len[key] += 1
		for key, value in info.items():
			if isinstance(value, list):
				for i in range(len(value)):
					value[i] /= info_len[key]
			else:
				info[key] /= info_len[key]
		return info		
	def add_list_avg(self, info_list):
		info = InfoPlot.list_avg(info_list)
		self.add(info)
	def plot(self):
		if not os.path.isdir(self.path):
			os.mkdir(self.path)
		if len(self.info_list) > self.max_data_plot:
			self.info_list = self.info_list[-self.max_data_plot:]
		for y_key in self.keys:
			_pdf_plot(os.path.join(self.path, y_key + '.pdf'), self.info_list, y_key)
		with open(os.path.join(self.path, 'plot.json'), 'w') as fp:
			json.dump(self.info_list, fp, indent=4, sort_keys=True, ensure_ascii=False)
	def save_output(self, output):
		with open(os.path.join(self.path, 'eval_output.txt'), 'w') as fp:
			for line in output:
				fp.write(line + '\n')

# png

def visualize(img, title=None, save=None):
	img = img.numpy()
	if save is None:
		# plt.imshow(inp, cmap=plt.cm.bone)
		plt.imshow(img)
		if title is not None:
			plt.title(title)
		plt.show()
	else:
		from cv2 import imwrite
		imwrite(save, img)

def visualize3d(img, title=None, save=None):
	from torchvision.utils import make_grid
	if img.dim() == 3:
		img = img.unsqueeze(1)
	img = make_grid(img)
	img = img.permute(1, 2, 0)
	visualize(img, title, save)

# beam

def beam_search(scores, max_beam_size):
	paths = [[]]
	beam_scores = scores[0].new([0]) if isinstance(scores, (tuple, list)) else scores.new([0])
	for score in scores:
		beam_size = score.size(0)
		if beam_size > max_beam_size: beam_size = max_beam_size
		top_score, top_index = score.topk(beam_size)
		cur_beam_size = beam_scores.size(0)
		top_choice = beam_scores.unsqueeze(1) + top_score.unsqueeze(0) # cur_beam_size x beam_size
		beam_scores, top_choice_index = top_choice.view(-1).topk(beam_size)
		top_index = top_index.tolist()
		top_choice_index = top_choice_index.tolist()
		paths = [paths[index // beam_size] + [top_index[index % beam_size]] for index in top_choice_index]
	return paths, beam_scores

def reduce_beam(beam, beam_size, key):
	beam.sort(key=key, reverse=True)
	return beam[:beam_size]

# clip norm

def clip_norm(data, max_norm, norm_type=2, eps=1e-4):
	norm = []
	norm_type = float(norm_type)
	if norm_type == float("inf"):
		norm = [None if d is None else d.abs().max().item() for d in data]
	else:
		norm = [None if d is None else d.norm(norm_type).item() for d in data]
	if max_norm is not None:
		max_norm = float(max_norm)
		for i, d in enumerate(data):
			if norm[i] is None or norm[i] < eps: continue
			assert norm[i] != float('nan') and norm[i] > 0, (norm[i], d)
			clip_coef = max_norm / norm[i]
			if clip_coef < 1:
				d.mul_(clip_coef)
	return norm

def clip_grad_norm(parameters, max_grad_norm, norm_type=2, eps=1e-6):
	data = [p.grad.data for p in parameters if p.grad is not None]
	return clip_norm(data, max_grad_norm, norm_type, eps)

def clip_weight_norm(parameters, max_weight_norm, norm_type=2, eps=1e-6):
	data = [p.data for p in parameters if p.grad is not None]
	return clip_norm(data, max_weight_norm, norm_type, eps)

# initialization

def weight_init(m):
	import torch.nn.init as init
	'''
	Usage:
		model = Model()
		model.apply(weight_init)
	'''
	if isinstance(m, nn.Conv1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm1d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm3d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.Bilinear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
		if m.weight.data.size(1) == m.weight.data.size(2):
			scatter_index = torch.arange(m.weight.data.size(1))
			scatter_index = scatter_index.unsqueeze_(0).unsqueeze_(2).expand_as(m.weight.data)
			m.weight.data.scatter_(1, scatter_index, 1.0)
	elif isinstance(m, nn.LSTM):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.LSTMCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRU):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRUCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)


class CompositeModel(nn.Module):

	# subclasses should return a list of sub-modules that do not want to be initialized or saved
	def externalSubModules(self):
		raise NotImplementedError()

	def __apply_to_non_external_sub_modules(self, action_to_apply):
		external_sub_modules = self.externalSubModules()
		removed = {}
		for name, module in self._modules.items():
			if module in external_sub_modules:
				self.__setattr__(name, None)
				removed[name] = module
		action_to_apply(self)
		for name, module in removed.items():
			self.__setattr__(name, module)

	def init_weights(self):
		self.__apply_to_non_external_sub_modules(lambda model: model.apply(weight_init))
	
	def load_model(self, model_path):
		def load(model):
			state_dict = torch.load(model_path, map_location=torch.device('cpu'))
			model.load_state_dict(state_dict)
		self.__apply_to_non_external_sub_modules(load)

	def save_model(self, model_file, previous_file=None):
		def save(model):
			if previous_file is not None and os.path.exists(previous_file):
				os.remove(previous_file)
			torch.save(model.state_dict(), model_file)
		self.__apply_to_non_external_sub_modules(save)


class FeedForward(nn.Module):
	def __init__(self, sizes, nonlinearity=nn.ReLU):
		super(FeedForward, self).__init__()
		self.layers = len(sizes) - 1
		self.nonlinearity = nonlinearity
		for i in range(self.layers):
			setattr(self, 'linear_%d' % (i + 1), nn.Linear(sizes[i], sizes[i + 1]))

	def forward(self, x):
		for i in range(self.layers):
			if i != 0:
				x = self.nonlinearity()(x)
			linear = getattr(self, 'linear_%d' % (i + 1))
			x = linear(x)
		return x

def _gc():
	import gc, resource
	gc.collect()
	max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	print("{:.2f} MB".format(max_mem_used / 1024))

# train

def get_relevant_model_dirs(model_path_base, default_postfix=None):
	components = model_path_base.split('/')
	directory = '/'.join(components[:-1])
	if os.path.isdir(directory):
		relevant_dirs = [os.path.join(directory, f) for f in os.listdir(directory) 
						if f.startswith(components[-1]) and 
							os.path.isdir(os.path.join(directory, f))]
	else:
		relevant_dirs = []
	if len(relevant_dirs) > 1 and default_postfix is not None:
		dirs = [d for d in relevant_dirs if d.endswith(default_postfix)]
		if len(dirs) == 1:
			relevant_dirs = dirs
	return relevant_dirs

def _load_model(model_path, model):
	print("Loading model from folder '%s' ..." % model_path)
	# main model
	assert isinstance(model, CompositeModel)
	model_name = type(model).__name__
	main_model_path = os.path.join(model_path, '%s.pth' % model_name)
	model.load_model(main_model_path)
	# external models
	external_sub_modules = model.externalSubModules()
	for m in external_sub_modules:
		if m is None: continue
		sub_model_name = type(m).__name__
		sub_model_path = os.path.join(model_path, '%s.pth' % sub_model_name)
		if os.path.isfile(sub_model_path):
			m.load_model(sub_model_path)		

def load_model(model_path_base, model, default_postfix='_latest', ensure_loaded=False):
	relevant_dirs = get_relevant_model_dirs(model_path_base, '_latest')
	assert len(relevant_dirs) <= 1, "Multiple possibilities {}".format(relevant_dirs)
	if ensure_loaded:
		assert len(relevant_dirs) == 1, "Cannot find a model to load"
	if len(relevant_dirs) == 0:
		return None
	_load_model(relevant_dirs[0], model)
	return relevant_dirs[0]

def _save_model(model, model_path, prev_model_path):
	print("Saving model to folder '%s' ..." % model_path)
	if not os.path.isdir(model_path):
		os.mkdir(model_path)
	# main model
	assert isinstance(model, CompositeModel)
	main_model_name = type(model).__name__
	main_model_path = os.path.join(model_path, '%s.pth' % main_model_name)
	prev_main_model_path = None if prev_model_path is None else os.path.join(prev_model_path, '%s.pth' % main_model_name)
	model.save_model(main_model_path, prev_main_model_path)
	# external models
	external_sub_modules = model.externalSubModules()
	for m in external_sub_modules:
		if m is None: continue
		sub_model_name = type(m).__name__
		sub_model_path = os.path.join(model_path, '%s.pth' % sub_model_name)
		prev_sub_model_path = None if prev_model_path is None else os.path.join(prev_model_path, '%s.pth' % sub_model_name)
		m.save_model(sub_model_path, prev_sub_model_path)
	if prev_model_path is not None and os.path.isdir(prev_model_path):
		print("Removing previous model from folder '%s' ..." % prev_model_path)
		for file in os.listdir(prev_model_path):
			file = os.path.join(prev_model_path, file)
			print("  Remove file '%s' ..." % file)
			os.unlink(file)
		os.rmdir(prev_model_path)
	return model_path


def _train_batch(args, device, model, optimizer, update_param, data, train_loss):
	is_train = (optimizer is not None)
	if is_train:
		model.train()
	else:
		model.eval()
	with torch.set_grad_enabled(is_train):
		loss, info = train_loss(args, model, data, device)
		if loss == 0:
			info['loss'] = 0
			return info
		info['loss'] = loss.item()
		assert not isinstance(loss, (int, float))
		if is_train:
			loss /= args.gradient_accumulation_steps
			loss.backward()
		loss = None
		assert info['loss'] != float('inf') and info['loss'] == info['loss'], info['loss']
		if is_train and update_param:
			# print(loss)
			# for name, param in model.named_parameters():
			# 	if param.grad is not None:
			# 		print(name, param.data.max(), param.grad.data.max())
			# print()
			if args.max_grad_norm is not None:
				info['grad_norm'] = clip_grad_norm(model.parameters(), args.max_grad_norm)
			optimizer.step()
			optimizer.zero_grad()
			if args.max_weight_norm is not None:
				info['weight_norm'] = clip_weight_norm(model.parameters(), args.max_weight_norm)
		return info

def _run_train(args, device, model, optimizer, train_loss, train_data, train_iters, train_plot):
	iterations = train_iters or (len(train_data) - 1) // args.batch_size + 1
	info_batch = []
	for it in range(iterations):
		progress_bar(it, iterations)
		update_param = ((it + 1) % args.gradient_accumulation_steps == 0 or it == iterations - 1)
		# if args.batch_size == 1:
		# 	data = train_data[it]
		# else:
		# 	data = train_data[it * args.batch_size : (it + 1) * args.batch_size]
		data = train_data(it, args.batch_size)
		train_info = _train_batch(args, device, model, optimizer, update_param, data, train_loss)

		if train_info is not None:
			info_batch.append(train_info)
		if ((it + 1) % args.plot_steps == 0 or it + 1 == iterations) and len(info_batch) > 0:
			if train_plot is not None: 
				train_plot.add_list_avg(info_batch)
				train_plot.plot()
			info_batch = []


# Python multi-processing
# https://www.jqhtml.com/37257.html?wpfpaction=add&postid=37257

class Train:

	def __init__(self, args, model, train_loss, evaluate, train_data, dev_data, model_prefix='', optimizer=None):
		self.args = args
		self.train_data = train_data
		self.dev_data = dev_data
		self.train_loss = train_loss
		self.evaluate = evaluate
		set_seed(args.seed)
		self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
		# model
		self.model_path_base = os.path.join(args.experiment_directory, '%smodel' % model_prefix)
		load_model(self.model_path_base, model)
		self.model = model.to(self.device)
		self.optimizer = optimizer or torch.optim.Adam(self.model.parameters())
		self.train_plot = InfoPlot(os.path.join(args.experiment_directory, '%splot_train' % model_prefix))
		self.eval_plot = InfoPlot(os.path.join(args.experiment_directory, '%splot_eval' % model_prefix))
		

	def train(self, num_train_epochs=None):
		if num_train_epochs is None:
			num_train_epochs = self.args.num_train_epochs
		best_result = None
		best_model_path = None
		for epoch in range(num_train_epochs):
			print('Epoch %d - %s' % (epoch + 1, datetimestr()))
			self.optimizer.zero_grad()
			# iterations = 1
			numpy.random.shuffle(self.train_data)
			_run_train(self.args, self.device, self.model, self.optimizer, self.train_loss, self.train_data, None, self.train_plot)
			latest_model_path = self.model_path_base + '_latest'
			_save_model(self.model, latest_model_path, None)

			if (epoch + 1) % self.args.eval_frequency == 0:
				eval_info, eval_output = self.eval()
				eval_info = self.eval_plot.add(eval_info)
				self.eval_plot.plot()
				self.eval_plot.save_output(eval_output)
				eval_acc = eval_info.get('eval_acc')
				print('eval_acc', eval_acc)
				if eval_acc is not None and (best_result is None or best_result < eval_acc):
					best_result = eval_acc
					new_best_model_path = self.model_path_base + '_dev=%.2f' % best_result
					_save_model(self.model, new_best_model_path, best_model_path)
					best_model_path = new_best_model_path

	def eval(self):
		return _run_eval(self.args, self.model, self.dev_data, self.evaluate, self.device)


def _run_eval(args, model, test_data, eval_func, device):
	model.eval()
	eval_info = []
	eval_output = []
	iterations = len(test_data)
	for it in range(iterations):
		progress_bar(it, iterations)
		if args.test_batch_size == 1:
			data = test_data[it]
		else:
			data = test_data[it * args.test_batch_size : (it + 1) * args.test_batch_size]
		info, output = eval_func(args, model, data, device)
		eval_info.append(info)
		eval_output.extend(output)
	eval_info = InfoPlot.list_avg(eval_info)
	return eval_info, eval_output

def test(args, model, test_func, test_data, train_loss=None, model_prefix=''):
	set_seed(args.seed)
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	# model
	model_path_base = os.path.join(args.experiment_directory, '%smodel' % model_prefix)
	load_model(model_path_base, model)
	model = model.to(device)
	eval_plot = InfoPlot(os.path.join(args.experiment_directory, '%splot_test' % model_prefix))
	if train_loss is not None:
		_run_train(args, device, model, None, train_loss, test_data, None, eval_plot)
	eval_info, eval_output = _run_eval(args, model, test_data, test_func, device)
	eval_plot.add(eval_info)
	eval_plot.plot()
	eval_plot.save_output(eval_output)


def add_train_arguments(ap):
	ap.add_argument("--num_train_epochs", default=1500, type=int, help="Total number of training epochs to perform.")
	ap.add_argument("--eval_frequency", type=int, default=5)
	ap.add_argument("--plot_steps", type=int, default=50)	 
	ap.add_argument("--batch_size", type=int, default=1)
	ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
	ap.add_argument("--beam_size", type=int, default=3)
	# data
	ap.add_argument("--data_directory", default='_data', type=str)
	ap.add_argument("--input_directory", default='_data', type=str)
	ap.add_argument("--experiment_directory", default='_exp', type=str)

def get_argument_parser():
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("--device", type=str, default='cuda:0')
	# optim
	ap.add_argument("--max_weight_norm", type=float, default=100)
	ap.add_argument("--max_grad_norm", type=float, default=1)
	# train
	ap.add_argument("--seed", type=int, default=0)	
	ap.add_argument("--test", action='store_true')
	# beam
	return ap

