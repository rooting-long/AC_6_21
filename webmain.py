
# FLASK_APP=webmain.py flask run
# FLASK_ENV=development FLASK_APP=webmain.py flask run

import os, json
from src import train, ccboard, MCTS, misc

args = train.get_args([])
device = train.get_device(args.device, 0)
model, _ = train.load_model(args, device, None, 'best_model_')
assert model is not None, 'Model file not found'

global_user_info = {}

# import webbrowser
# webbrowser.open("http://127.0.0.1:5000/")


############################################################

from flask import Flask, session, redirect, url_for, escape, request, jsonify

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
	return redirect(url_for('static', filename='index.html'))

@app.route('/json/<service_name>', methods=['GET', 'POST'])
def json_services(service_name):
	if service_name == 'initboard': return initboard()
	if service_name == 'isvalid': return isvalid()
	if service_name == 'review': return review()
	if service_name == 'oppomove': return oppomove()
	return jsonify({'error': 'No such service: %s' % service_name})

def _user_name():
	import random, string
	rand_name = None
	if 'rand_name' in session:
		rand_name = session['rand_name']
	else:
		rand_name = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(64)])
		session['rand_name'] = rand_name
	return rand_name

def initboard():
	user_name = _user_name()
	board = ccboard.ChessBoard()
	global global_user_info
	global_user_info[user_name] = { 'board' : board }
	return jsonify(board.state_dict())

def _get_board():
	user_name = _user_name()
	global global_user_info
	if user_name not in global_user_info:
		return {'reply':'无玩家信息'}, None, None
	user_info = global_user_info[user_name]
	if 'board' not in user_info: return {'reply':'无正在进行的棋局'}, None, None
	board = user_info['board']
	return None, user_info, board

def isvalid():
	error, user_info, board = _get_board()
	if error is not None: return jsonify(error)
	params = request.form
	chess_pos = (int(params['pos_x']), int(params['pos_y']))
	action = (str(params['chess_type']), (int(params['mov_x']), int(params['mov_y'])))
	valid, chess_removed = board.take_action(chess_pos, action)
	print('--> isvalid', chess_pos, action, valid)
	if not valid: return jsonify({'reply':False})
	if board.is_terminated():
		del user_info['board']
	return jsonify({'reply':True, 'terminated':board.is_terminated(), 'winner':board.is_winner() })

def _oppo_action(chess_pos, action):
	chess_pos = (8 - chess_pos[0], 9 - chess_pos[1])
	action = (action[0], (-action[1][0], -action[1][1]))
	return chess_pos, action

def oppomove():
	error, user_info, board = _get_board()
	if error is not None: return jsonify(error)
	board.switch_players()
	chess_pos, action = MCTS.next_action_for_evaluation(args, device, model, board)
	if isinstance(action, int):
		act_list = board.action_list()
		action = act_list[action]
	board.take_action(chess_pos, action)
	board.switch_players()
	chess_pos, action = _oppo_action(chess_pos, action)
	if board.is_terminated():
		del user_info['board']
	reply = {'pos_x':chess_pos[0], 'pos_y':chess_pos[1],
		'chess_type':action[0], 'mov_x':action[1][0], 'mov_y':action[1][1], 
		'terminated':board.is_terminated(), 'winner':board.is_winner() }
	# print(reply)
	return jsonify(reply)

def _list_reviewable_games():
	reviewable_games = {}
	game_path = os.path.join(args.experiment_directory, 'eval')
	for root, dirs, files in os.walk(game_path):
		for file in files:
			if file.endswith('.actions.json'):
				game_name = file.split('.')[0]
				actions_path = os.path.join(root, file)
				reviewable_games[game_name] = actions_path
	return reviewable_games

def _reviewable_games(reviewable_games):
	game_names = list(reviewable_games.keys())
	game_names.sort(reverse=True)
	for game_name in game_names:
		game_path = reviewable_games[game_name]
		with open(game_path) as fp:
			history = json.load(fp)
		actions = []
		for i, step in enumerate(history):
			action = step['a']
			print(action)
			if i % 2 == 1:
				action[0], action[1] = _oppo_action(action[0], action[1])
			actions.append(action)
		reviewable_games[game_name] = actions
	return game_names

def review():
	reviewable_games = _list_reviewable_games()
	game_names = _reviewable_games(reviewable_games)
	return jsonify({ 'game_names': game_names,
		'game_actions': reviewable_games,
		'init_board': ccboard.ChessBoard().state_dict() })

