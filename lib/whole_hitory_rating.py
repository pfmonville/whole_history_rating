from player import Player
from playerday import PlayerDay
from game import Game
from collections import defaultdict
import numpy as np
import time
class UnstableRatingException(Exception):
	pass

class Base:

	def __init__(self, config= None):
		if config is None:
			self.config = defaultdict(lambda: None)
		else:
			self.config = config
		if self.config.get("w2", None) is None:
			self.config["w2"] = 300.0
		self.games = []
		self.players = {}

	def print_ordered_ratings(self):
		players = [x for x in self.players.values() if len(x.days) > 0]
		players.sort(key=lambda x: x.days[-1].gamma())
		for p in players:
			if len(p.days) > 0:
				print("{} => {}".format(p.name, [x.elo() for x in p.days]))

	def get_ordered_ratings(self):
		result = []
		players = [x for x in self.players.values() if len(x.days) > 0]
		players.sort(key=lambda x: x.days[-1].gamma())
		for p in players:
			if len(p.days) > 0:
				result.append([x.elo() for x in p.days])
		return result

	def log_likelihood(self):
		score = 0.0
		for p in self.players.values():
			if len(p.days) > 0:
				score += p.log_likelihood()
		return score

	def player_by_name(self,name):
		if self.players.get(name, None) is None:
			self.players[name] = Player(name, self.config)
		return self.players[name]

	def ratings_for_player(self,name):
		player = self.player_by_name(name)
		return [[d.day, round(d.elo()), round(d.uncertainty*100)] for d in player.days]

	def setup_game(self,black,white,winner,time_step,handicap,extras={}):
		if black == white:
			raise(AttributeError("Invalid game (black player == white player)"))
			return None
		white_player = self.player_by_name(white)
		black_player = self.player_by_name(black)
		game = Game(black_player, white_player, winner, time_step, handicap, extras)
		return game

	def create_game(self, black, white, winner, time_step, handicap, extras = {}):
		game = self.setup_game(black, white, winner, time_step, handicap, extras)
		return self.add_game(game)

	def add_game(self, game):
		game.white_player.add_game(game)
		game.black_player.add_game(game)
		if game.bpd is None:
			print("Bad game")
		self.games.append(game)
		return game

	def iterate(self, count):
		for _ in range(count):
			self.run_one_iteration()
		for name, player in self.players.items():
			player.update_uncertainty()
		return None

	def auto_iterate(self, time_limit = 10):
		start = time.time()
		self.iterate(40)
		a = self.get_ordered_ratings()
		i = 40
		while True:
			self.iterate(10)
			i += 10
			b = self.get_ordered_ratings()
			if self.test_stability(a,b):
				return i
			if time.time() - start > time_limit:
				return False
			a = b

	def test_stability(self,v1,v2):
		v1 = [x for y in v1 for x in y]
		v2 = [x for y in v2 for x in y]
		for x1,x2 in zip(v1,v2):
			if abs(x2-x1) > 10E-3:
				return False
		return True

	def probability_for_the_match(self, black, white, time_step, handicap, extras = {}):
	  # Avoid self-played games (no info)
	  if black == white:
	    raise(AttributeError("Invalid game (black player == white player)"))
	    return None
	  white_player = self.player_by_name(white)
	  black_player = self.player_by_name(black)
	  game = Game(black_player, white_player, "unknown", time_step, handicap, extras)
	  new_pday_white = PlayerDay(white_player, game.day)
	  if len(white_player.days) == 0:
	  	new_pday_white.is_first_day = True
	  	new_pday_white.set_gamma(1)
	  else:
	  	new_pday_white.set_gamma(white_player.days[-1].gamma())
	  new_pday_black = PlayerDay(black_player, game.day)
	  if len(black_player.days) == 0:
	  	new_pday_black.is_first_day = True
	  	new_pday_black.set_gamma(1)
	  else:
	  	new_pday_black.set_gamma(black_player.days[-1].gamma())
	  game.wpd = new_pday_white
	  game.bpd = new_pday_black
	  print("win probability: {}:{}; {}:{}".format(black,game.black_win_probability(),white,game.white_win_probability()))
	  return game.black_win_probability(),game.white_win_probability()

	def run_one_iteration(self):
		for name,player in self.players.items():
			player.run_one_newton_iteration()

# whr = Base()
# whr.create_game("shusaku", "shusai", "B", 1, 0)
# whr.create_game("shusaku", "shusai", "W", 2, 0)
# c = whr.create_game("shusaku", "shusai", "W", 3, 0)
# a = whr.create_game("shusaku", "PF", "W", 3, 0)
# print(whr.auto_iterate())
# print(whr.ratings_for_player("shusaku"))
# print(whr.ratings_for_player("shusai"))
# whr.probability_for_the_match("shusai", "PF", 1, 0)
# print(a.bpd)


