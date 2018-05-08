import sys

class Game:

	def __init__(self,black, white, winner, time_step, handicap = 0, extras = {}):
		self.day = time_step
		self.white_player = white
		self.black_player = black
		self.winner = winner
		self.extras = extras
		self.handicap = handicap
		self.handicap_proc = handicap

	def opponents_adjusted_gamma(self, player):
		if player == self.white_player:
			opponent_elo = self.bpd.elo() + self.handicap
		elif player == self.black_player:
			opponent_elo = self.wpd.elo() - self.handicap
		else:
			raise(AttributeError("No opponent for {}, since they're not in this game: {}.".format(player.__str__(), self.__str__())))
		rval = 10**(opponent_elo/400.0)
		if rval == 0 or rval > sys.maxsize:
			raise(AttributeError("bad adjusted gamma"))
		return rval

	def opponent(self, player):
		if player == self.white_player:
			return self.black_player
		elif player == self.black_player:
			return self.white_player

	def prediction_score(self):
		if self.white_win_probability() == 0.5:
			return 0.5
		else:
			return 1.0 if ((self.winner == "W" and self.white_win_probability() > 0.5) or (self.winner == "B" and self.white_win_probability()<0.5)) else 0.0

	def inspect(self):
		return "{} : W:#{white_player.name}(r={}) B:{}(r={}) winner = {}, komi = {}, handicap = {}".format(self.__str__, self.white_player.name, self.wpd.r if self.wpd is not None else '?', self.black_player.name, self.bpd.r if self.bpd is not None else '?',self.winner, self.komi, self.handicap)

	def white_win_probability(self):
		return self.wpd.gamma()/(self.wpd.gamma() + self.opponents_adjusted_gamma(self.white_player))

	def black_win_probability(self):
		return self.bpd.gamma()/(self.bpd.gamma() + self.opponents_adjusted_gamma(self.black_player))
