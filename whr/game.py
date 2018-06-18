import sys

class Game:

	def __init__(self,black, white, winner, time_step, handicap = 0, extras = None):
		self.day = time_step
		self.white_player = white
		self.black_player = black
		self.winner = winner
		self.handicap = handicap
		self.handicap_proc = handicap
		if extras is None:
			self.extras = dict()
			self.extras["komi"] = 6.5
		else:
			self.extras = extras
			if self.extras.get("komi") is None:
				self.extras["komi"] = 6.5


	def opponents_adjusted_gamma(self, player):
		if player == self.white_player:
			opponent_elo = self.bpd.elo() + self.handicap
		elif player == self.black_player:
			opponent_elo = self.wpd.elo() - self.handicap
		else:
			raise(AttributeError(f"No opponent for {player.__str__()}, since they're not in this game: {self.__str__()}."))
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
		return f"{self.__str__()} : W:{self.white_player.name}(r={self.wpd.r if self.wpd is not None else '?'}) B:{self.black_player.name}(r={self.bpd.r if self.bpd is not None else '?'}) winner = {self.winner}, komi = {self.extras['komi']}, handicap = {self.handicap}"

	def white_win_probability(self):
		return self.wpd.gamma()/(self.wpd.gamma() + self.opponents_adjusted_gamma(self.white_player))

	def black_win_probability(self):
		return self.bpd.gamma()/(self.bpd.gamma() + self.opponents_adjusted_gamma(self.black_player))
