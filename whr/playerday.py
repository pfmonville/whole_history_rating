import numpy as np
import math
import sys


class PlayerDay:

    def __init__(self, player, day):
        self.day = day
        self.player = player
        self.is_first_day = False
        self.won_games = []
        self.lost_games = []

    def set_gamma(self, value):
        self.r = math.log(value)

    def gamma(self):
        return math.exp(self.r)

    def set_elo(slf, value):
        self.r = value * (math.log(10) / 400)

    def elo(self):
        return (self.r * 400) / (math.log(10))

    def clear_game_terms_cache(self):
        self._won_game_terms = None
        self._lost_game_terms = None

    def won_game_terms(self):
        if self._won_game_terms is None:
            self._won_game_terms = []
            for g in self.won_games:
                other_gamma = g.opponents_adjusted_gamma(self.player)
                if other_gamma == 0 or other_gamma is None or other_gamma > sys.maxsize:
                    print(f"other_gamma ({g.opponent(self.player).__str__()}) = {other_gamma}")
                self._won_game_terms.append([1.0, 0.0, 1.0, other_gamma])
            if self.is_first_day:
                # win against virtual player ranked with gamma = 1.0
                self._won_game_terms.append([1.0, 0.0, 1.0, 1.0])
        return self._won_game_terms

    def lost_game_terms(self):
        if self._lost_game_terms is None:
            self._lost_game_terms = []
            for g in self.lost_games:
                other_gamma = g.opponents_adjusted_gamma(self.player)
                if other_gamma == 0 or other_gamma is None or other_gamma > sys.maxsize:
                    print(f"other_gamma ({g.opponent(self.player).__(str__())}) = {other_gamma}")
                self._lost_game_terms.append([0.0, other_gamma, 1.0, other_gamma])
            if self.is_first_day:
                # win against virtual player ranked with gamma = 1.0
                self._lost_game_terms.append([0.0, 1.0, 1.0, 1.0])
        return self._lost_game_terms

    def log_likelihood_second_derivative(self):
        result = 0.0
        for a, b, c, d in self.won_game_terms() + self.lost_game_terms():
            result += (c * d) / ((c * self.gamma() + d)**2.0)
        return -1 * self.gamma() * result

    def log_likelihood_derivative(self):
        tally = 0.0
        for a, b, c, d in (self.won_game_terms() + self.lost_game_terms()):
            tally += c / (c * self.gamma() + d)
        return len(self.won_game_terms()) - self.gamma() * tally

    def log_likelihood(self):
        tally = 0.0
        for a, b, c, d in self.won_game_terms():
            tally += math.log(a * self.gamma())
            tally -= math.log(c * self.gamma() + d)
        for a, b, c, d in self.lost_game_terms():
            tally += math.log(b)
            tally -= math.log(c * self.gamma() + d)
        return tally

    def add_game(self, game):
        if (game.winner == "W" and game.white_player == self.player) or (game.winner == "B" and game.black_player == self.player):
            self.won_games.append(game)
        else:
            self.lost_games.append(game)

    def update_by_1d_newtons_method(self):
        dlogp = self.log_likelihood_derivative()
        d2logp = self.log_likelihood_second_derivative()
        dr = (self.log_likelihood_derivative() /
              self.log_likelihood_second_derivative())
        new_r = self.r - dr
        self.r = new_r
