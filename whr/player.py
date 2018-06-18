from whr.playerday import PlayerDay
import math
import sys
import numpy as np

class Player():

	def __init__(self, name, config):
		self.name = name
		self.debug = config["debug"]
		self.w2 = (math.sqrt(config["w2"]) * math.log(10) / 400)**2
		self.days = []

	def log_likelihood(self):
		result = 0.0
		sigma2 = self.compute_sigma2()
		n = len(self.days)
		for i in range(n):
			prior = 0
			if i < (n - 1):
				rd = self.days[i].r - self.days[i + 1].r
				prior += (1 / math.sqrt(2 * math.pi))
			if i > 0:
				rd = self.days[i].r - self.days[i - 1].r
				prior += (1 / (math.sqrt(2 * math.pi *
						  sigma2[i - 1]))) * math.exp(-(rd**2) / 2 * sigma2[i - 1])
			if prior == 0:
				result += self.days[i].log_likelihood()
			else:
				if self.days[i].log_likelihood() >= sys.maxsize or math.log(prior) >= sys.maxsize:
					print(f"Infinity at {self.__str__()}: {self.days[i].log_likelihood()} + {math.log(prior)}: prior = {prior}, days = {self.days}")
					return
				result += self.days[i].log_likelihood() + math.log(prior)
		return result

	def hessian(self, days, sigma2):
		n = len(days)
		mat = np.zeros((n, n))
		for row in range(n):
			for col in range(n):
				if row == col:
					prior = 0
					if row < (n - 1):
						prior += -1.0 / sigma2[row]
					if row > 0:
						prior += -1.0 / sigma2[row - 1]
					mat[row, col] = days[row].log_likelihood_second_derivative() + \
																			   prior - 0.001
				elif row == col - 1:
					mat[row, col] = 1.0 / sigma2[row]
				elif row == col + 1:
					mat[row, col] = 1.0 / sigma2[col]
				else:
					mat[row, col] = 0
		return mat

	def gradient(self, r, days, sigma2):
		g = []
		n = len(days)
		for idx, day in enumerate(days):
			prior = 0
			if idx < (n - 1):
				prior += -(r[idx] - r[idx + 1]) / sigma2[idx]
			if idx > 0:
				prior += -(r[idx] - r[idx - 1]) / sigma2[idx - 1]
			if self.debug:
				print(f"g[{idx}] = {day.log_likelihood_derivative()} + {prior}")
			g.append(day.log_likelihood_derivative() + prior)
		return g

	def run_one_newton_iteration(self):
		for day in self.days:
			day.clear_game_terms_cache()
		if len(self.days) == 1:
			self.days[0].update_by_1d_newtons_method()
		elif len(self.days) > 1:
			self.update_by_ndim_newton()

	def compute_sigma2(self):
		sigma2 = []
		for d1,d2 in zip(*(self.days[i:] for i in range(2))):
			sigma2.append(abs(d2.day - d1.day)*self.w2)
		return sigma2

	def update_by_ndim_newton(self):
		# r
		r = [d.r for d in self.days]

		# sigma squared (used in the prior)
		sigma2 = self.compute_sigma2()
		
		h = self.hessian(self.days, sigma2)
		g = self.gradient(r, self.days, sigma2)
		n = len(r)  
		a = np.zeros((n,))
		d = np.zeros((n,))
		b = np.zeros((n,))
		d[0] = h[0,0]
		b[0] = h[0,1]
		
		for i in range(1,n):
			a[i] = h[i,i-1] / d[i-1]
			d[i] = h[i,i] - a[i] * b[i-1]
			if i < n-1:
				b[i] = h[i,i+1]


		y = np.zeros((n,))
		y[0] = g[0]
		for i in range(1,n):
			y[i] = g[i] - a[i] * y[i-1]
		
		x = np.zeros((n,))
		x[n-1] = y[n-1] / d[n-1]
		for i in range(n-2,-1,-1):
			x[i] = (y[i] - b[i] * x[i+1]) / d[i]    
		
		new_r = [ri-xi for ri,xi in zip(r,x)]
		
		for r in new_r:
			if r > 650:
				# raise UnstableRatingException, "Unstable r (#{new_r}) on player #{inspect}"
				raise Exception("unstable r on player")

		
		# if @debug
		#     puts "Hessian = #{h}"
		#     puts "gradient = #{g}"
		#     puts "a = #{a}"
		#     puts "d = #{d}"
		#     puts "b = #{b}"
		#     puts "y = #{y}"
		#     puts "x = #{x}"
		#     puts "#{inspect} (#{r}) => (#{new_r})"
		
		for idx, day in enumerate(self.days):
			day.r = day.r - x[idx]

	def covariance(self):
		r = [d.r for d in self.days]

		sigma2 = self.compute_sigma2()
		h = self.hessian(self.days, sigma2)
		g = self.gradient(r, self.days, sigma2)
		n = len(r)

		a = np.zeros((n,))
		d = np.zeros((n,))
		b = np.zeros((n,))
		d[0] = h[0,0]
		b[0] = h[0,1] if h.size>2 else 0

		for i in range(1,n):
			a[i] = h[i,i-1] / d[i-1]
			d[i] = h[i,i] - a[i] * b[i-1]
			if i < n-1:
				b[i] = h[i,i+1]

		dp = np.zeros((n,))
		dp[n-1] = h[n-1,n-1]
		bp = np.zeros((n,))
		bp[n-1] = h[n-1,n-2]
		ap = np.zeros((n,))
		for i in range(n-2,-1,-1):
			ap[i] = h[i,i+1] / dp[i+1]
			dp[i] = h[i,i] - ap[i]*bp[i+1]
			if i > 0:
				bp[i] = h[i,i-1]

		v = np.zeros((n,))
		for i in range(n-1):
			v[i] = dp[i+1]/(b[i]*bp[i+1] - d[i]*dp[i+1])
		v[n-1] = -1/d[n-1]

		mat = np.zeros((n,n))
		for row in range(n):
			for col in range(n):
				if row == col:
					mat[row,col] = v[row]
				elif row == col-1:
					mat[row,col] = -1*a[col]*v[col]
				else:
					mat[row,col] = 0

		return mat

	def update_uncertainty(self):
		if len(self.days) > 0:
			c = self.covariance()
			u = [c[i,i] for i in range(len(self.days))] # u = variance
			for i,d in enumerate(self.days):
				d.uncertainty = u[i]
			return None
		else:
			return 5

	def add_game(self,game):
		if len(self.days) == 0 or self.days[-1].day != game.day:
			new_pday = PlayerDay(self, game.day)
			if len(self.days) == 0:
				new_pday.is_first_day = True
				new_pday.set_gamma(1)
			else:
				new_pday.set_gamma(self.days[-1].gamma())
			self.days.append(new_pday)
		if game.white_player == self:
			game.wpd = self.days[-1]
		else:
			game.bpd = self.days[-1]
		self.days[-1].add_game(game)
