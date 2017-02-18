import math

#TODO: logsum trick

def poisson_distribution(gamma, k):
	return ( (gamma ** k) * math.e ** (-gamma) )/ math.factorial(k)