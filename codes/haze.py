import numpy as np
from noise import pnoise2

def haze(frame,beta,A,n=0):

	
	if n==1:
		noise = np.random.normal(0,0.01,1)
		A+=noise
		noise = np.random.normal(0,0.1,1)
		beta+=noise

	tx=np.exp(-beta)
	I=frame*tx+A*(1-tx)
	#I/=np.amax(I)
	return I


def perlin_gen(frame,freq=35,var_freq=10,var_perlin=0.6,octaves=4):
	octaves = 4
	freq = (freq+np.random.normal(0,var_freq,1)) * octaves
	

	y_max = frame.shape[0]
	x_max = frame.shape[1]


	perlin= np.empty(frame.shape,dtype=float)
	offset=1-var_perlin/2
	for y in range(y_max):
	    for x in range(x_max):
	        val = pnoise2(x / freq, y / freq, octaves) * var_perlin+offset
	        perlin[y][x]= val/offset
	     
	return perlin
	

def perlin_haze(frame,perlin,beta,A,n=0):



	if n==1:
		noise = np.random.normal(0,0.02,1)
		A+=noise
		noise = np.random.normal(0,0.3,1)
		beta+=noise

	tx=np.exp(-beta*perlin)
	I=frame*tx+A*(1-tx)
	I/=1.0
	return I