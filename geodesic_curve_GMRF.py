''' Simulação de Monte Carlo via Metropolis para campos aleatórios Markovianos Gaussianos'''

import numpy as np
import scipy.misc as spm
import matplotlib.pyplot as plt
import time
import warnings
from mpl_toolkits.mplot3d import Axes3D
from imageio import imwrite
from numpy import log
from skimage.io import imsave

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

SIZE = 64		# original value was 128 (decreased to speed up)
MAX_IT = 10		# original value was 20 (decreased to speed up)

###############################################################################
# Generates an output of a pairwise isotropic Gaussian-Markov random field 
###############################################################################
def mcmc(media, variance, beta):
	# Initial configuration (sigma is the standard deviation)
	sigma = np.sqrt(variance)
	img = np.random.normal(mean, sigma, (SIZE, SIZE))
	# Boundary value problem
	K = 1
	img = np.lib.pad(img, ((K,K), (K,K)), 'symmetric')
	nlin, ncol = img.shape
	sample = img.copy()
	# Metropolis-Hastings
	for iteracoes in range(0, MAX_IT):
		# To extract the patches from the random field output
		amostras = np.zeros(((nlin-2)*(ncol-2), 9))
		ind = 0
		#print('Iteração ', iteracoes)
		# Main loop
		for i in range(K, nlin-K):
			for j in range(K, ncol-K):
				neigh = img[i-1:i+2, j-1:j+2]
				neigh = np.reshape(neigh, neigh.shape[0]*neigh.shape[1])
				amostras[ind,:] = neigh
				ind += 1
				vizinhanca = np.concatenate((neigh[0:(neigh.size//2)], neigh[(neigh.size//2)+1:neigh.size]))
				central = neigh[neigh.size//2]
				# Probability of the current xi
				P1 = (1/np.sqrt(2*np.pi*sigma**2))*np.exp((-1/(2*sigma**2))*(central - mean - beta*sum(vizinhanca - mean))**2)
				# Sample g from Gaussian
				g = np.random.normal(mean, sigma)
				# To avoid outliers
				while (g < mean - 3*sigma) or (g > mean + 3*sigma):	
					g = np.random.normal(mean, sigma)
				# Probability of the novel value g
				P2 = (1/np.sqrt(2*np.pi*sigma**2))*np.exp((-1/(2*sigma**2))*(g - mean - beta*(vizinhanca - mean).sum())**2)
				# Accept with probability P = P2/P1
				limiar = 1
				razao = P2/P1
				if (razao < 1):
					limiar = razao
				epson = np.random.rand()
				if epson <= limiar:
					sample[i, j] = g

		img = sample.copy()
	
	nucleo = img[K:nlin-K, K:ncol-K]

	return nucleo, amostras

#############################################################################################
# Computes the metric tensor of the parametric space (first-order Fisher information matrix)
#############################################################################################
def metric_tensor(samples, mean, variance, beta):  
	# Metric tensor
	tensor = np.zeros((3,3))
	# number of neighbors
	delta = 8
	# Computes the covariance matrix of the patches
	mc = np.cov(samples.T)	
	# Submatrix
	sigma_minus = mc.copy()
	sigma_minus[:,4] = 0
	sigma_minus[4,:] = 0
	# vector rho
	left_half = mc[4, :4]
	right_half = mc[4, 5:]
	rho = np.concatenate((left_half, right_half))
	# g_11
	g_11 = (1/variance)*((1 - beta*delta)**2)*(1 - (1/variance)*(2*beta*rho.sum() - (beta**2)*sigma_minus.sum()))
	tensor[0,0] = g_11
	# g_12 = g_21 = 0
	# g_13 = g_31 = 0
	# g_22
	rho_sig = np.kron(rho, sigma_minus)
	sig_sig = np.kron(sigma_minus, sigma_minus)
	g_22 = (1/(2*variance**2)) - (1/variance**3)*(2*beta*rho.sum() - (beta**2)*sigma_minus.sum()) + (1/variance**4)*(3*(beta**2)*sum(np.kron(rho, rho)) - 3*(beta**3)*rho_sig.sum() + 3*(beta**4)*sig_sig.sum())
	tensor[1,1] = g_22
	# g_23 = g_32
	g_23 = (1/variance**2)*(rho.sum() - beta*sigma_minus.sum()) - (1/(2*variance**3))*(6*beta*sum(np.kron(rho, rho)) - 9*(beta**2)*rho_sig.sum() + 3*(beta**3)*sig_sig.sum())
	g_32 = g_23
	tensor[1,2] = g_23
	tensor[2,1] = g_32
	# g_33
	T1 = (1/variance)*sigma_minus.sum()
	# Segundo termo
	T2 = (2/variance**2)*sum(np.kron(rho, rho))
	# Terceiro termo
	T3 = -6*beta*rho_sig.sum()/variance**2
	# Quarto termo
	T4 = 3*(beta**2)*sig_sig.sum()/variance**2
	g_33 = (T1+T2+T3+T4)
	tensor[2,2] = g_33
	# Regularization to avoid numerical issues (0.01)
	g = tensor + 0.01*np.eye(3)
	#g = tensor
	# Computes the entropy
	entropy = 0.5*(np.log(2*np.pi*variance) + 1) - (1/(variance))*( beta*rho.sum() - ((beta**2)/2)*sigma_minus.sum() )

	return g, entropy


########################################################################
# Computes the metric tensor of the parametric space
########################################################################
def Christoffel_symbols(samples, g, mean, variance, beta):  
	# Computes the inverse metric tensor
	# Regularization to avoid numerical issues
	#g_ = g + 0.01*np.eye(3)
	g_inv = np.linalg.inv(g)
	#print(g_inv)
	# Matrices
	Gamma1 = np.zeros((3,3))
	Gamma2 = np.zeros((3,3))
	Gamma3 = np.zeros((3,3))
	# number of neighbors
	delta = 8
	# Computes the covariance matrix of the patches
	mc = np.cov(samples.T)	
	# Submatrix
	sigma_minus = mc.copy()
	sigma_minus[:,4] = 0
	sigma_minus[4,:] = 0
	# vector rho
	left_half = mc[4, :4]
	right_half = mc[4, 5:]
	rho = np.concatenate((left_half, right_half))
	# Computes the Kronecker products
	rho_rho = np.kron(rho, rho)
	rho_sig = np.kron(rho, sigma_minus)
	sig_sig = np.kron(sigma_minus, sigma_minus)
	# The derivatives of the metric tensor components w.r.t theta_1 (mean) are zero!
	# Derivatives of the metric tensor components w.r.t theta_2 (sigma^2)
	dg_11_2 = ((-1/variance**2)*(1 - beta*delta)**2)*( 1 - (1/variance)*(2*beta*rho.sum() - (beta**2)*sigma_minus.sum()) ) + ((1/variance)*(1 - beta*delta)**2)*( (1/variance**2)*(2*beta*rho.sum() - (beta**2)*sigma_minus.sum()) )  
	dg_22_2 = -1/variance**3 + (3/variance**4)*(2*beta*rho.sum() - (beta**2)*sigma_minus.sum()) - (4/variance**5)*(3*beta**2*rho_rho.sum() - 3*beta**3*rho_sig.sum() + 3*beta**4*sig_sig.sum())
	dg_23_2 = (-2/variance**3)*(rho.sum() - beta*sigma_minus.sum()) + (3/(2*variance**4))*(6*beta*rho_rho.sum() - 9*beta**2*rho_sig.sum() + 3*beta**3*sig_sig.sum())
	dg_32_2 = dg_23_2
	dg_33_2 = (-1/variance**2)*sigma_minus.sum() - (2/variance**3)*(2*rho_rho.sum() - 6*beta*rho_sig.sum() + 3*beta**2*sig_sig.sum())   
	# Derivatives of the metric tensor components w.r.t theta_3 (beta)
	dg_11_3 = (-1/variance)*(2*delta)*(1 - beta*delta)*( 1 - (1/variance)*(2*beta*rho.sum() - (beta**2)*sigma_minus.sum()) ) - ((1/variance**2)*(1 - beta*delta)**2)*(2*rho.sum() - 2*beta*sigma_minus.sum())   
	dg_22_3 = (-1/variance**3)*(2*rho.sum() - 2*beta*sigma_minus.sum()) + (1/variance**4)*(6*beta*rho_rho.sum() - 9*beta**2*rho_sig.sum() + 12*beta**3*sig_sig.sum())
	dg_23_3 = (-1/variance**2)*sigma_minus.sum() - (1/(2*variance**3))*(6*rho_rho.sum() - 18*beta*rho_sig.sum() + 9*beta**2*sig_sig.sum())
	dg_32_3 = dg_23_3
	dg_33_3 = (1/variance**2)*(6*beta*sig_sig.sum() - 6*rho_sig.sum())

	# Components of Gamma1
	G_11_1 = -0.5*( dg_11_2*g_inv[1, 0] + dg_11_3*g_inv[2, 0] )
	G_12_1 = 0.5*( dg_11_2 )*g_inv[0, 0]
	G_13_1 = 0.5*( dg_11_3 )*g_inv[0, 0]
	G_21_1 = G_12_1
	G_22_1 = 0
	G_23_1 = 0
	G_31_1 = G_13_1
	G_32_1 = 0
	G_33_1 = 0
	Gamma1[0,0] = G_11_1
	Gamma1[0,1] = G_12_1
	Gamma1[0,2] = G_13_1
	Gamma1[1,0] = G_21_1
	Gamma1[1,1] = G_22_1
	Gamma1[1,2] = G_23_1
	Gamma1[2,0] = G_31_1
	Gamma1[2,1] = G_32_1
	Gamma1[2,2] = G_33_1
	# Components of Gamma2
	G_11_2 = -0.5*( dg_11_2*g_inv[1, 1] + dg_11_3*g_inv[2, 1] )
	G_12_2 = 0 
	G_13_2 = 0
	G_21_2 = 0
	G_22_2 = 0.5*( dg_22_2*g_inv[1, 1] + (2*dg_23_2 - dg_22_3)*g_inv[2, 1] )
	G_23_2 = 0.5*( dg_22_3*g_inv[1, 1] + dg_33_2*g_inv[2, 1] )
	G_31_2 = 0
	G_32_2 = G_23_2
	G_33_2 = 0.5*( (2*dg_32_3 - dg_33_2)*g_inv[1, 1] + dg_33_3*g_inv[2, 1] )
	Gamma2[0,0] = G_11_2
	Gamma2[0,1] = G_12_2
	Gamma2[0,2] = G_13_2
	Gamma2[1,0] = G_21_2
	Gamma2[1,1] = G_22_2
	Gamma2[1,2] = G_23_2
	Gamma2[2,0] = G_31_2
	Gamma2[2,1] = G_32_2
	Gamma2[2,2] = G_33_2
	# Components of Gamma3
	G_11_3 = -0.5*( dg_11_2*g_inv[1, 2] + dg_11_3*g_inv[2, 2] )
	G_12_3 = 0
	G_13_3 = 0
	G_21_3 = 0
	G_22_3 = 0.5*( dg_22_2*g_inv[1, 2] + (2*dg_23_2 - dg_22_3)*g_inv[2, 2] )
	G_23_3 = 0.5*( dg_22_3*g_inv[1, 2] + dg_33_2*g_inv[2, 2] )
	G_31_3 = 0
	G_32_3 = G_23_3
	G_33_3 = 0.5*( (2*dg_32_3 - dg_33_2)*g_inv[1, 2] + dg_33_3*g_inv[2, 2] )
	Gamma3[0,0] = G_11_3
	Gamma3[0,1] = G_12_3
	Gamma3[0,2] = G_13_3
	Gamma3[1,0] = G_21_3
	Gamma3[1,1] = G_22_3
	Gamma3[1,2] = G_23_3
	Gamma3[2,0] = G_31_3
	Gamma3[2,1] = G_32_3
	Gamma3[2,2] = G_33_3
	
	return Gamma1, Gamma2, Gamma3

###########################################################################
# System of differential equations
# Equations that define the system of first-order differential equations
###########################################################################
def F(t, alpha1, alpha2, alpha3):
	return alpha1

def G(t, alpha1, alpha2, alpha3):
	return alpha2

def H(t, alpha1, alpha2, alpha3):
	return alpha3

def P(t, alpha1, alpha2, alpha3, Gamma1):
	alpha = np.array([alpha1, alpha2, alpha3])
	return -np.dot(alpha, np.dot(Gamma1, alpha))

def Q(t, alpha1, alpha2, alpha3, Gamma2):
	alpha = np.array([alpha1, alpha2, alpha3])
	return -np.dot(alpha, np.dot(Gamma2, alpha))

def R(t, alpha1, alpha2, alpha3, Gamma3):
	alpha = np.array([alpha1, alpha2, alpha3])
	return -np.dot(alpha, np.dot(Gamma3, alpha))


###########################################################################
## Fourth-order Runge-Kutta method
# a, b: interval in t
# n: number of steps
###########################################################################
def RK4(a, b, n, mean, variance, beta, alpha1, alpha2, alpha3):
	# To store the information along the simulation
	mean_vector = np.zeros(n)
	variance_vector = np.zeros(n)
	beta_vector = np.zeros(n)
	alpha1_vector = np.zeros(n)
	alpha2_vector = np.zeros(n)
	alpha3_vector = np.zeros(n)
	entropy_vector = np.zeros(n)
	geodesic_vector = np.zeros(n)
	# Computes the step size
	h = (b - a)/n
	print('Parameters:')
	print('a = %f' %a)
	print('b = %f' %b)
	print('n = %f' %n)
	print('h = %f' %h)
	print()
	# Set up the initial parameter values
	t = a
	# Geodesic distance
	geodist = 0
	# Main loop: computations
	for i in range(n):
		print('Iteration %d' %i)
		# First: generation of an output of the model
		output, samples = mcmc(mean, variance, beta)
		# Save image file
		if i == 0:
			A = np.uint8(255*(output - output.min())/(output.max() - output.min()))
			imsave('A.png', A)
		elif i == n-1:
			B = np.uint8(255*(output - output.min())/(output.max() - output.min()))
			imsave('B.png', B)
		# Second: computation of the metric tensor
		g, entropy = metric_tensor(samples, mean, variance, beta)
		entropy_vector[i] = entropy
		# Third: computation the Christoffel symbols
		G1, G2, G3 = Christoffel_symbols(samples, g, mean, variance, beta)
		# Fourth-order variables
		# Part one
		k0 = h*F(t, alpha1, alpha2, alpha3)
		l0 = h*G(t, alpha1, alpha2, alpha3)
		m0 = h*H(t, alpha1, alpha2, alpha3)
		x0 = h*P(t, alpha1, alpha2, alpha3, G1)
		y0 = h*Q(t, alpha1, alpha2, alpha3, G2)
		z0 = h*R(t, alpha1, alpha2, alpha3, G3)
		# Part two
		k1 = h*F(t+0.5*h, alpha1+0.5*k0, alpha2+0.5*l0, alpha3+0.5*m0)
		l1 = h*G(t+0.5*h, alpha1+0.5*k0, alpha2+0.5*l0, alpha3+0.5*m0)
		m1 = h*H(t+0.5*h, alpha1+0.5*k0, alpha2+0.5*l0, alpha3+0.5*m0)
		x1 = h*P(t+0.5*h, alpha1+0.5*k0, alpha2+0.5*l0, alpha3+0.5*m0, G1)
		y1 = h*Q(t+0.5*h, alpha1+0.5*k0, alpha2+0.5*l0, alpha3+0.5*m0, G2)
		z1 = h*R(t+0.5*h, alpha1+0.5*k0, alpha2+0.5*l0, alpha3+0.5*m0, G3)
		# Part three
		k2 = h*F(t+0.5*h, alpha1+0.5*k1, alpha2+0.5*l1, alpha3+0.5*m1)
		l2 = h*G(t+0.5*h, alpha1+0.5*k1, alpha2+0.5*l1, alpha3+0.5*m1)
		m2 = h*H(t+0.5*h, alpha1+0.5*k1, alpha2+0.5*l1, alpha3+0.5*m1)
		x2 = h*P(t+0.5*h, alpha1+0.5*k1, alpha2+0.5*l1, alpha3+0.5*m1, G1)
		y2 = h*Q(t+0.5*h, alpha1+0.5*k1, alpha2+0.5*l1, alpha3+0.5*m1, G2)
		z2 = h*R(t+0.5*h, alpha1+0.5*k1, alpha2+0.5*l1, alpha3+0.5*m1, G3)
		# Part four
		k3 = h*F(t+h, alpha1+k2, alpha2+l2, alpha3+m2)
		l3 = h*G(t+h, alpha1+k2, alpha2+l2, alpha3+m2)
		m3 = h*H(t+h, alpha1+k2, alpha2+l2, alpha3+m2)
		x3 = h*P(t+h, alpha1+k2, alpha2+l2, alpha3+m2, G1)
		y3 = h*Q(t+h, alpha1+k2, alpha2+l2, alpha3+m2, G2)
		z3 = h*R(t+h, alpha1+k2, alpha2+l2, alpha3+m2, G3)
		# Updating
		mean = mean + (1/6)*(k0 + 2*k1 + 2*k2 + k3)
		mean_vector[i] = mean
		print('Mean = %f' %mean)
		variance = variance + (1/6)*(l0 + 2*l1 + 2*l2 + l3)
		variance_vector[i] = variance
		print('Variance = %f' %variance)
		beta = beta + (1/6)*(m0 + 2*m1 + 2*m2 + m3)
		beta_vector[i] = beta
		print('Beta = %f' %beta)
		print('Entropy = %f' %entropy)
		print()
		alpha1 = alpha1 + (1/6)*(x0 + 2*x1 + 2*x2 + x3)
		alpha1_vector[i] = alpha1
		alpha2 = alpha2 + (1/6)*(y0 + 2*y1 + 2*y2 + y3)
		alpha2_vector[i] = alpha2
		alpha3 = alpha3 + (1/6)*(z0 + 2*z1 + 2*z2 + z3)
		alpha3_vector[i] = alpha3
		t = t + h
		# Update the geodesic distance
		tangent = np.array([alpha1, alpha2, alpha3])
		ds = np.sqrt(np.dot(tangent, tangent)) 
		geodist = geodist + ds*abs(h)
		geodesic_vector[i] = geodist

	return (mean_vector, variance_vector, beta_vector, alpha1_vector, alpha2_vector, alpha3_vector, geodesic_vector, entropy_vector)


#################################################

if __name__ == '__main__':
	
	# Define parameters of the Runge-Kutta
	reverse = False		# If True the reverse process in time will be performed
	a = 0 				# initial time
	b = 6  				# final time
	n = 200  			# number of steps
	# Initial point
	mean = 10.0
	variance = 10.0
	beta = 0.0
	# Initial tangent vector
	alpha1 = -0.1
	alpha2 = 0.1
	alpha3 = 0.2
	# Print initial parameters
	print('INITIAL POINT IN THE MANIFOLD - A')
	print('Initial mean: %.2f' %mean)
	print('Initial variance: %.2f' %variance)
	print('Initial inverse temperature: %.2f' %beta)
	print()
	print('INITIAL TANGENT VECTOR')
	print('Direction 1 (mean): %.2f' %alpha1)
	print('Direction 2 (variance): %.2f' %alpha2)
	print('Direction 3 (inverse temperature): %.2f' %alpha3)
	print()
	# Fourth-order Runge-Kutta
	L = RK4(a, b, n, mean, variance, beta, alpha1, alpha2, alpha3)
	# Extract values
	means = L[0]
	variances = L[1]
	betas = L[2]
	alphas1 = L[3]
	alphas2 = L[4]
	alphas3 = L[5]
	geodesics = L[6]
	entropies = L[7]
	# Initial and final points
	A = np.array([means[0], variances[0], betas[0]])
	B = np.array([means[-1], variances[-1], betas[-1]])
	# Print results
	print('FINAL POINT IN THE MANIFOLD - B')
	print('Final mean: %f' %means[-1])
	print('Final variance: %f' %variances[-1])
	print('Final inverse temperature: %f' %betas[-1])
	print()
	print('FINAL TANGENT VECTOR')
	print('Direction 1 (mean): %f' %alphas1[-1])
	print('Direction 2 (variance): %f' %alphas2[-1])
	print('Direction 3 (inverse temperature): %f' %alphas3[-1])
	print()
	print('Geodesic distance from A to B: %f' %geodesics[-1])
	print('Euclidean distance from A to B: %f' %np.linalg.norm(A-B))
	print()
	# Plot the geodesic curve
	fig = plt.figure(1)
	ax = fig.gca(projection='3d')
	ax.plot(means, variances, betas, 'b')
	# If the reverse process in time will be performed
	if reverse:
		# Reverse the process in time
		LR = RK4(b, a, n, means[-1], variances[-1], betas[-1], alphas1[-1], alphas2[-1], alphas3[-1])
		# Extract values
		means_reverse = LR[0]
		variances_reverse = LR[1]
		betas_reverse = LR[2]
		alphas1_reverse = LR[3]
		alphas2_reverse = LR[4]
		alphas3_reverse = LR[5]
		geodesics_reverse = LR[6]
		entropies_reverse = LR[7]
		# Initial and final points
		B = np.array([means_reverse[0], variances_reverse[0], betas_reverse[0]])
		C = np.array([means_reverse[-1], variances_reverse[-1], betas_reverse[-1]])
		# Print results
		print('REVERSE POINT IN THE MANIFOLD - C')
		print('Final mean: %f' %means_reverse[-1])
		print('Final variance: %f' %variances_reverse[-1])
		print('Final inverse temperature: %f' %betas_reverse[-1])
		print()
		print('FINAL TANGENT VECTOR')
		print('Direction 1 (mean): %f' %alphas1_reverse[-1])
		print('Direction 2 (variance): %f' %alphas2_reverse[-1])
		print('Direction 3 (inverse temperature): %f' %alphas3_reverse[-1])
		print()
		print('Geodesic distance from B to C: %f' %geodesics_reverse[-1])
		print('Euclidean distance from B to C: %f' %np.linalg.norm(B-C))
		print()
		# Plot geodesic curve
		ax.plot(means_reverse, variances_reverse, betas_reverse, 'r')
		entropies = np.concatenate((entropies, entropies_reverse))
	# Show the graph in the screen
	plt.show()
	# Plot entropy
	plt.figure(2)
	plt.plot(entropies)
	plt.show()