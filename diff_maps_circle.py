"""
Implementation of diffusion maps algorithm
Approximation of eigenvalues and eigenfunctions of the 0-Laplacian
uo to a constant scaling factor on the unit circle S1
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy.linalg import eig as eig
import multiprocess as mp
from scipy.integrate import quad
from scipy.integrate import solve_ivp


# Parameters
I = 20          # Inner index for eigenfunctions
J = 5           # Outer index for eigenfunctions
K = 5           # Index for gradients of eigenfunctions
n = 8          # Number of approximated tangent vectors
N = 800         # Number of Monte Carlo training data points 

epsilon = 0.2  # RBF bandwidth parameter
tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
C = 1.5         # Component function parameter for vector field v



"""
Training data set
with pushforward of vector fields v on the circle
and Embedding map F with pushforward vF
"""


# Deterministically sampled Monte Carlo training data points
def monte_carlo_points(a = 0, b = 2*np.pi, N = 800):
    u = np.arange(a, b, 2*np.pi/N)
    # subsets = np.arange(0, N+1, N/400)
    # for i in range(0, 400):
    #    start = int(subsets[i])
    #    end = int(subsets[i+1])
    #    u[start:end] = random.uniform(low = (i/400)*b, high = ((i+1)/400)*b, size = end - start)
    # random.shuffle(u)
    
    training_data = np.empty([2, N], dtype = float)
    for j in range(0, N):
            training_data[:, j] = np.array([np.cos(u[j]), np.sin(u[j])])
    
    return u, training_data

u, training_data = monte_carlo_points()
plt.scatter(training_data[0,:], training_data[1,:])
plt.show


# n pushforward of vector field v ("arrows") on the circle
# given points (x, y) specified by angle theat on the circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))
X_func = lambda theta: np.cos(theta)
Y_func = lambda theta: np.sin(theta)
TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))


TRAIN_V = np.empty([n, 4], dtype = float)
for i in range(0, n):
    TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])
    # TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i] - 2*TRAIN_Y[i]*TRAIN_X[i], TRAIN_X[i] + 2*TRAIN_X[i]*TRAIN_X[i]])
    # TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -np.exp(2*TRAIN_X[i])*TRAIN_Y[i], np.exp(2*TRAIN_X[i])*TRAIN_X[i]])

X_1, Y_1, U_1, V_1 = zip(*TRAIN_V)

# print(U_1)
# print(V_1)


# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta: np.array([np.cos(theta), np.sin(theta)])
v1F = lambda theta: np.array([-np.sin(theta), np.cos(theta)])
v2F = lambda theta: np.array([-np.sin(theta) - C*np.sin(theta)*np.cos(theta), np.cos(theta) + C*(np.cos(theta))**2])
v3F = lambda theta: np.array([-np.exp(C*np.cos(theta))*np.sin(theta), np.exp(C*np.cos(theta))*np.cos(theta)])

# Functions used in finding fixed points of v i.e. roots of vF - F
# v1F1_root = lambda theta: -np.sin(theta) - np.cos(theta)
# v1F2_root = lambda theta: np.cos(theta) - np.sin(theta)
# v2F1_root = lambda theta: -np.sin(theta) - C*np.sin(theta)*np.cos(theta) - np.cos(theta)
# v2F2_root = lambda theta: np.cos(theta) + C*(np.cos(theta))**2 - np.sin(theta)
# v3F1_root = lambda theta: -np.exp(C*np.cos(theta))*np.sin(theta) - np.cos(theta)
# v3F1_root = lambda theta: np.exp(C*np.cos(theta))*np.cos(theta) - np.sin(theta)


# Component functions as part of the vector field v
h1 = lambda theta: 1 + C*np.cos(theta)
h2 = lambda theta: np.exp(C*np.cos(theta))                  # Jump function




"""
Functions utilized in the following program
"""


# Double and triple products of functions
def double_prod(f, g):
    def fg(x):
        return f(x) * g(x)
    return fg

def triple_prod(f, g, h):
    def fgh(x):
        return f(x) * g(x) * h(x)
    return fgh


# Distance matrix function
# Given two clouds of points in nD-dimensional space
# represented by the  arrays x_1 and x_2, respectively of size [nD, nX1] and [nD, nX2]
# y = dist_matrix(x_1, x_2) returns the distance array y of size [nX1, nX2] such that 
# y(i, j) = norm(x_1(:, i) - x_2(:, j))^2
def dist_matrix(x_1,x_2):
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    y = -2 * np.matmul(np.conj(x_1).T, x_2)
    w_1 = np.sum(np.power(x_1, 2), axis = 0)
    y = y + w_1[:, np.newaxis]
    w_2 = np.sum(np.power(x_2, 2), axis = 0)
    y = y + w_2
    return y

# %%



# %%

# Diffusion maps algorithm
# Normalization function q that corresponds to diagonal matrix Q
def make_normalization_func(k, x_train):
    def normalized(x):
        y = np.sum(k(x, x_train), axis = 1)
        return y
    return normalized

# Heat kernel function k
k = lambda x_1, x_2: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))

# Build kernel matrix K
# K = k(training_data, training_data)

# Normalized kernel function k_hat
def make_k_hat(k, q):
    def k_hat(x, y):
        q_x = q(x).reshape(q(x).shape[0], 1)
        q_y = q(y).reshape(1, q(y).shape[0])
        # treat qx as column vector
        k_hat_xy = np.divide(k(x, y), np.matmul(q_x, q_y))
        return k_hat_xy
    return k_hat

# Build normalized kernel matrix K_hat
q = make_normalization_func(k, training_data)
k_hat = make_k_hat(k, q)
K_hat = k_hat(training_data, training_data)
# print(K_hat[:3,:3])

# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data)
D = d(training_data)


# Markov kernel function p
def make_p(k_hat, d):
    def p(x, y):
        d_x = d(x).reshape(d(x).shape[0], 1)

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p

# Build Markov kernel matrix P
p = make_p(k_hat, d)
P = p(training_data, training_data)

print(np.sum(P, axis = 1))
# %%

# Similarity transformation function s
def make_s(p, d):
    def s(x, y):
        d_x = np.power(d(x).reshape(d(x).shape[0], 1), (1/2))
        d_y = np.power(d(y).reshape(1, d(y).shape[0]), (1/2))
        
        s_xy = np.divide(np.multiply(p(x, y), d_x), d_y)
        return s_xy
    return s

# Build Similarity matrix S
s = make_s(p, d)
S = s(training_data, training_data)
# print(S[:3,:3])


# Solve eigenvalue problem for similarity matrix S
eigenvalues, eigenvectors = eig(S) 
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])

# Compute approximated 0-Laplacian eigengunctions
lambs = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            lambs[i] = 4*(-np.log(np.real(Lambs[i]))/(epsilon**2)) 

print(lambs)         
# %%

# %%
x1 = np.arange(0, 9, 1).reshape((3,3))
x2 = np.arange(0, 6, 2).reshape((3,1))
print(np.multiply(x2, x1))
# %%



# %%
# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N, 2*I+1], dtype = float)
D_sqrt = np.power(D, (1/2))

for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.divide(np.real(Phis[:, j]), D_sqrt)

Phis_normalized = Phis_normalized/Phis_normalized[0, 0]

print(np.max(Phis_normalized[:, 0]))
print(np.min(Phis_normalized[:, 0]))

# %%


# %%
# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = phis / lambs
    
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    
    return varphi

# Produce continuous extentions varphi_j for the eigenfunctions Phi_j
Lambs_normalized = np.power(Lambs, 4)
varphi = make_varphi(p, training_data, Lambs, Phis_normalized)
# %%


# %%
"""
Check accuracy of diffusion maps approximation
for eigenvalues and eigenfunctions of 0-Laplacian
"""

# Check approximations for Laplacian eigenbasis agree with true eigenbasis
# by ploting against linear combinations of true eigenfunctions 

# Get x values of the sine wave
time = u
time2 = u

# Amplitude of the sine wave is sine of a variable like time
amplitude = Phis_normalized[:, 2]
amplitude2 = np.real(varphi(training_data)[:, 2])

# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time, amplitude, color = 'blue')
plt.scatter(time2, amplitude2, color = 'red')

# Give a title for the sine wave plot
plt.title('Sine wave')

# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.show()
# %%