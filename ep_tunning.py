"""
Autotuning for bandwidth parameter epsilon
"""

# %%
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import random
# %%

# Parameters
I = 80          # Inner index for eigenfunctions
J = 40           # Outer index for eigenfunctions
K = 25           # Index for gradients of eigenfunctions
n = 100          # Number of approximated tangent vectors
N = 100         # Number of Monte Carlo training data points 

# epsilon = 0.95    # RBF bandwidth parameter
alpha = 1       # Weight parameter for Markov kernel matrix
a = 5/3           # Radius of the latitude circle of the torus
b = 3/5           # Radius of the meridian circle of the torus


"""
Training data set
with pushforward of vector fields v on the torus
and smbedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# the latotude and meridian circles with radius a and b
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 100, a = 5/3, b = 3/5):
    u_a = np.arange(start_pt, end_pt, 2*np.pi/N)
    u_b = np.arange(start_pt, end_pt, 2*np.pi/N)
    
    # subsets = np.arange(0, N+1, (N/50))
    # for i in range(0, int(N/2)):
    #    start = int(subsets[i])
    #    end = int(subsets[i+1])
    #    u_a[start:end] = random.uniform(low = (i/(N/2))*end_pt, high = ((i+1)/(N/2))*end_pt, size = end - start)
    #    u_b[start:end] = random.uniform(low = (i/(N/2))*end_pt, high = ((i+1)/(N/2))*end_pt, size = end - start)
    
    # random.shuffle(u_a)
    # random.shuffle(u_b)

    training_data_a = np.empty([2, N], dtype = float)
    training_data_b = np.empty([2, N], dtype = float)
    
    for j in range(0, N):
            training_data_a[:, j] = np.array([a*np.cos(u_a[j]), a*np.sin(u_a[j])])
            training_data_b[:, j] = np.array([b*np.cos(u_b[j]), b*np.sin(u_b[j])])
    
    return u_a, u_b, training_data_a, training_data_b

u_a, u_b, training_data_a, training_data_b = monte_carlo_points()


# Create mesh of angles theta and rho for the latitude and meridian cricles
# and transform into grid of points with these two angles
THETA_LST, RHO_LST = np.meshgrid(u_a, u_b)

training_angle = np.vstack([THETA_LST.ravel(), RHO_LST.ravel()])
# %%


# %%
# Parameterization functions specifying the coordinates in R^3
# using the angles theat and rho for the latitude and meridian circles
X_func = lambda theta, rho: (a + b*np.cos(rho))*np.cos(theta)
Y_func = lambda theta, rho: (a + b*np.cos(rho))*np.sin(theta)
Z_func = lambda rho: b*np.sin(rho)

# N*N training data points corrdinates in the x, y and z coordinates
TRAIN_X = X_func(training_angle[0, :], training_angle[1, :])
TRAIN_Y = Y_func(training_angle[0, :], training_angle[1, :])
TRAIN_Z = Z_func(training_angle[1, :])

# N*N training data points containing all three coordinates of each point
training_data = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z])


# x = (a + b*np.cos(training_angle[1, :]))*np.cos(training_angle[0, :])
# y = (a + b*np.cos(training_angle[1, :]))*np.sin(training_angle[0, :])
# z = b*np.sin(training_angle[1, :])
# %%


# Parameterization functions specifying the coordinates in R^4
# correspondong to flat torus embedding
# using the angles theat and rho for the latitude and meridian circles
# %%
X_func_flat = lambda theta: a*np.cos(theta)
Y_func_flat = lambda theta: a*np.sin(theta)
Z_func_flat = lambda rho: b*np.cos(rho)
W_func_flat = lambda rho: b*np.sin(rho)

# N*N training data points corrdinates in the x, y and z coordinates
TRAIN_X_flat = X_func_flat(training_angle[0, :])
TRAIN_Y_flat = Y_func_flat(training_angle[0, :])
TRAIN_Z_flat = Z_func_flat(training_angle[1, :])
TRAIN_W_flat = W_func_flat(training_angle[1, :])


# N*N training data points containing all four coordinates of each point
training_data_flat = np.vstack([TRAIN_X_flat, TRAIN_Y_flat, TRAIN_Z_flat, TRAIN_W_flat])
# %%



# %%
"""
Functions utilized in the following program
"""

# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([(a + b*np.cos(rho))*np.cos(theta), (a + b*np.cos(rho))*np.sin(theta), b*np.sin(rho)])
v1F = lambda theta, rho: np.array([-b*np.sin(rho)*np.cos(theta) - (a + b*np.cos(rho))*np.sin(theta), -b*np.sin(rho)*np.sin(theta) + (a + b*np.cos(rho))*np.cos(theta), b*np.cos(rho)])

# Analytical tangent vector coordinates
ana_dir_coords = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, v1F(training_angle[0, :], training_angle[1, :])])


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
"""
Implementation of diffusion maps algorithm
Approximation of eigenvalues and eigenfunctions of the 0-Laplacian
uo to a constant scaling factor
"""


# Diffusion maps algorithm

# Heat kernel function k
k = lambda x_1, x_2, epsilon: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))

# Array of candidate epsilon index l
l_lst = np.arange(-3, 3.01, 0.01, dtype = float)

# Array of candidate epsilon
epsilon_lst = 10**l_lst

# Array of kernel matrix
# and array of the sum of kernel mateix elements

# K_lst = np.empty(np.shape(epsilon_lst)[0], dtype = float)
Sigma_lst = np.empty(np.shape(epsilon_lst)[0], dtype = float)

for l in range(0, np.shape(epsilon_lst)[0]):
    K_l = k(training_data, training_data, epsilon_lst[l])
    Sigma_lst[l] = (1/(N**2))*np.sum(K_l)
    
# %%

# %%
Sigma_prime_lst = np.empty(np.shape(epsilon_lst)[0], dtype = float)

for l in range(0, np.shape(epsilon_lst)[0] - 1):
    Sigma_prime_lst[l] = (np.log(Sigma_lst[l+1]) - np.log(Sigma_lst[l]))/(np.log(epsilon_lst[l+1]) - np.log(epsilon_lst[l]))

Sigma_max = np.max(Sigma_prime_lst[0:600])
l_max = np.where(Sigma_prime_lst == Sigma_max)

print(l_max)
print(epsilon_lst[l_max])
# %%


# %%
plt.figure(figsize=(60, 6))
plt.scatter(l_lst, Sigma_prime_lst, color = 'blue')

plt.xticks(np.arange(0.25, 0.31, 0.05, dtype = float))
plt.xlabel('l (power of 10)')
plt.yticks(np.arange(0, 4, 0.1, dtype = float))
plt.ylabel('Sigma prime (dimension of the manifold)')
plt.title('Autotuning Function for Epsilon')

plt.show()
# %%

# %%
print(Sigma_prime_lst[207])
# %%

# %%
Sigma_prime_lst_new = np.abs(Sigma_prime_lst - 2)
Sigma_min = np.min(Sigma_prime_lst_new)
l_max_new = np.where(Sigma_prime_lst_new == Sigma_min)

print(Sigma_min)
print(epsilon_lst[l_max_new])

print(epsilon_lst[208])

# %%
print(Sigma_prime_lst[269])
# %%
