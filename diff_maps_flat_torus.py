"""
Implementation of diffusion maps algorithm
Approximation of eigenvalues and eigenfunctions of the 0-Laplacian
uo to a constant scaling factor on the flat torus embedded in R4
"""

# %%
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import random
from numpy.linalg import eig as eig
from scipy.sparse.linalg import eigs as eigs
from scipy.spatial import distance_matrix
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import multiprocess as mp


# Parameters
I = 80          # Inner index for eigenfunctions
J = 70           # Outer index for eigenfunctions
K = 40           # Index for gradients of eigenfunctions
n = 100          # Number of approximated tangent vectors
N = 100         # Number of Monte Carlo training data points 

# epsilon = 0.31622777    # RBF bandwidth parameter
# epsilon = 0.30902954    # RBF bandwidth parameter
epsilon = 0.08

tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
a = 2           # Radius of the latitude circle of the torus
b = 0.5           # Radius of the meridian circle of the torus
R = 1
r = 1


"""
Training data set
with pushforward of vector fields v on the torus
and smbedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# the latotude and meridian circles with radius a and b
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 100, a = 1, b = 1):
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


sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(x = training_data_a[0,:], y = training_data_a[1,:], color = 'green')
ax1.set_xlim([-5,5])
ax1.set_ylim([-5,5])
ax1.set_title('Monte Carlo Sampled Latitude Circle with Radius a')

ax2.scatter(x = training_data_b[0,:], y = training_data_b[1,:], color = 'orange')
ax2.set_xlim([-5,5])
ax2.set_ylim([-5,5])
ax2.set_title('Monte Carlo Sampled Meridian Circle with Radius b')

plt.show()
# %%


# %%
# Parameterization functions specifying the coordinates in R^3
# corresponding to donut torus embedding
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
# %%

# %%
# Parameterization functions specifying the coordinates in R^4
# correspondong to flat torus embedding
# using the angles theat and rho for the latitude and meridian circles

X_func_flat = lambda theta: R*np.cos(theta)
Y_func_flat = lambda theta: R*np.sin(theta)
Z_func_flat = lambda rho: r*np.cos(rho)
W_func_flat = lambda rho: r*np.sin(rho)

# N*N training data points corrdinates in the x, y and z coordinates
TRAIN_X_flat = X_func_flat(training_angle[0, :])
TRAIN_Y_flat = Y_func_flat(training_angle[0, :])
TRAIN_Z_flat = Z_func_flat(training_angle[1, :])
TRAIN_W_flat = W_func_flat(training_angle[1, :])


# N*N training data points containing all four coordinates of each point
training_data_flat = np.vstack([TRAIN_X_flat, TRAIN_Y_flat, TRAIN_Z_flat, TRAIN_W_flat])
# %%




# %%
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)


# ax.plot_surface(x_t_ode, y_t_ode, z_t_ode, antialiased=True, alpha = 0.6, color='orange')

 
ax.scatter3D(training_data[0, :], training_data[1, :], training_data[2, :], color = "green")
plt.title("Solutions to ODE under the true system on the torus")
 
plt.show()
# %%



# x = (a + b*np.cos(training_angle[1, :]))*np.cos(training_angle[0, :])
# y = (a + b*np.cos(training_angle[1, :]))*np.sin(training_angle[0, :])
# z = b*np.sin(training_angle[1, :])



# Partial detivatives of the parameterization functions w.r.t. theta
# X_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.cos(rho)
# Y_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.sin(rho)
# Z_func_dtheta = lambda theta: b*np.cos(theta)


# Partial detivatives of the parameterization functions w.r.t. rho
# X_func_drho = lambda theta, rho: -(a + b*np.cos(theta))*np.sin(rho)
# Y_func_drho = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
# Z_func_drho = lambda theta: 0


# N*N partial derivative training data point corrdinates w.r.t. theta in the x, y and z coordinates
# TRAIN_X_dtheta = X_func_dtheta(training_angle[0, :], training_angle[1, :])
# TRAIN_Y_dtheta = Y_func_dtheta(training_angle[0, :], training_angle[1, :])
# TRAIN_Z_dtheta = Z_func_dtheta(training_angle[0, :])

# N*N partial derivative training data point corrdinates w.r.t. rho in the x, y and z coordinates
# TRAIN_X_drho = X_func_drho(training_angle[0, :], training_angle[1, :])
# TRAIN_Y_drho = Y_func_drho(training_angle[0, :], training_angle[1, :])
# TRAIN_Z_drho = Z_func_drho(training_angle[0, :])

# N*N linear combinations of partial derivatives of training data point corrdinates in the x, y and z coordinates
# TRAIN_X_DERIVATIVE = TRAIN_X_dtheta + TRAIN_X_drho
# TRAIN_Y_DERIVATIVE = TRAIN_Y_dtheta + TRAIN_Y_drho
# TRAIN_Z_DERIVATIVE = TRAIN_Z_dtheta + TRAIN_Z_drho


# N*N analytic directional coordinates of tangent vectors
# originated from N*N training data points in R^3
# ana_dir_coords = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, TRAIN_X_DERIVATIVE, TRAIN_Y_DERIVATIVE, TRAIN_Z_DERIVATIVE])
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
k = lambda x_1, x_2: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))


# Normalization function q corresponding to diagonal matrix Q
def make_normalization_func(k, x_train):
    def normalized(x):
        y = np.sum(k(x, x_train), axis = 1)
        return y
    return normalized


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
q = make_normalization_func(k, training_data_flat)
k_hat = make_k_hat(k, q)
K_hat = k_hat(training_data_flat, training_data_flat)
# print(K_hat[:2,:2])
# %%



# %%
# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data_flat)
D = d(training_data_flat)
# %%


# %%
# Markov kernel function p
def make_p(k_hat, d):
    def p(x, y):
        d_x = d(x).reshape(d(x).shape[0], 1)

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p

# Build Markov kernel matrix P
p = make_p(k_hat, d)
P = p(training_data_flat, training_data_flat)
# print(P[:3,:3])

print(np.trace(P))
print(np.pi/(4*epsilon**2))
# %%


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
S = s(training_data_flat, training_data_flat)
# print(S[:3,:3])
# %%


# %%
# Solve eigenvalue problem for similarity matrix S
# eigenvalues, eigenvectors = eigs(P, k = 300) 
eigenvalues, eigenvectors = eig(S)
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])
# %%


# %%
# Compute approximated 0-Laplacian eigengunctions
lambs = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            # lambs[i] = (4)*(-np.log(np.real(Lambs[i]))/(epsilon**2))
            lambs[i] = 4*(1 - np.real(Lambs[i]))/(epsilon**2)   

print(lambs)         



# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N**2, 2*I+1], dtype = float)
D_sqrt = np.power(D, (1/2))
for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.divide(np.real(Phis[:, j]), D_sqrt)

Phis_normalized = Phis_normalized/Phis_normalized[0, 0]

# %%


# %%
print(np.dot(Phis_normalized[:, 78], Phis_normalized[:, 78]))
# %%

# %%
print(lambs/(4*np.pi**2))
print(Phis_normalized[:, 0])
print(np.max(Phis_normalized[:, 0]))
print(np.min(Phis_normalized[:, 0]))
# %%



# %%
# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = np.real(phis / lambs)
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    return varphi

# Produce continuous extentions varphi_j for the eigenfunctions Phi_j
# varphi = make_varphi(p, training_data, Lambs, Phis_normalized)
varphi_flat = make_varphi(p, training_data_flat, Lambs, Phis_normalized)
# %%




# %%
# Apply the coninuous extensiom varphi to the training data set
# varphi_xyzw = varphi(training_data)
varphi_xyzw = varphi_flat(training_data_flat)

# print(varphi_xyz[:,3])
# %%

# %%
"""
Check accuracy of diffusion maps approximation
for eigenvalues and eigenfunctions of 0-Laplacian
"""


z_true = np.reshape(Phis_normalized[:, 1], (N, N))
z_dm = np.reshape(np.real(varphi_xyzw[:, 2]), (N, N))

plt.figure(figsize=(12, 12))
plt.pcolormesh(THETA_LST, RHO_LST, z_dm)

plt.show()
# %%


# %%
# Slice of the heat map
# for specific theta (latitude circle angle) values
y_test = np.reshape(varphi_xyzw[:, 160], (100, 100))

print(np.amax(y_test))
print(np.amin(y_test))


plt.scatter(u_a, y_test[0, :])
plt.show 
# %%