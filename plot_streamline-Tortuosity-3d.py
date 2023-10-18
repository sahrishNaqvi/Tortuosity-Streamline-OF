import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Read the data file
data = np.loadtxt("/home/sahrish/Downloads/test_0.csv", delimiter=',', skiprows=1)


# Extract only the needed columns (x,y,u,v)
data = data[1:, 0:6]

# Discard domain margins in x-direction
x0 = 15
x1 = 45
data = data[np.abs(data[:, 0] - 0.5*(x0 + x1)) < 0.5*(x1 - x0)]

# Calculate velocity magnitude
vmag = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)

# Parameters
n_it = 300
max_length = 30
dt = max_length/n_it
v_th = 1e-10

# Normalize velocities
filtered_data = data[vmag > v_th]
x = filtered_data[:, 0]
y = filtered_data[:, 1]
z = filtered_data[:, 2]
u = filtered_data[:, 3] / vmag[vmag > v_th]
v = filtered_data[:, 4] / vmag[vmag > v_th]
w = filtered_data[:, 5] / vmag[vmag > v_th]

# Choose every which point to use for approximation
every = 100
idx = np.arange(0, len(u), every)
x, y, z, u, v, w = x[idx], y[idx], z[idx], u[idx], v[idx], w[idx]

# Interpolators for u,v,w velocities
int_u = Rbf(x, y, z, u)
int_v = Rbf(x, y, z, v)
int_w = Rbf(x, y, z, w)

# Define initial positions of the tracers
n_tr = int(5e2)
z_positions = np.random.rand(n_tr) * 16.0
y_positions = np.random.rand(n_tr) * 16.0
x_positions = np.ones(n_tr) * x0

x_positions = np.expand_dims(x_positions, axis=1)
y_positions = np.expand_dims(y_positions, axis=1)
z_positions = np.expand_dims(z_positions, axis=1)

for it in range(n_it):
    x_current = x_positions[:, it]
    y_current = y_positions[:, it]
    z_current = z_positions[:, it]

    xyz_combined = np.column_stack((x_current, y_current, z_current))

    x_new = x_current + dt * int_u(xyz_combined[:, 0], xyz_combined[:, 1], xyz_combined[:, 2])
    y_new = y_current + dt * int_v(xyz_combined[:, 0], xyz_combined[:, 1], xyz_combined[:, 2])
    z_new = z_current + dt * int_w(xyz_combined[:, 0], xyz_combined[:, 1], xyz_combined[:, 2])

    x_positions = np.column_stack((x_positions, x_new))
    y_positions = np.column_stack((y_positions, y_new))
    z_positions = np.column_stack((z_positions, z_new))

# Sieve out corrupted pathlines
print("Sieving out corrupted pathlines...")

# (1) Remove NaNs
good = []
for l in range(x_positions.shape[0]):
    skip = False
    for val in x_positions[l]:
        if np.isnan(val):
            skip = True
            break
    if not skip:
        good.append(l)

x_positions = x_positions[good]
y_positions = y_positions[good]
z_positions = z_positions[good]

# (2) Remove pathlines which fell outside of the domain
good = []
for l in range(x_positions.shape[0]):
    skip = False
    for k, x_val in enumerate(x_positions[l]):
        y_val = y_positions[l][k]
        z_val = z_positions[l][k]
        if y_val < 0 or y_val > 16 or z_val < 0 or z_val > 16:
            skip = True
            break
    if not skip:
        good.append(l)

x_positions = x_positions[good]
y_positions = y_positions[good]
z_positions = z_positions[good]

# Pathline plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for l in range(len(x_positions)):
    ax.plot(x_positions[l, :], y_positions[l, :], z_positions[l, :], c="k", linewidth=0.5, alpha=0.2)

# Plot cosmetics
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim([x0, x1])
ax.set_ylim([0, 16])
ax.set_zlim([0, 16])

# Calculate tortuosity
# ... Your code here ...

#def calculate_tortuosity(x_path, y_path, z_path):
 #   path_length = np.sum(np.sqrt(np.diff(x_path)**2 + np.diff(y_path)**2 + np.diff(z_path)**2))
  #  straight_line = np.sqrt((x_path[-1] - x_path[0])**2 + (y_path[-1] - y_path[0])**2 + (z_path[-1] - z_path[0])**2)
   # return path_length / straight_line
    
#tortuosities = []
#for l in range(len(x_positions)):
 #   tort = calculate_tortuosity(x_positions[l, :], y_positions[l, :], z_positions[l, :])
  #  tortuosities.append(tort)
    
#avg_tortuosity = np.mean(tortuosities)
#print(f"Average Tortuosity: {avg_tortuosity}")

def calculate_path_length_within_porous(x_path, y_path, z_path, porous_x0, porous_x1):
    path_length = 0.0
    for k in range(1, len(x_path)):
        if porous_x0 < x_path[k] < porous_x1:
            dx = x_path[k] - x_path[k-1]
            dy = y_path[k] - y_path[k-1]
            dz = z_path[k] - z_path[k-1]
            path_length += np.sqrt(dx**2 + dy**2 + dz**2)
    return path_length

porous_x0 = 20
porous_x1 = 36
L = []

for l in range(len(x_positions)):
    L_now = calculate_path_length_within_porous(x_positions[l, :], y_positions[l, :], z_positions[l, :], porous_x0, porous_x1)
    L.append(L_now)

tau = sum(L) / len(L) / (porous_x1 - porous_x0)
print(f"tau = {tau:.5f}")


plt.show()
