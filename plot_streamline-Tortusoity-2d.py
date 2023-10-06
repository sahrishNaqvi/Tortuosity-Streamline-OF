
###############################################################################
# Reads the mesh
# .. note:: It reads the mesh coordinates and stores them in variables x, y
# and z

# import readmesh function from fluidfoam package     
#It defines a variable sol to store the path to the mesh file.
#It uses readmesh to read the mesh coordinates (x, y, z) from the specified directory.
from fluidfoam import readmesh

sol = 'path/to/working-directory/'

x, y, z = readmesh(sol)

###############################################################################
# Reads vector and scalar field
# -----------------------------
# .. note:: It reads vector and scalar field from an unstructured mesh
#           and stores them in vel and pressure variables
# import readvector and readscalar functions from fluidfoam package

from fluidfoam import readvector

timename = 'lastTime' # define lasttime of the solution directory
vel = readvector(sol, timename, 'U')

###############################################################################
# Interpolate the fields on a structured grid
# ----------------------------------------
# note:: The vector and scalar fields are interpolated on a specified
#           structured grid
#It imports necessary libraries like NumPy for array operations and SciPy's 
#griddata function for interpolation.
import numpy as np
from scipy.interpolate import griddata
from dipy.tracking.streamline import length
import matplotlib.pyplot as plt

#the number of divisions (ngridx and ngridy) for linear interpolation. 
ngridx = 10000
ngridy = 5000

# parameters for interpolation grid dimensions (xinterpmin, xinterpmax, yinterpmin, yinterpmax)
xinterpmin = 0
xinterpmax = 114
yinterpmin = 0
yinterpmax = 64

# Interpolation grid
xi = np.linspace(xinterpmin, xinterpmax, ngridx)
yi = np.linspace(yinterpmin, yinterpmax, ngridy)
print("shapes pf x,y,z,vel", np.shape(x), np.shape(y), np.shape(z), np.shape(vel))
# It creates a structured grid (xinterp, yinterp) using NumPy's meshgrid
xinterp, yinterp = np.meshgrid(xi, yi)

# Interpolation of vector field components on the structured grid using function griddata
velx_i = griddata((x, y), vel[0, :], (xinterp, yinterp), method='linear')
vely_i = griddata((x, y), vel[1, :], (xinterp, yinterp), method='linear')


#################################################################################
# It imports the matplotlib.pyplot library for plotting streamlines
# -------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Define plot parameters
fig = plt.figure(figsize=(5, 5), dpi=100)
plt.rcParams.update({'font.size': 10})
plt.xlabel('x/d')
plt.ylabel('y/d')
d = 1
L=64

#It defines seed points for streamlines, which are evenly spaced points along the y axis
#seed_points = np.array([[1, 1, 1, 1 ,1 ,1, 1, 1,1,1,1], [1,2,3,4,5,6,7,8,9,10, 63]])
y_values = np.arange(1, 64, 2)

# Create an empty array to store the seed points
seed_points = np.zeros((2, len(y_values))) #creates a NumPy array called seed_points with 
                                            #dimensions 2xN, where N is the length of 
                                            #the y_values array. 

# Loop to create seed points with fixed x=20 
#for loop iterates through the elements of the y_values array using the enumerate function


for i, y_value in enumerate(y_values):
    seed_points[0, i] = 20  # Fixed x-coordinate
    seed_points[1, i] = y_value
print ('array of seed points=',seed_points)

# Plot the seed points
plt.plot(seed_points[0], seed_points[1], 'bo')
#################################################################################
# Plots the streamlines of the interpolted vector field vel. 
# -------------------------------------------------------------------------------


#plt.plot(seed_points[0], seed_points[1], 'bo')
# Plots the streamlines
# ... [Previous code remains the same]

# Plots the streamlines
streamlines = plt.streamplot(xi/d, yi/d, velx_i, vely_i, color='k', density=70,
               linewidth=1, arrowsize=0.1, integration_direction='forward', start_points=seed_points.T)

plt.xlim([0, 114])
plt.ylim([0, 64])

# Cut-off x-value
cut_off_x = 84

# New total streamline length after cut-off
new_total_streamline_length = 0.0

# Access to paths of the streamlines and modify them
for streamline in streamlines.lines.get_paths():
    vertices = streamline.vertices
    # Find the index where streamline crosses x = cut_off_x
    cut_idx = np.where(vertices[:, 0] * d > cut_off_x)[0]

    # If there's an index where streamline crosses x = cut_off_x, cut it off
    if len(cut_idx) > 0:
        streamline.vertices = vertices[:cut_idx[0]]
        x_coords, y_coords = streamline.vertices[:, 0], streamline.vertices[:, 1]
        length_streamline = length([np.column_stack((x_coords, y_coords))])  # Calculate streamline length
        new_total_streamline_length += length_streamline
    else:  # if it doesn't cross, add the whole streamline length
        x_coords, y_coords = streamline.vertices[:, 0], streamline.vertices[:, 1]
        length_streamline = length([np.column_stack((x_coords, y_coords))])  # Calculate streamline length
        new_total_streamline_length += length_streamline

print("New Total Streamline Length:", new_total_streamline_length)
straight_line_distance = 64

# Calculating tortuosity
T = new_total_streamline_length / straight_line_distance/len(y_values)
print("Tortuosity (T):", T)
plt.show()
# ... [Rest of the code remains the same]

