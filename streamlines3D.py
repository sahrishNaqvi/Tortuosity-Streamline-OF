from fluidfoam import readmesh, readvector
import numpy as np
from mayavi import mlab
import logging
from scipy.interpolate import griddata
import multiprocessing
import time
from mayavi.modules.streamline import Streamline

logging.basicConfig(filename='debug.log', level=logging.INFO)
case_path = '/home/sahrish/Downloads/3D_OF_Model_16x16x16_Re8.8587/OF_Model_8.8587'

# Read mesh data
logging.info('Start reading mesh...')
from fluidfoam import readmesh
x, y, z = readmesh(case_path)

# Load 3D velocity field data (U field)

from fluidfoam import readvector
timename = '500'
Ux, Uy, Uz = readvector(case_path, '500', 'U')

# Interpolation settings
ngridx, ngridy, ngridz = 100, 50, 50
xi, yi, zi = np.linspace(0, 66, ngridx), np.linspace(0, 16, ngridy), np.linspace(0, 16, ngridz)
xinterp, yinterp, zinterp = np.meshgrid(xi, yi, zi, indexing='ij')

# Parallel interpolation
def interpolate_velocity(args):
    return griddata((args[0], args[1], args[2]), args[3], (xinterp, yinterp, zinterp), method='linear')

pool = multiprocessing.Pool()

results = pool.map(interpolate_velocity, [
    (x, y, z, Ux),
    (x, y, z, Uy),
    (x, y, z, Uz)
])

pool.close()
pool.join()

velx_i, vely_i, velz_i = results
print(np.shape(results))

# handle NaN/Inf values
def handle_nans(data):
    nan_mask = np.isnan(data)
    data[nan_mask] = np.nanmean(data)
    return data

velx_i, vely_i, velz_i = handle_nans(velx_i), handle_nans(vely_i), handle_nans(velz_i)

# Streamlines visualization using Mayavi


# Create an empty array to store the seed points
y_values = np.arange(1, 15, 1)
z_values = np.arange(1, 15, 1)  # Adjust this as per your requirement

num_points = len(y_values) * len(z_values)
seed_points = np.zeros((3, num_points))

# Loop to create seed points with fixed x=0
index = 0
for y_value in y_values:
    for z_value in z_values:
        seed_points[0, index] = 20  # Fixed x-coordinate
        seed_points[1, index] = y_value
        seed_points[2, index] = z_value
        index += 1
print ('array of seed points=',seed_points)

# Plot the seed points
mlab.points3d(seed_points[0], seed_points[1], seed_points[2], scale_factor=0.5, color=(1, 0, 0))  # red color

# Create streamlines based on the vector field
src = mlab.pipeline.vector_field(velx_i, vely_i, velz_i)
streamline_obj = mlab.pipeline.streamline(src, seedtype='point')
streamline_obj.seed.widget.position = [seed_points[0, 0], seed_points[1, 0], seed_points[2, 0]]
streamline_obj.seed.widget.enabled = True

# Iterate over the other seed points and generate streamlines for each
for i in range(1, seed_points.shape[1]):
    streamline = mlab.pipeline.streamline(src, seedtype='point')
    streamline.seed.widget.position = [seed_points[0, i], seed_points[1, i], seed_points[2, i]]
    streamline.seed.widget.enabled = False
    streamline.stream_tracer.maximum_propagation = 17  # Adjust as necessary
    #streamline.stream_tracer.integration_step_length = 0.1  # Adjust as necessary

mlab.outline(extent=[0, 66, 0, 16, 0, 16], color=(0.7, 0.7, 0.7))

# Show the plot
mlab.show()
