import numpy as np

trans =  np.array([
    [0.03328, -0.077565, 0.307622, -0.428054],
    [-0.317168, -0.015161, 0.03049, 0.71483],
    [0.007206, -0.309045, -0.078703, 0.371978],
    [0.0, 0.0, 0.0, 1.0]
])

aabb = np.array([
    [-0.179, -0.0767, 0.0014],
    [0.1631, 0.2996, 0.3388]
])

aabb_transformed = np.linalg.inv(trans) @ np.vstack((aabb.T, np.ones((1,2))))
aabb_transformed = aabb_transformed[:3, :].T
x_lims = (min(aabb_transformed[:, 0]), max(aabb_transformed[:, 0]))
y_lims = (min(aabb_transformed[:, 1]), max(aabb_transformed[:, 1]))
z_lims = (min(aabb_transformed[:, 2]), max(aabb_transformed[:, 2]))
print("x lims", x_lims)
print("y lims", y_lims)
print("z lims", z_lims)

num_places = 4
print("x (smallest, width)", round(x_lims[0], num_places), round(x_lims[1]-x_lims[0], num_places))
print("y (smallest, width)", round(y_lims[0], num_places), round(y_lims[1]-y_lims[0], num_places))
print("z (smallest, width)", round(z_lims[0], num_places), round(z_lims[1]-z_lims[0], num_places))
