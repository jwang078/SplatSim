import numpy as np

def normalize(v):
    """
    Normalizes a 3D vector.
    
    Args:
        v (np.ndarray): A numpy array representing a 3D vector.
    
    Returns:
        np.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def get_rotation_matrix_from_look_at(camera_position, look_at_point, up_vector_given):
    """
    Calculates a 3x3 rotation matrix from a camera's look-at point and up vector.

    Args:
        camera_position (list or np.ndarray): The camera's (x, y, z) position.
        look_at_point (list or np.ndarray): The point the camera is looking at.
        up_vector_given (list or np.ndarray): The world's 'up' direction vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # 1. Convert inputs to numpy arrays for easier vector math
    camera_position = np.array(camera_position, dtype=float)
    look_at_point = np.array(look_at_point, dtype=float)
    up_vector_given = np.array(up_vector_given, dtype=float)

    # 2. Calculate the forward vector (the new z-axis of the camera)
    forward = normalize(look_at_point - camera_position)

    # 3. Calculate the right vector (the new x-axis of the camera)
    # This is the cross product of the given up vector and the forward vector.
    right = normalize(np.cross(up_vector_given, forward))

    # 4. Calculate the corrected up vector (the new y-axis of the camera)
    # This ensures the new up vector is perfectly orthogonal to the other two.
    up = np.cross(forward, right)
    
    # 5. Construct the rotation matrix
    # The columns of the rotation matrix are the new basis vectors.
    # Note: Some conventions use row vectors instead, but column vectors are common.
    rotation_matrix = np.array([right, up, forward]).T

    return rotation_matrix

if __name__ == "__main__":
    # TODO provide this information
    # It can be obtained from https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php?art=1&cu=0,1,0&cp=0,1,0&cla=1,0,0&aa=false&2d=false&sh=0
    # if you find a good view and then press I to see the debug info
    camera_pos = [-2.19024, -0.51942, 1.78499]
    look_at_pos = [1.52976, 2.27776, 1.65898]
    up_vec = [0.00000, -0.87991, -0.47515]

    # Calculate the rotation matrix
    R = get_rotation_matrix_from_look_at(camera_pos, look_at_pos, up_vec)

    # Print the resulting matrix
    print("The calculated rotation matrix R is:")
    print(R)
