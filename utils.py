import os
import numpy as np
import motornet as mn
import datetime

def create_directory(directory_name=None):
    if directory_name is None:
        directory_name = datetime.datetime.now().date().isoformat()

    # Get the user's home directory
#    home_directory = os.path.expanduser("~")

    # Create the full directory path
#    directory_path = os.path.join(home_directory, "Documents", "Data","MotorNet", directory_name)

    directory_path = os.path.join(directory_name)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    # Return the created directory's name (whether it was newly created or already existed)
    return directory_path


def calculate_angles_between_vectors(vel, tg, xy):
    """
    Calculate angles between vectors X2 and X3.

    Parameters:
    - vel (numpy.ndarray): Velocity array.
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - angles (numpy.ndarray): An array of angles in degrees between vectors X2 and X3.
    """
    
    # Compute the magnitude of velocity and find the index to the maximum velocity
    vel_norm = np.linalg.norm(vel, axis=-1)
    idx = np.argmax(vel_norm, axis=1)

    tg = np.array(tg)
    xy = np.array(xy)

    # Calculate vectors X2 and X3
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]
    X3 = xy[np.arange(xy.shape[0]), idx, :]

    X2 = X2 - X1
    X3 = X3 - X1
    
    # Calculate the angles in degrees
    angles = np.degrees(np.arccos(np.sum(X2 * X3, axis=1) / (1e-8+np.linalg.norm(X2, axis=1) * np.linalg.norm(X3, axis=1)))).mean()

    return angles