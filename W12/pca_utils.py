import matplotlib.pyplot as plt
import numpy as np
import scipy
#import cv2 as cv

import os
from glob import glob

def read_shape_file(path):
    """Read shapes from asf files.

    Args:
        path: Path to asf data file.

    Returns:
        A tuple (point_list, types) where point_list is a list containing
        lists of points describing each face shape. Types are the shapes
        corresponding shape type (see DTU documentation for details).
    """
    with open(path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        num_points, *lines, _ = [
            line for line in lines if '#' not in line and line != ''
        ]

        num_points = int(num_points)
        points = [line.split() for line in lines]

        info = [{
            'path': int(point[0]),
            'coord': (float(point[2]), float(point[3])),
            'type': int(point[1])
        } for point in points]

        num_paths = max(info, key=lambda x: x['path'])['path']

        result = []
        types = []
        for i in range(num_paths + 1):
            ls = [e for e in info if e['path'] == i]
            result.append([e['coord'] for e in ls])
            types.append(ls[0]['type'])

        return result, types


def read_shape_file_vector(path):
    """Read shape file as point vector from asf files.

    Args:
        path: Path to asf data file.

    Returns:
        A single Nx2 array containing all points from the shape file.
    """
    shapes, _ = read_shape_file(path)
    res = np.concatenate(shapes)
    res = res - np.mean(res, axis=0)
    return res


# def read_image_files(path, scale=0.25):
#     image_files = glob(os.path.join(path, '*.jpg'))
#     images = [cv.imread(imf) for imf in image_files]
#     return [cv.resize(img, (0, 0), fx=scale, fy=scale) for img in images]


def face_shape_data(path):
    """Reads all shape (asf) files for the IMM dataset and images.

    Args:
        path: Path to IMM dataset folder.

    Returns:
        A tuple (shapes, images). Shapes is a NxM matrix where each row is a
        sample and M are the flattened points. Images are image arrays of the
        corresponding face images.
    """

    image_files = glob(os.path.join(path, '*.jpg'))
    bases = [os.path.splitext(os.path.basename(img))[0] for img in image_files]
    shape_files = [os.path.join(path, 'asf', f'{b}.asf') for b in bases]

    shapes = [read_shape_file_vector(sf).reshape(-1) for sf in shape_files]
    #images = [cv.imread(imf) for imf in image_files]
    images = [1,2]
    return np.array(shapes), np.array(images)


def plot_face(vec):
    r = vec.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r[:,0], r[:,1])
    ax.set_ylim((-0.3, 0.3))
    ax.set_xlim((-0.3, 0.3))
    ax.invert_yaxis()
    
def plot_many_faces(faces,name=None):
    """
    Plot multiple faces, takes a list of faces as input
    """
    
    plt.figure(figsize=(int(6*len(faces)),int(1+len(faces)//7)*6))
   
    if name!=None:
        plt.suptitle(name, fontsize=30, y=1.05)
    # Enumarate the ID, window name and faces passed as parameter.
    if len(faces) > 6:
        row = 6
    else:
        row = len(faces)
    for (pos, vec) in enumerate(faces):
        # Show the image in a new subplot.
        
        plt.subplot(int(1+len(faces)//6),row, pos + 1)
        title = "Face %d plot" %(pos)
        plt.title(title)
        r = vec.reshape(-1, 2)
        plt.scatter(r[:,0], r[:,1])
        plt.ylim((-0.3, 0.3))
        plt.xlim((-0.3, 0.3))
        plt.gca().invert_yaxis()

    # Show the face plots
    plt.show()



