
import numpy as np
from math import radians, cos
from scipy.spatial.distance import cdist

class ProcessImage:

    @staticmethod
    def closest_node(node, nodes, num=1):
        nodes = np.stack((nodes))
        cd = cdist(node, nodes)
        index = cd.argmin()
        dist = nodes[index] - node
        return index, dist

    @staticmethod
    def pixel_to_stage(dist, tile, tiltAngle=0):
        apix = tile.PixelSpacing / 10_000
        dist *= apix
        specimen_dist = ProcessImage.rotate_axis(dist,tile.RotationAngle)
        coords = tile.StagePosition + specimen_dist / \
            np.array([1, cos(radians(round(tiltAngle, 1)))])
        return np.around(coords, decimals=3)

    @staticmethod
    def rotate_axis(coord, angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rotated = np.sum(R * np.reshape(coord, (-1, 1)), axis=0)
        return rotated
    
    @staticmethod
    def pixel_to_stage_from_vectors(coord, transform_vector:str):
        # transform_vector = np.array(transform_vector.split(' ')).astype(float)
        x = np.sum(coord * transform_vector[:2]) + transform_vector[-2]
        y = np.sum(coord * transform_vector[2:4]) + transform_vector[-1]
        return np.array([x, y])