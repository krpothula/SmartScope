import numpy as np

from ..grid.transformations import register_targets_by_proximity

def test_register_targets_by_proximity():
    new_targets = np.array([[0,0],[1,1],[2,2],[2.2,2.2],[3,3],[4,4],[5,5]]).astype(float)
    targets = np.array([[0,0],[1,1],[1.1,1.1],[2.2,2.2],[4,4]]).astype(float)
    registration = register_targets_by_proximity(targets,new_targets)
    assert (registration == [0,1,2,3,5]).all()
    new_targets = np.array([[0,0],[1,1],[2,2],[2.2,2.2],[3,3]]).astype(float)
    targets = np.array([[0,0],[1,1],[1.1,1.1],[2.2,2.2],[4,4],[5,5]]).astype(float)
    registration = register_targets_by_proximity(targets,new_targets)
    assert (registration == [0,1,2,3,4,-1]).all()
    new_targets = np.array([[0,0],[1,1],[2,2],[2.2,2.2],[3,3]]).astype(float)
    targets = np.array([[-1,-1],[0,0],[1.9,1.9],[1.1,1.1],[2.1,2.1],[4,4],[5,5]]).astype(float)
    registration = register_targets_by_proximity(targets,new_targets)
    assert (registration == [-1,0,2,1,3,4,-1]).all()


def test_register_target_by_proximity_with_acculating_error():
    coordinates = np.column_stack((np.linspace(-12, 12, 25), np.linspace(-12, 12, 25)))
    print(coordinates, coordinates.shape)    # Copy and rotate the array by 3 degrees
    theta = np.radians(30)  # Convert degrees to radians
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    rotated_coordinates = coordinates @ rotation_matrix.T
    print(rotated_coordinates, rotated_coordinates.shape)
    registration = register_targets_by_proximity(coordinates, rotated_coordinates + 0.3)

