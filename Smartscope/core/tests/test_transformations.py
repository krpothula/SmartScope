import numpy as np

from ..grid.transformations import register_targets_by_proximity

def test_register_targets_by_proximity():
    new_targets = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]]).astype(float)
    targets = np.array([[0,0],[1,1],[1.1,1.1],[2.2,2.2],[4,4],[5,5]]).astype(float)
    registration = register_targets_by_proximity(targets,new_targets)
    print(registration)
    assert (registration & [0,1,2,3,4]).any()
