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
    assert (registration == [-1,0,1,2,3,4,-1]).all()

