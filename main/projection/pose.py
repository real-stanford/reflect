import uuid
import numpy as np

from transforms3d import quaternions
from transforms3d import euler
from transforms3d import affines


class Pose:
    def __init__(self, position, orientation,
                 frame="world", name=None):
        self.frame = frame
        self.name = name if name is not None else uuid.uuid4()
        self.position = position
        self.orientation = orientation

    @classmethod
    def make_identity_pose(cls, frame="world"):
        position = np.asarray([0, 0, 0])
        orientation = np.asarray([0, 0, 0, 1])
        return cls(position, orientation, frame)

    @classmethod
    def from_matrix_4f(cls, matrix4f, frame="world", name=None):
        position = matrix4f[0:3, 3].numpy()
        orientation = quaternions.mat2quat(matrix4f[:3, :3].numpy())
        return cls(position, orientation, frame, name)

    def to_matrix_4f(self):
        rotation = quaternions.quat2mat(self.orientation)
        mat = affines.compose(self.position, rotation, [1, 1, 1])
        return mat