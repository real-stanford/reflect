import torch.nn as nn
import torch

from projection.image_to_pointcloud import ImageToPointcloud
from projection.pointcloud_to_voxelgrid import PointcloudToVoxels


class ImageToVoxels(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_to_pointcloud = ImageToPointcloud()
        self.pointcloud_to_voxels = PointcloudToVoxels()

    def forward(self, scene, depth, extrinsics4f, hfov_deg, mark_agent=False):
        # CPU doesn't support most of the half-precision operations.
        if scene.device == "cpu":
            scene = scene.float()
            depth = depth.float()
        point_coords, img = self.image_to_pointcloud(scene, depth, extrinsics4f, hfov_deg)
        voxel_grid = self.pointcloud_to_voxels(point_coords, img)
        return voxel_grid