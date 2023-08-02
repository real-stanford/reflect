from typing import Iterable

import torch
import torch.nn as nn

from lgp.models.alfred.voxel_grid import VoxelGrid
from lgp.models.alfred.projection.projection_ops import project_3d_camera_points_to_2d_pixels, project_3d_points, scatter_add_and_pool
from lgp.models.alfred.projection.image_to_pointcloud import ImageToPointcloud
from lgp.models.alfred.projection.pointcloud_voxelgrid_intersection import PointcloudVoxelgridIntersection

from lgp.ops.depth_estimate import DepthEstimate


DISTR = True


class VoxelMaskToImageMask(nn.Module):
    """
    Projects a first-person image to a PointCloud
    """
    def __init__(self):
        super().__init__()
        self.image_to_pointcloud = ImageToPointcloud()
        self.pointcloud_voxel_intersection = PointcloudVoxelgridIntersection()

    def forward(self,
                voxel_grid : VoxelGrid,
                extrinsics4f : torch.tensor, # B x 1 x 4 x 4
                depth_image : torch.tensor,
                hfov_deg : float):

        if DISTR and isinstance(depth_image, DepthEstimate):
            domain = depth_image.domain_image()#res=100)
            b, c, h, w = domain.shape
            domain_b = domain.reshape((b*c, 1, h, w))
            domain_pc_b, domain_pa_b = self.image_to_pointcloud(
                    domain_b, domain_b, extrinsics4f, hfov_deg, min_depth=0.5)

            masks = self.pointcloud_voxel_intersection(domain_pc_b, voxel_grid)
            masks = masks.reshape((b, c, h, w))

            # Don't look at the depth to increase recall at expense of precision
            point_mask = masks.sum(dim=1, keepdims=True).clamp(0, 1)
        else:
            depth_images = [depth_image]

            # For each percentile depth, compute a corresponding pointcloud
            all_img_point_coords = []
            for dimg in depth_images:
                point_coordinates, point_attributes = self.image_to_pointcloud(
                    dimg, dimg, extrinsics4f, hfov_deg, min_depth=0.5)
                all_img_point_coords.append(point_coordinates)

            # Generate a fpv mask for each depth image
            point_masks = [self.pointcloud_voxel_intersection(pc, voxel_grid) for pc in all_img_point_coords]

            # Or the masks together
            point_mask = torch.stack(point_masks).max(dim=0).values

        return point_mask