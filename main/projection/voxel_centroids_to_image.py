from typing import Iterable

import torch
import torch.nn as nn

from lgp.models.alfred.voxel_grid import VoxelGrid
from lgp.models.alfred.projection.projection_ops import project_3d_camera_points_to_2d_pixels, project_3d_points, scatter_add_and_pool


class VoxelsToImage(nn.Module):
    """
    Projects a first-person image to a PointCloud
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                voxel_grid : VoxelGrid,
                extrinsics4f : torch.tensor, # B x 1 x 4 x 4
                image_shape : Iterable,
                hfov_deg : float):

        b, c, ih, iw = image_shape
        _, _, w, l, h = voxel_grid.data.shape
        device = voxel_grid.data.device
        extrinsics4f = extrinsics4f.to(device).float() # Projects from world coordinates to camera

        # Represent voxel grid by a point cloud of voxel centers
        voxel_coordinates_3d_world_meters = voxel_grid.get_centroid_coord_grid()

        # Project points into camera pixels
        voxel_coordinates_3d_cam_meters = project_3d_points(
            extrinsics4f, voxel_coordinates_3d_world_meters)
        voxel_coordinates_cam_pixels, pixel_z = project_3d_camera_points_to_2d_pixels(
            ih, iw, hfov_deg, voxel_coordinates_3d_cam_meters)

        # Compute a mask indicating which voxels are within camera FOV and in front of the camera
        voxels_in_image_bounds = (
            (voxel_coordinates_cam_pixels[:, 0:1, :, :, :] > 0) *
            (voxel_coordinates_cam_pixels[:, 0:1, :, :, :] < ih) *
            (voxel_coordinates_cam_pixels[:, 1:2, :, :, :] > 0) *
            (voxel_coordinates_cam_pixels[:, 1:2, :, :, :] < iw) *
            (pixel_z > 0)
        )

        # Map all out-of-bounds pixels to pixel 0, 0 in the image (just as a dummy value that's in-bounds)
        voxel_coordinates_cam_pixels = (voxel_coordinates_cam_pixels * voxels_in_image_bounds.int()).int()
        voxel_coordinates_cam_pixels = voxel_coordinates_cam_pixels[:, 0:2, :, :, :] # drop the z coordinate

        voxel_coordinates_in_flat_cam_pixels = voxel_coordinates_cam_pixels[:, 1:2, :, :, :] * iw + voxel_coordinates_cam_pixels[:, 0:1, :, :, :]
        flat_voxel_coordinates_in_flat_cam_pixels = voxel_coordinates_in_flat_cam_pixels.view([b, -1]).long()

        # Project all voxels that are "occupied", i.e. exist within the voxel grid, and are visible from the camera
        included_voxel_mask = voxels_in_image_bounds * voxel_grid.occupancy

        # Flatten spatial coordinates so that we can run the scatter operation
        image_data_flat = torch.zeros([b, c, ih * iw], dtype=torch.float32, device=device)
        voxel_data_flat = voxel_grid.data.view([b, c, -1])
        point_in_voxel_flat_coords = flat_voxel_coordinates_in_flat_cam_pixels.view([b, 1, -1])
        included_voxel_mask_flat = included_voxel_mask.view([b, 1, -1])

        image_data_pooled_flat, occupancy_pooled_flat = scatter_add_and_pool(
            image_data_flat,
            voxel_data_flat,
            included_voxel_mask_flat,
            point_in_voxel_flat_coords,
            pool="max"
        )
        image_data_pooled = image_data_pooled_flat.view([b, c, ih, iw])
        return image_data_pooled