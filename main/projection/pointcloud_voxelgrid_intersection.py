import torch
import torch.nn as nn
from lgp.models.alfred.voxel_grid import VoxelGrid

from lgp.models.alfred.projection.constants import ROUNDING_OFFSET


class PointcloudVoxelgridIntersection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, point_coordinates : torch.Tensor, voxelgrid: VoxelGrid):
        """
        Given a pointcloud arranged in a 2D spatial grid (B x (3 + C) x H x W), where the first 3 channels are 3D coordinates,
        and given a 3D voxel grid mask compute a mask of size Bx1xHxW indicating which points in the pointcloud intersect with
        voxels of value > 0.5.
        """
        #point_coordinates = pointcloud[0]#[:, :3, :, :]
        #point_attributes = pointcloud[1]#[:, 3:, :, :]
        b, c, w, l, h = voxelgrid.data.shape
        bp, _, ih, iw = point_coordinates.shape

        if b == 1 and bp > 1:
            voxelgrid__data = voxelgrid.data.repeat((bp, 1, 1, 1, 1))
            b = bp
        else:
            voxelgrid__data = voxelgrid.data

        # Compute which voxel coordinates (integer) each point falls within
        point_in_voxel_coords_f = (point_coordinates - voxelgrid.origin[:, :, None, None]) / voxelgrid.voxel_size
        point_in_voxel_coords = (point_in_voxel_coords_f + ROUNDING_OFFSET).long()

        # Compute a mask of which points land within voxel grid bounds
        min_bounds, max_bounds = voxelgrid.get_integer_bounds()
        point_in_bounds_mask = torch.logical_and(point_in_voxel_coords >= min_bounds[None, :, None, None],
                                                 point_in_voxel_coords < max_bounds[None, :, None, None])
        point_in_bounds_mask = point_in_bounds_mask.min(dim=1, keepdim=True).values  # And across all coordinates
        num_oob_points = (point_in_bounds_mask.int() == 0).int().sum().detach().cpu().item()
        #print(f"Number of OOB points: {num_oob_points}")

        # Mask out voxels that land out of hte map
        point_in_voxel_coords = point_in_voxel_coords * point_in_bounds_mask

        # Convert coordinates into a flattened voxel grid
        point_in_voxel_flat_coords = point_in_voxel_coords[:, 0] * l * h + point_in_voxel_coords[:, 1] * h + point_in_voxel_coords[:, 2]

        # Flatten spatial coordinates so that we can run the scatter operation
        voxeldata_flat = voxelgrid__data.view([b, c, -1])

        # Gather voxel data corresponding to each pointcloud point
        point_mask_flat = torch.gather(voxeldata_flat, dim=2, index=point_in_voxel_flat_coords.view([b, 1, -1]))
        point_mask = point_mask_flat.view([b, 1, ih, iw])

        return point_mask
