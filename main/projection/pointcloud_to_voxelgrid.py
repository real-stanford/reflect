import torch
import torch.nn as nn
from voxel_grid import VoxelGrid
from projection.projection_ops import scatter_add_and_pool
from projection.constants import ROUNDING_OFFSET

# This many points need to land within a voxel for that voxel to be considered "occupied".
# Too low, and noisy depth readings can generate obstacles.
# Too high, and objects far away don't register in the map
MIN_POINTS_PER_VOXEL = 10


class PointcloudToVoxels(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, point_coordinates : torch.Tensor, point_attributes : torch.Tensor):

        dtype = torch.float32 if "cpu" == str(point_coordinates.device) else torch.half
        point_attributes = point_attributes.type(dtype)

        voxelgrid = VoxelGrid.create_empty(batch_size=point_coordinates.shape[0],
                                           channels=point_attributes.shape[1],
                                           device=point_coordinates.device,
                                           data_dtype=dtype)
        b, c, w, l, h = voxelgrid.data.shape

        # Compute which voxel coordinates (integer) each point falls within
        point_in_voxel_coords_f = (point_coordinates - voxelgrid.origin[:, :, None, None]) / voxelgrid.voxel_size

        point_in_voxel_coords = (point_in_voxel_coords_f + ROUNDING_OFFSET).long()

        # Compute a mask of which points land within voxel grid bounds
        min_bounds, max_bounds = voxelgrid.get_integer_bounds()
        point_in_bounds_mask = torch.logical_and(point_in_voxel_coords >= min_bounds[None, :, None, None],
                                                 point_in_voxel_coords < max_bounds[None, :, None, None])
        point_in_bounds_mask = point_in_bounds_mask.min(dim=1, keepdim=True).values  # And across all coordinates
        num_oob_points = (point_in_bounds_mask.int() == 0).int().sum().detach().cpu().item()
        print(num_oob_points)
        if num_oob_points > 20000:
            print(f"Number of OOB points: {num_oob_points}")

        # Convert coordinates into a flattened voxel grid
        point_in_voxel_flat_coords = point_in_voxel_coords[:, 0] * l * h + point_in_voxel_coords[:, 1] * h + point_in_voxel_coords[:, 2]

        # Flatten spatial coordinates so that we can run the scatter operation
        voxeldata_flat = voxelgrid.data.view([b, c, -1])
        point_data_flat = point_attributes.view([b, c, -1])
        point_in_voxel_flat_coords = point_in_voxel_flat_coords.view([b, 1, -1])
        point_in_bounds_mask_flat = point_in_bounds_mask.view([b, 1, -1])

        voxeldata_new_pooled, voxeloccupancy_new_pooled = scatter_add_and_pool(
            voxeldata_flat,
            point_data_flat,
            point_in_bounds_mask_flat,
            point_in_voxel_flat_coords,
            pool="max",
            occupancy_threshold=MIN_POINTS_PER_VOXEL
        )

        # Convert dtype to save space
        voxeloccupancy_new_pooled = voxeloccupancy_new_pooled.type(dtype)

        # Unflatten the results
        voxeldata_new_pooled = voxeldata_new_pooled.view([b, c, w, l, h])
        voxeloccupancy_new_pooled = voxeloccupancy_new_pooled.view([b, 1, w, l, h])

        voxelgrid_new = VoxelGrid(voxeldata_new_pooled, voxeloccupancy_new_pooled, voxelgrid.voxel_size, voxelgrid.origin)
        return voxelgrid_new
