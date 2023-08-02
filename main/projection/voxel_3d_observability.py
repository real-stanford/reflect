import torch
import torch.nn as nn

from lgp.models.alfred.projection.projection_ops import project_3d_camera_points_to_2d_pixels, project_3d_points
from lgp.models.alfred.voxel_grid import VoxelGrid


class Voxel3DObservability(nn.Module):
    """
    Computes which voxels in the voxel grid are visible, by projecting each voxel back into the image.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                voxel_grid : VoxelGrid,
                extrinsics4f : torch.tensor, # B x 1 x 4 x 4
                depth_image : torch.tensor, # B x 1 x H x W
                hfov_deg : float):
        b, c, ih, iw = depth_image.shape
        _, _, w, l, h = voxel_grid.data.shape
        device = depth_image.device
        dtype = dtype = torch.float32 if "cpu" == str(voxel_grid.data.device) else torch.half
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

        # Map all out-of-bounds pixels to pixel (0, 0) in the image (just as a dummy value that's in-bounds)
        voxel_coordinates_cam_pixels = (voxel_coordinates_cam_pixels * voxels_in_image_bounds.int()).int()
        voxel_coordinates_cam_pixels = voxel_coordinates_cam_pixels[:, 0:2, :, :, :] # drop the z coordinate

        # Compute coordinates of each voxel into a 1D flattened image
        voxel_coordinates_in_flat_cam_pixels = voxel_coordinates_cam_pixels[:, 1:2, :, :, :] * iw + voxel_coordinates_cam_pixels[:, 0:1, :, :, :]
        flat_voxel_coordinates_in_flat_cam_pixels = voxel_coordinates_in_flat_cam_pixels.view([b, -1]).long()
        flat_depth_image = depth_image.view([b, -1])
        # Gather the depth image values corresponding to each "voxel"
        flat_voxel_ray_bounce_depths = torch.stack([torch.index_select(flat_depth_image[i], dim=0, index=flat_voxel_coordinates_in_flat_cam_pixels[i]) for i in range(b)])

        # Depth where a ray cast from the camera to this voxel hits an object in the depth image
        voxel_ray_bounce_depths = flat_voxel_ray_bounce_depths.view([b, 1, w, l, h]) # Unflatten
        # Depth of the voxel itself
        voxel_depths = voxel_coordinates_3d_cam_meters[:, 2:3, :, :, :]

        # Compute which voxels are observed by this camera taking depth image into account:
        #       All voxels along camera rays that hit an obstacle are considered observed
        #       Consider a voxel that's immediately behind an observed point as observed
        #       ... to make sure that we consider the voxels that contain objects as observed
        depth_bleed_tolerance = voxel_grid.voxel_size / 2
        voxel_observability_mask = torch.logical_and(
            voxel_depths <= voxel_ray_bounce_depths + depth_bleed_tolerance,
            voxels_in_image_bounds).long()

        # Consider all voxels that contain stuff as observed
        voxel_observability_mask = torch.max(voxel_observability_mask, (voxel_grid.data.max(1, keepdim=True).values > 0.2).long())
        voxel_observability_mask = voxel_observability_mask.long()

        # output
        observability_grid = VoxelGrid(voxel_observability_mask.type(dtype), # Observability
                                       voxel_observability_mask.type(dtype), # Occupancy (same as observability - consider observed voxels to be occupied)
                                       voxel_grid.voxel_size,
                                       voxel_grid.origin)

        return observability_grid, voxel_ray_bounce_depths