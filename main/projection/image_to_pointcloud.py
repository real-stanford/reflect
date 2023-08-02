import os
import numpy as np
import torch
import torch.nn as nn
import imageio

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.depth import depth_to_3d
from projection.utils import make_pinhole_camera_matrix

# from lgp.utils.utils import standardize_image
# from lgp.paths import get_artifact_output_path

# import lgp.utils.render3d as r3d
# from lgp.utils.utils import save_gif


class ImageToPointcloud(nn.Module):
    """
    Projects a first-person image to a PointCloud
    """
    def __init__(self):
        super().__init__()

    def forward(self, camera_image, depth_image, extrinsics4f, hfov_deg, min_depth=0.7):
        batch_size = camera_image.shape[0]
        dev = camera_image.device
        b, c, h, w = camera_image.shape

        intrinsics = make_pinhole_camera_matrix(
            height_px=h,
            width_px=w,
            hfov_deg = hfov_deg
        )
        intrinsics = intrinsics.to(dev)
        intrinsics = intrinsics[None, :, :].repeat((batch_size, 1, 1))
        # Extrinsics project world points to camera
        extrinsics = extrinsics4f.to(dev).float()
        # Inverse extrinsics project camera points to the world
        inverse_extrinsics = extrinsics.inverse()
        # Repeat over batch if needed
        if inverse_extrinsics.shape[0] == 1 and b > 1:
            inverse_extrinsics = inverse_extrinsics.repeat((b, 1, 1))

        # Points3D - 1 x 3 x H x W grid of coordinates
        points_3d_wrt_camera = depth_to_3d(depth=depth_image,
                                           camera_matrix=intrinsics,
                                           normalize_points=True)

        has_depth = depth_image > min_depth

        # Project to world reference frame by applying the extrinsic homogeneous transformation matrix
        homo_ones = torch.ones_like(points_3d_wrt_camera[:, 0:1, :, :])
        homo_points_3d_wrt_camera = torch.cat([points_3d_wrt_camera, homo_ones], dim=1)
        homo_points_3d_wrt_world = torch.einsum("bxhw,byx->byhw", homo_points_3d_wrt_camera, inverse_extrinsics)
        points_3d_wrt_world = homo_points_3d_wrt_world[:, :3, :, :]

        # Plot to sanity check
        if False:
            imgid = 26
            outdir = os.path.join(get_artifact_output_path(), "3dviz")
            os.makedirs(outdir, exist_ok=True)
            img_w = r3d.render_aligned_point_cloud(points_3d_wrt_world, camera_image, animate=True)
            img_c = r3d.render_aligned_point_cloud(points_3d_wrt_camera, camera_image, animate=True)
            save_gif(img_w, os.path.join(outdir, f"pointcloud_test_global_{imgid}.gif"))
            save_gif(img_c, os.path.join(outdir, f"pointcloud_test_camera_{imgid}.gif"))
            #imageio.imsave(os.path.join(outdir, f"pointcloud_test_global_{imgid}.gif"), img_w)
            #imageio.imsave(os.path.join(outdir, f"pointcloud_test_camera_{imgid}.gif"), img_c)

            imageio.imsave(os.path.join(outdir, f"pointcloud_test_scene_{imgid}.png"), standardize_image(camera_image[0]))
            imageio.imsave(os.path.join(outdir, f"pointcloud_test_depth_{imgid}.png"), standardize_image(depth_image[0]))

            # o3d.visualization.draw_geometries([pcd])

        # points_3d_wrt_world_with_attrs = torch.cat([points_3d_wrt_world, camera_image], dim=1)
        # B x (3 + C) x H x W  : tensor of 3D points with color/feature attributes

        # Zero out the points that are at the camera location
        # points_3d_wrt_world_with_attrs = points_3d_wrt_world_with_attrs * has_depth

        points_3d_wrt_world = points_3d_wrt_world * has_depth
        camera_image = camera_image * has_depth

        return points_3d_wrt_world, camera_image