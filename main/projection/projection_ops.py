import torch
from projection.utils import make_pinhole_camera_matrix


def project_3d_points(mat4f, points):
    sdf = "wlhabcd"[0:len(points.shape) - 2]
    homo_ones = torch.ones_like(points[:, 0:1])
    points_projected = (
        torch.einsum(f"by{sdf},bxy->bx{sdf}",
                     torch.cat([points, homo_ones], dim=1),
                     mat4f)[:, 0:3])
    return points_projected


def project_3d_camera_points_to_2d_pixels(img_h, img_w, hfov_deg, points_3d_camera):
    batch_size = points_3d_camera.shape[0]
    device = points_3d_camera.device

    intrinsics = make_pinhole_camera_matrix(
        height_px=img_h,
        width_px=img_w,
        hfov_deg=hfov_deg
    )
    intrinsics = intrinsics.to(device)
    intrinsics = intrinsics[None, :, :].repeat((batch_size, 1, 1))

    # Spatial dimension string
    sdf = "wlhabcd"[0:len(points_3d_camera.shape)-2]

    tmp = torch.einsum(f"by{sdf},bxy->bx{sdf}", points_3d_camera, intrinsics)
    points_px_camera = tmp[:, 0:2] / (tmp[:, 2:3] + 1e-10)
    point_px_depths = tmp[:, 2:3]
    return points_px_camera, point_px_depths


def project_3d_world_points_to_2d_pixels(extrinsics4f, img_h, img_w, hfov_deg, points_3d_world):
    """
    Args:
        extrinsics4f: 4x4 camera extrinsics matrix that projects 3D points in world coordinates
            to 3d points in camera coordinates
        img_h, img_w, hfov_deg: image height and width in pixels, horizontal field of view in degrees
        points_3d_world: Bx3xAx(B)x(C) dimensional tensor of 3D coordinates, where A and optionally B, C are spatial dimensions
    Returns:
        Bx2xAx(B)x(C) tensor of 2D image pixel coordinates corresponding to each 3D point
    """
    points_3d_camera = project_3d_points(extrinsics4f, points_3d_world)
    points_px_camera, point_px_depths = project_3d_camera_points_to_2d_pixels(img_h, img_w, hfov_deg, points_3d_camera)
    return points_px_camera, point_px_depths


def project_2d_depth_to_3d():
    ...


def scatter_add_and_pool(dest_tensor, source_data, source_mask, src_dst_mapping, pool="max", occupancy_threshold=1.0):
    """
    dest_tensor: BxCxM tensor - destination tensor to add the information to
    source_data: BxCxN tensor - containing source data that will be scattered
    source_mask: Bx1xN tensor - indicating which source elements to copy
    src_dst_mapping: Bx1xN tensor - for each source data point,
        indicates the coordinate in dest_tensor where this data point should be added.
    """
    b, c, m = dest_tensor.shape
    _, _, n = source_data.shape

    # Destination tensor - voxel grid. Add a "counter" layer.
    dest_with_counters = torch.cat([dest_tensor,
                                    torch.zeros_like(dest_tensor[:, 0:1, :])], dim=1)

    # Source tensor - point cloud data. Add a "counter" layer
    src_with_counters = torch.cat([source_data,
                                   torch.ones_like(source_data[:, 0:1, :])], dim=1)

    # Mapping from source tensor to destination tensor.
    # Repeat across channels to specify that the same mapping is used for each channel.
    src_dst_mapping = src_dst_mapping.repeat((1, c + 1, 1))
    source_mask = source_mask.repeat((1, c + 1, 1)).int()

    # Mask out the points that shouldn't be projected
    src_dst_mapping = src_dst_mapping * source_mask
    src_with_counters = src_with_counters * source_mask

    # Project point cloud data onto the destination tensor, adding the features in case of collisions
    dest_with_counters = torch.scatter_add(
        dest_with_counters, dim=2, index=src_dst_mapping, src=src_with_counters)

    # Slice off the layer that counts how many pixels were projected on this voxel
    # ... and average voxel representations that landed on the voxel
    dest_data = dest_with_counters[:, 0:c, :]
    dest_counters = dest_with_counters[:, c:, :]

    if pool == "max":
        # Maximum of all objects within a voxel
        dest_data_pooled = dest_data.clamp(0, 1)
    else:
        # Average of all objects within a voxel
        dest_data_pooled = dest_data / (dest_counters + 1e-10)

    dest_occupancy_pooled = (dest_counters >= occupancy_threshold).int().float()
    return dest_data_pooled, dest_occupancy_pooled
