import os
import math
import numpy as np
import torch
torch.set_grad_enabled(False)
torch.manual_seed(0)

from point_cloud_utils import *
import open3d as o3d
from PIL import Image
from scene_graph import *
from utils import *

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def get_scene_graph(step_idx, event, object_list, total_points_dict, bbox3d_dict, obj_held_prev, task):
    pcd_dict, depth_dict  = {}, {}
    height, width, channel = event.frame.shape
    camera_space_xyz = depth_frame_to_camera_space_xyz(
            depth_frame=torch.as_tensor(event.depth_frame.copy()), mask=None, fov=event.metadata['fov'])
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']

    if not event.metadata['agent']['isStanding']:
        y = y - 0.22

    world_points = camera_space_xyz_to_world_xyz(
        camera_space_xyzs=camera_space_xyz,
        camera_world_xyz=torch.as_tensor([x, y, z]),
        rotation=event.metadata['agent']['rotation']['y'],
        horizon=event.metadata['agent']['cameraHorizon'],
    ).reshape(channel, height, width).permute(1, 2, 0)

    sinkbasin_pts = None
    for object_id in event.instance_masks:
        if object_id.split("|")[0] in ["Window", "Floor", "Wall", "Ceiling", "Cabinet"]:
            continue
        label = object_id

        # register this box in 3D
        mask = event.instance_masks[object_id].reshape(height, width)
        obj_points = torch.as_tensor(world_points[mask])
        
        if len(obj_points) < 700:
            continue
        
        depth_dict[label] = event.depth_frame[mask]
        # obj_colors = torch.as_tensor(world_colors[mask])

        # downsample point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=0.01)

        # denoise point cloud
        if "Pan" == label.split("|")[0] or "EggCracked" == label.split("|")[0] or "Bowl" == label.split("|")[0] or "Pot" == label.split("|")[0]:
            _, ind = voxel_down_pcd.remove_radius_outlier(nb_points=30, radius=0.03)
            inlier = voxel_down_pcd.select_by_index(ind)
            pcd_dict[label] = torch.tensor(np.array(inlier.points))
        elif "CounterTop" == label.split("|")[0]:
            _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
            inlier = voxel_down_pcd.select_by_index(ind)
            pcd_dict[label] = torch.tensor(np.array(inlier.points))
        elif "SinkBasin" in label:
            sinkbasin_pts = torch.tensor(np.array(voxel_down_pcd.points))
        else:
            pcd_dict[label] = torch.tensor(np.array(voxel_down_pcd.points))
            # assert points[label].shape == colors[label].shape
    #==============================================================

    for label in pcd_dict.keys():
        for keyword in ["Sliced", "Cracked"]:
            if keyword in label:
                tmp = ""
                for key in total_points_dict.keys():
                    if len(key.split("|")) == 4 and key.split("|")[0] == label.split("|")[0]:
                        tmp = key
                if len(tmp)!=0 and (keyword not in tmp) and (tmp in total_points_dict):
                    print("remove object:", tmp)
                    del total_points_dict[tmp]

        if label not in total_points_dict:
            total_points_dict[label] = pcd_dict[label]

        if is_receptacle(label, event):
            if is_moving(label, event) or is_picked_up(label, event) or obj_held_prev == label:
                total_points_dict[label] = pcd_dict[label]
            else:
                total_points_dict[label] = torch.unique(torch.cat((total_points_dict[label], pcd_dict[label]), 0), dim=0)
        else:
            total_points_dict[label] = pcd_dict[label]

        if label.split("|")[0] == "Sink" and sinkbasin_pts is not None:
            total_points_dict[label] = torch.unique(torch.cat((total_points_dict[label], sinkbasin_pts), 0), dim=0)
    
    # remove dropped object 
    if obj_held_prev not in pcd_dict.keys():
        if obj_held_prev in total_points_dict.keys():
            print("remove object:", obj_held_prev)
            del total_points_dict[obj_held_prev]

    for _, label in enumerate(total_points_dict.keys()):
        boxes3d_pts = o3d.utility.Vector3dVector(total_points_dict[label])
        box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
        bbox3d_dict[label] = box

    # Generate local scene graph
    local_sg = SceneGraph(event, task)
    for label in pcd_dict.keys():
        name = get_label_from_object_id(label, [event], task)
        bbox = get_2d_bbox_from_3d_pcd(event, label, total_points_dict)
        if name is not None and bbox is not None:
            node = Node(name, 
                        object_id=label, 
                        pos3d=bbox3d_dict[label].get_center(), 
                        corner_pts=np.array(bbox3d_dict[label].get_box_points()), 
                        bbox2d=bbox, 
                        pcd=total_points_dict[label],
                        depth=depth_dict[label])
            local_sg.add_node_wo_edge(node)

    # check if need to add a node and its edges for each object in the instance segmentation
    for label in pcd_dict.keys():
        object_name = label.split("|")[0]
        if object_name in object_list:
            node = next((node for node in local_sg.total_nodes if node.object_id == label), None)
            if node is not None:
                local_sg.add_node(node)

    obj_held_prev = local_sg.add_agent()
    
    return local_sg, total_points_dict, obj_held_prev, bbox3d_dict


def get_2d_bbox_from_3d_pcd(event, label, total_points_dict):
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']

    gt_mask = event.instance_masks[label].reshape(event.metadata["screenHeight"], event.metadata["screenWidth"])
    gt_mask_indices = np.where(gt_mask==1)

    mask_img = np.zeros_like(event.frame)
    pred_mask = np.array(world_space_xyz_to_2d_pixel(
        world_space_xyzs=total_points_dict[label].permute(1, 0),
        camera_world_xyz=torch.as_tensor([x, y, z]),
        rotation=event.metadata['agent']['rotation']['y'],
        horizon=event.metadata['agent']['cameraHorizon'],
        fov=event.metadata["fov"], 
        width=event.metadata["screenWidth"], 
        height=event.metadata["screenHeight"],
    ))
    
    try:
        mask_img[gt_mask_indices[0], gt_mask_indices[1]] = (0, 255, 0)
        pred_mask[1, :] = -pred_mask[1, :] + (event.metadata["screenHeight"]-1)

        valid_pixel = np.logical_and(pred_mask[0, :] >= 0,
            np.logical_and(pred_mask[0, :] < event.metadata["screenWidth"],
            np.logical_and(pred_mask[1, :] >= 0,
            pred_mask[1, :] < event.metadata["screenHeight"])))
        
        filtered_pred_mask = pred_mask[:, valid_pixel]

        # gt_bbox = (np.min(gt_mask_indices[0]), np.min(gt_mask_indices[1]), np.max(gt_mask_indices[0]), np.max(gt_mask_indices[1]))
        bbox = (np.min(filtered_pred_mask[1, :]), np.min(filtered_pred_mask[0, :]), np.max(filtered_pred_mask[1, :]), np.max(filtered_pred_mask[0, :]))
    except ValueError:
        return None
    return bbox


def save_pcd(folder_name, total_points_dict, camera_coord=False):
    total_points, total_colors = None, None
    os.system("mkdir -p scene/{}".format(folder_name))
    for i, label in enumerate(total_points_dict.keys()):
        if total_points is None:
            total_points = total_points_dict[label]
            c = torch.tensor(COLORS[i%len(COLORS)])
            total_colors = c.repeat(len(total_points_dict[label]), 1)
        else:
            total_points = torch.cat((total_points, total_points_dict[label]), 0)
            c = torch.tensor(COLORS[i%len(COLORS)])
            total_colors = torch.cat((total_colors, c.repeat(len(total_points_dict[label]), 1)), 0)

    # save pcd to file
    if total_points is not None:
        saved_pcd = o3d.geometry.PointCloud()
        saved_pcd.points = o3d.utility.Vector3dVector(total_points)
        saved_pcd.colors = o3d.utility.Vector3dVector(total_colors)
        if camera_coord:
            o3d.io.write_point_cloud("scene/{}/scene-cam.ply".format(folder_name), saved_pcd)
        else:
            o3d.io.write_point_cloud("scene/{}/scene.ply".format(folder_name), saved_pcd)
