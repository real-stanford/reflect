import numpy as np
from clip_utils import *
from utils import *
from constants import *
from point_cloud_utils import *


# =========  Parameters for spatial relation heuristics ============
IN_CONTACT_DISTANCE = 0.1
CLOSE_DISTANCE = 0.4
INSIDE_THRESH = 0.5
ON_TOP_OF_THRESH = 0.7
NORM_THRESH_FRONT_BACK = 0.9
NORM_THRESH_UP_DOWN = 0.9
NORM_THRESH_LEFT_RIGHT = 0.8
OCCLUDE_RATIO_THRESH = 0.5
DEPTH_THRESH = 0.9
# ==================================================================


state_dict = {
    "Fridge": ["open", "closed"],
    "Faucet": ["turned on", "turned off"],
    "Pot": ["filled with coffee", "filled with water", "filled with wine" "empty", "dirty", "clean"],
    "StoveBurner": ["turned on", "turned off"],
    "Egg": ['cracked', 'not cracked'],
    "Bread": ['sliced', 'not sliced'],
    "Toaster": ["turned on", "turned off"],
    "CoffeeMachine": ["turned on", "turned off"],
    "Mug": ["filled with coffee", "filled with water", "filled with wine" "empty", "dirty", "clean"],
    "HousePlant": ["watered", "not watered"],
    "Microwave": ["open", "closed", "turned on", "turned off"],
    "Television": ["turned on", "turned off"],
    "Laptop": ["open", "closed", "turned on", "turned off"],
    "Pan": ["clean", "dirty"],
    "Plate": ["clean", "dirty"]
}

# ref: https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py
def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1. get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin, 0.)
    ih = np.maximum(iymax-iymin, 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1]) +
           (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou, inters

def get_node_dist(pts_A, pts_B):
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pts_A)
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(pts_B)

    dists = pcd_A.compute_point_cloud_distance(pcd_B)
    dist = np.min(np.array(dists))
    return dist

def get_object_state(node_name, img):
    object_name = node_name.split("|")[0]
    if object_name in state_dict:
        states = state_dict[object_name]

        # rank based on current image
        img_feats = get_img_feats(img)
        state_feats = get_text_feats(states)
        sorted_states, sorted_scores = get_nn_text(states, state_feats, img_feats)
        return sorted_states[0]
    return None

def get_gt_object_state(node_name, event):
    object_name = node_name.split("|")[0]
    if object_name in state_dict and len(node_name.split("|")) == 4:
        gt_states = []
        for obj in event.metadata["objects"]:
            if node_name == obj["objectId"]:
                if obj["openable"]:
                    if obj["isOpen"]:
                        gt_states.append("open")
                    else:
                        gt_states.append("closed")
                if obj["sliceable"]:
                    if not obj["isSliced"]:
                        gt_states.append("not sliced")
                if node_name.split("|")[0] == "Egg":
                    if not obj['isBroken']:
                        gt_states.append("not cracked")
                if obj["canFillWithLiquid"]:
                    if obj["isFilledWithLiquid"]:
                            gt_states.append(f"filled with {obj['fillLiquid']}")
                    else:
                        gt_states.append("empty")
                if obj["toggleable"]:
                    if obj["isToggled"]:
                        gt_states.append("turned on")
                    else:
                        gt_states.append("turned off")
                if obj["dirtyable"]:
                    if obj["isDirty"]:
                        gt_states.append("dirty")
                    else:
                        gt_states.append("clean")
            elif obj["controlledObjects"] is not None and node_name in obj["controlledObjects"]:
                # print("ControlledObject: ", node_name, obj['objectId'])
                if obj["isToggled"]:
                    gt_states.append("turned on")
                else:
                    gt_states.append("turned off")
            
        if len(gt_states) > 0:
            return " and ".join(gt_states)
    
    return None

class Node(object):
    def __init__(self, name, object_id=None, pos3d=None, corner_pts=None, bbox2d=None, pcd=None, depth=None, global_node=False):
        self.name = name
        self.object_id = object_id # ai2thor object_id
        self.bbox2d = bbox2d # 2d bounding box (4x1)
        self.pos3d = pos3d # object position
        self.corner_pts = corner_pts # corner points of 3d bbox (8x3)
        self.pcd = pcd # point cloud (px3)
        self.depth = depth
        self.name_w_state = None
        self.global_node = global_node

    def set_state(self, state):
        self.name_w_state = state

    def __str__(self):
        return self.get_name()

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return True if self.get_name() == other.get_name() else False

    def get_name(self):
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return self.name


class Edge(object):
    def __init__(self, start_node, end_node, edge_type="none"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type
    
    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return str(self.start) + "->" + self.edge_type + "->" + str(self.end)


class SceneGraph(object):
    """
    Create a spatial scene graph
    """
    def __init__(self, event, task):
        self.event = event
        self.nodes = []
        self.total_nodes = []
        self.edges = {}
        self.task = task

        if event is not None:
            x = event.metadata['agent']['position']['x']
            y = event.metadata['agent']['position']['y']
            z = event.metadata['agent']['position']['z']
            self.camera_world_xyz = torch.as_tensor([x, y, z])
            self.rotation = event.metadata['agent']['rotation']['y']
            self.horizon = event.metadata['agent']['cameraHorizon']

    def add_node_wo_edge(self, node):
        self.total_nodes.append(node)

    def add_node(self, new_node):
        merge = False
        for idx, node in enumerate(self.nodes):
            if new_node.name == node.name:
                merge, dist = is_merge(np.array(new_node.pcd), np.array(node.pcd))
                if merge:
                    node.pcd = torch.cat((node.pcd, new_node.pcd), 0)
                    self.nodes[idx] = node
                    self.add_object_state(new_node, self.event, mode="gt")
                    return new_node
        if not merge:
            if new_node.object_id != self.in_robot_gripper():
                for node in self.total_nodes:
                    if node.name != new_node.name:
                        self.add_edge(node, new_node)
                        self.add_edge(new_node, node)
            self.nodes.append(new_node)
            self.add_object_state(new_node, self.event, mode="gt")
        return new_node

    def add_edge(self, node, new_node):
        pos_A = world_space_xyz_to_camera_space_xyz(torch.tensor(np.array([node.pos3d])).reshape(3, 1), 
            self.camera_world_xyz, self.rotation, self.horizon).flatten()
        pos_B = world_space_xyz_to_camera_space_xyz(torch.tensor(np.array([new_node.pos3d])).reshape(3, 1), 
            self.camera_world_xyz, self.rotation, self.horizon).flatten()
        cam_arr = pos_B - pos_A
        norm_vector = cam_arr / np.linalg.norm(cam_arr)

        box_A, box_B = np.array(node.corner_pts), np.array(new_node.corner_pts)
        if len(node.pcd) == 0 or len(new_node.pcd) == 0:
            return
        else:
            dist = get_pcd_dist(node.pcd, new_node.pcd)
        
        box_A_pts, box_B_pts = np.array(node.pcd), np.array(new_node.pcd)

        if new_node.object_id == self.in_robot_gripper(): # skip spatial relations if object is in robot gripper
            return

        # IN CONTACT
        if dist < IN_CONTACT_DISTANCE:
            if new_node.name not in BULKY_OBJECTS:
                if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):
                    if "countertop" in node.name or "stove burner" in node.name: # address the "inside countertop" issue
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
                    else:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
                elif len(np.where((box_B_pts[:, 0] < box_A[4, 0]) & (box_B_pts[:, 0] > box_A[0, 0]) & 
                        (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:
                    if len(np.where(box_B_pts[:, 1] > box_A[4, 1])[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:
                        # thor specific - fruits unrealistically stay upright in the bowl leading to "on top of bowl" relation instead of "inside bowl" relation
                        if 'slice' in new_node.name and node.name == 'bowl':
                            self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
                        else:
                            self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
                    elif len(np.where(box_A_pts[:, 1] > box_B[4, 1])[0]) > len(box_A_pts) * ON_TOP_OF_THRESH:
                        if node.name not in BULKY_OBJECTS:
                            # thor specific - fruits unrealistically stay upright in the bowl leading to "on top of bowl" relation instead of "inside bowl" relation 
                            if 'slice' in node.name and new_node.name == 'bowl':
                                self.edges[(node.name, new_node.name)] = Edge(node, new_node, "inside")
                            else:
                                self.edges[(node.name, new_node.name)] = Edge(node, new_node, "on top of")

        # CLOSE TO
        if dist < CLOSE_DISTANCE and (new_node.name, node.name) not in self.edges and (not new_node.global_node):
            if node.name not in BULKY_OBJECTS and new_node.name not in BULKY_OBJECTS:
                if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:
                    if norm_vector[1] > 0:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "above")
                    else:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "below")
                elif abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:
                    if norm_vector[0] > 0:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the right of")
                    else:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the left of")
                elif abs(norm_vector[2]) > NORM_THRESH_FRONT_BACK and new_node.bbox2d is not None and node.bbox2d is not None and new_node.depth is not None and node.depth is not None:
                    iou, inters = get_iou(new_node.bbox2d, node.bbox2d)
                    occlude_ratio = inters / ((node.bbox2d[2]-node.bbox2d[0]) * (node.bbox2d[3]-node.bbox2d[1]))
                    # print("new_node, node: ", new_node.name, node.name)
                    # print("iou, occlude_ratio: ", iou, occlude_ratio)
                    if occlude_ratio > OCCLUDE_RATIO_THRESH and len(np.where(new_node.depth <= np.min(node.depth))[0]) > len(new_node.depth) * DEPTH_THRESH:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "blocking")
    
    def add_object_state(self, node, event, mode="gt"):
        if mode == "gt":
            state = get_gt_object_state(node.object_id, event)
        elif mode == "clip":
            state = get_object_state(node.object_id, event.frame)
        if state is not None:
            node.set_state(f"{node.name} ({state})")
        return node

    def add_agent(self):
        object_id = self.in_robot_gripper()
        if object_id:
            name = get_label_from_object_id(object_id, [self.event], self.task)
            self.edges[(name, "robot gripper")] = Edge(Node(name), Node("robot gripper"), "inside")
        else:
            self.edges[("nothing", "robot gripper")] = Edge(Node("nothing"), Node("robot gripper"), "inside")
        return object_id

    def in_robot_gripper(self):
        for obj in self.event.metadata["objects"]:
            if obj["isPickedUp"]:
                return obj['objectId']
        return None

    def __eq__(self, other):
        if (set(self.nodes) == set(other.nodes)) and (set(self.edges.values()) == set(other.edges.values())):
            return True
        else:
            return False

    def __str__(self):
        visited = []
        res = "[Nodes]:\n"
        for node in set(self.nodes):
            res += node.get_name()
            res += "\n"
        res += "\n"
        res += "[Edges]:\n"
        for edge_key, edge in self.edges.items():
            name_1, name_2 = edge_key
            edge_key_reversed = (name_2, name_1)
            if (edge_key not in visited and edge_key_reversed not in visited) or edge.edge_type in ['on top of', 'inside', 'occluding']:
                res += str(edge)
                res += "\n"
            visited.append(edge_key)
        return res
