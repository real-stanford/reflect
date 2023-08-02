import numpy as np
import open3d as o3d
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import constants as cons
import scipy.spatial

translation_lm_id =  'stsb-roberta-large' # 'all-distilroberta-v1'
# device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
translation_lm = SentenceTransformer(translation_lm_id).to(device)


def get_pcd_dist(pts_A, pts_B):
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pts_A)
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(pts_B)

    dists = pcd_A.compute_point_cloud_distance(pcd_B)
    dist = np.min(np.array(dists))
    return dist

def is_merge(pts_A, pts_B):
    if len(np.array(pts_A)) == 0 or len(np.array(pts_B)) == 0:
        return True
    else:
        dist = get_pcd_dist(pts_A, pts_B)
        if dist < 0.01:
            return True, dist
        else:
            return False, dist

def get_label_from_object_id(object_id, events, task):
    try:
        if len(events) == 1:
            src_obj = next(obj for obj in events[0].metadata["objects"] if obj["objectId"] == object_id)
        else:
            found = False
            for event in events:
                for obj in event.metadata["objects"]:
                    if obj["objectId"] == object_id:
                        src_obj = obj
                        found = True
                        break
                if found:
                    break

        if src_obj['name'] in task['unity_name_map']:
            label = cons.NAME_MAP[task['unity_name_map'][src_obj['name']]]
        elif src_obj['name'] == "Bread_2_Slice_1":
            label = cons.NAME_MAP['BreadSliced']
        elif src_obj['objectType'] in cons.NAME_MAP:
            label = cons.NAME_MAP[src_obj['objectType']]
        else:
            label = src_obj['objectType'].lower()
        return label
    except:
        return None

def is_moving(object_id, event):
    for obj in event.metadata["objects"]:
        if object_id == obj["objectId"]:
            return (obj["moveable"] and obj["isMoving"])
    return False

def is_picked_up(object_id, event):
    for obj in event.metadata["objects"]:
        if object_id == obj["objectId"]:
            return (obj["pickupable"] and obj["isPickedUp"])
    return False

def is_receptacle(object_id, event):
    for obj in event.metadata["objects"]:
        if object_id == obj["objectId"]:
            return obj["receptacle"]
    return False

def transform_point3s(t, ps):
    """Transfrom 3D points from one space to another.
    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).
    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]

def get_admissible_actions(object_list, last_event):
    for obj_name in cons.NAME_MAP:
        if obj_name.split("-")[0] in object_list and obj_name not in object_list:
            object_list.append(obj_name)
    available_actions = []
    # print("available objects:", object_list)
   
    pickable_classes = []
    for obj in last_event.metadata["objects"]:
        if obj['pickupable']:
            pickable_classes.append(obj['objectType'])

    recep_ids = cons.get_receptacle_ids()
    recep_classes = [cons.object_intid_to_string(recep_id) for recep_id in recep_ids]
    toggle_ids = cons.get_togglable_ids()
    toggle_classes = [cons.object_intid_to_string(toggle_id) for toggle_id in toggle_ids]
    openable_ids = cons.get_openable_ids()
    openable_classes = [cons.object_intid_to_string(openable_id) for openable_id in openable_ids]

    sliceable_classes = cons._SLICEABLES
    crackable_classes = cons._CRACKABLE
    flat_recep_classes = cons._FLAT_RECEPT
    fillable_classes = cons._FILLABLE
    
    for object_name in object_list:
        object_class = object_name.split("-")[0]
        if object_class in pickable_classes:
            available_actions.append(f"(pick_up, {object_name})")
        if object_class in recep_classes:
            for tmp in object_list:
                if tmp.split("-")[0] in pickable_classes and tmp != object_name:
                    if object_class in flat_recep_classes:
                        available_actions.append(f"(put_on, {tmp}, {object_name})")
                    else:
                        available_actions.append(f"(put_in, {tmp}, {object_name})")
        if object_class in toggle_classes:
            available_actions.append(f"(toggle_on, {object_name})")
            available_actions.append(f"(toggle_off, {object_name})")
        if object_class in openable_classes:
            available_actions.append(f"(open_obj, {object_name})")
            available_actions.append(f"(close_obj, {object_name})")
        if object_class in sliceable_classes:
            available_actions.append(f"(slice_obj, {object_name})")
        if object_class in crackable_classes:
            available_actions.append(f"(crack_obj, {object_name})")
        if object_class in fillable_classes:
            for tmp in object_list:
                if (object_class in fillable_classes) and (tmp.split("-")[0] in fillable_classes):
                    if object_class != tmp and "Sink" not in tmp:
                        available_actions.append(f"(pour, {tmp}, {object_class})")
    return available_actions

def get_initial_plan(actions):
    plan = ""
    idx = 0
    for action in actions:
        params = action[1:-1].split(", ")
        if params[0] != "navigate_to_obj":
        # if True:
            for i, obj_name in enumerate(params[1:]):
                if obj_name in cons.NAME_MAP.keys():
                    params[i+1] = cons.NAME_MAP[obj_name]
                else:
                    params[i+1] = obj_name.lower()
            plan += f"{idx+1}. ({', '.join(params)})\n"
            idx += 1
    return plan[:-1]

def get_robot_plan(folder_name, step=None, with_obs=False):
    with open('state_summary/{}/state_summary_L2.txt'.format(folder_name), 'r') as f:
        L2_captions = f.readlines()
    
    with open('state_summary/{}/state_summary_L1.txt'.format(folder_name), 'r') as f:
        L1_captions = f.readlines()

    if with_obs is False:
        captions = L2_captions
    else:
        captions = L1_captions

    robot_plan = ""
    for caption in captions:
        if step is not None and step in caption:
            break
        if with_obs:
            robot_plan += caption
        else:
            robot_plan += caption[:caption.find("Visual observation")-1] + "\n"
    return robot_plan

def get_replan_prefix():
    return f"""Provide a plan with the available actions for the robot to recover from the failure and finish the task.
Available actions: pick up, put in some container, put on some receptacle, open (e.g. fridge), close, toggle on (e.g. faucet), toggle off, slice object, crack object (e.g. egg), pour (liquid) from A to B.
The robot can only hold one object in its gripper, in other words, if there's object in the robot gripper, it can no longer pick up another object.
###
To clean a dirty object A, the plan is
1. (pick_up, A)
2. (put_in, A, sink)
3. (toggle_on, faucet)
4. (toggle_off, faucet)
5. (pour, A, sink)
6. (pick_up, A)
Object A should be clean after executing these actions.
###
The plan should 1) not contain any if statements 2) contain only the available actions 3) resemble the format of the initial plan.
"""

# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score

def get_cos_sim(str1, str2):
    embedding_1 = translation_lm.encode(str1, convert_to_tensor=True, device=device)
    embedding_2 = translation_lm.encode(str2, convert_to_tensor=True, device=device)
    cos_scores = st_utils.pytorch_cos_sim(embedding_1, embedding_2)[0].detach().cpu().numpy()
    return cos_scores[0]

def translate_plan(plan, object_list, last_event):
    translated_plan = ""
    available_actions = get_admissible_actions(object_list, last_event)
    action_list_embedding = translation_lm.encode(available_actions, batch_size=32, convert_to_tensor=True, device=device)
    idx = 0
    for step_instruct in plan.split("\n"):
        parsed_instruct = "".join(step_instruct.split(".")[1:]).strip()
        if ")" in parsed_instruct:
            parsed_instruct = parsed_instruct[:parsed_instruct.find(")")+1]
        most_similar_idx, matching_score = find_most_similar(parsed_instruct, action_list_embedding)
        translated_action = available_actions[most_similar_idx]
        print(step_instruct, "=>", translated_action, "score:", matching_score)
        if matching_score < 0.6:
            continue
        translated_plan += f"{translated_action}\n"
        idx += 1
    return translated_plan

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p)>=0

def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    hull_vertices = np.array([[0,0,0]])
    for v in hull.vertices:
        hull_vertices = np.vstack((hull_vertices, np.array([target_pts[v,0], target_pts[v,1], target_pts[v,2]])))
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False
    
def convert_step_to_timestep(step, video_fps):
    seconds = step // video_fps
    formatted_time = '{:02d}:{:02d}'.format(int(seconds / 60), seconds % 60)
    return formatted_time

def convert_timestep_to_step(timestep, video_fps):
    minutes, seconds = timestep.split(":")
    step = int(minutes) * 60 * video_fps + int(seconds) * video_fps
    return step

def check_task_success(task_idx, last_event):
    if task_idx == 1: # boil water
        for obj in last_event.metadata["objects"]:
            if "Pot" == obj["objectType"]:
                return obj["isFilledWithLiquid"] and 'water' == obj["fillLiquid"] and obj['temperature'] == 'Hot'
    if task_idx == 2: # toast bread
        for obj in last_event.metadata["objects"]:
            if "BreadSliced" == obj["objectType"]:
                return obj["isCooked"]
    if task_idx == 3: # cook egg
        egg_cracked_is_cooked = False
        for obj in last_event.metadata["objects"]:
            if "EggCracked" == obj["objectType"]:
                egg_cracked_is_cooked = obj["isCooked"]
            if "Pan" == obj["objectType"]:
                pan_is_clean = not obj["isDirty"]
        return egg_cracked_is_cooked and pan_is_clean
    if task_idx == 4: # heat potato
        for obj in last_event.metadata["objects"]:
            if "Potato" == obj["objectType"]:
                potato_is_cooked = obj["isCooked"]
            if "Plate" == obj["objectType"]:
                plate_is_clean = not obj["isDirty"]
        return potato_is_cooked and plate_is_clean
    if task_idx == 5: # make coffee
        for obj in last_event.metadata["objects"]:
            if "Mug" == obj["objectType"]:
                return obj["isFilledWithLiquid"] and 'coffee' == obj["fillLiquid"] and not obj["isDirty"]
    if task_idx == 6: # water plant
        for obj in last_event.metadata["objects"]:
            if "HousePlant" == obj["objectType"]:
                return obj["isFilledWithLiquid"] and 'water' == obj["fillLiquid"]
    if task_idx == 7: # store egg
        for obj in last_event.metadata["objects"]:
            if "Egg" == obj["objectType"]:
                succ_1 = False
                for p in obj['parentReceptacles']:
                    if "Bowl" in p:
                        succ_1 = True
            if "Bowl" == obj["objectType"]:
                succ_2 = False
                for p in obj['parentReceptacles']:
                    if "Fridge" in p:
                        succ_2 = True
        return succ_1 and succ_2
    if task_idx == 8: # make salad
        succ_1 = False
        lettuceSliced, potatoSliced, tomatoSliced = False, False, False
        for obj in last_event.metadata["objects"]:
            if "Bowl" == obj["objectType"]:
                if obj['parentReceptacles'] is not None and "Fridge" in obj['parentReceptacles'][0].split('|'):
                    succ_1 = True
                for fruit_obj_id in obj['receptacleObjectIds']:
                    # Need the second condition cause sometimes the object id in thor looks like - 
                    # 'Tomato|+01.30|+00.96|-01.08|TomatoSliced_4' instead of 'TomatoSliced|+01.30|+00.96|-01.08|'
                    if ('LettuceSliced' in fruit_obj_id.split('|')) or ('LettuceSliced' in fruit_obj_id.split('|')[-1].split('_')):
                        lettuceSliced = True
                    elif ('TomatoSliced' in fruit_obj_id.split('|')) or ('TomatoSliced' in fruit_obj_id.split('|')[-1].split('_')):
                        tomatoSliced = True
                    elif ('PotatoSliced' in fruit_obj_id.split('|')) or ('PotatoSliced' in fruit_obj_id.split('|')[-1].split('_')):
                        potatoSliced = True
                return succ_1 and lettuceSliced and potatoSliced and tomatoSliced
    if task_idx == 9: # switch devices
        for obj in last_event.metadata["objects"]:
            if "Laptop" == obj["objectType"]:
                succ_1 = "TVStand" in obj['parentReceptacles'][0] and not obj['isOpen']
            if "Television" == obj["objectType"]:
                succ_2 = obj['isToggled']
        return succ_1 and succ_2
    if task_idx == 10: # warm water
        for obj in last_event.metadata["objects"]:
            if "Mug" == obj["objectType"]:
                succ_1 = obj["isFilledWithLiquid"] and 'water' == obj["fillLiquid"] and obj['temperature'] == 'Hot'
            if "Cup" == obj["objectType"]:
                succ_2 = obj["isFilledWithLiquid"] and 'water' == obj["fillLiquid"] and obj['temperature'] == 'Hot'
        return succ_1 or succ_2
    return False
