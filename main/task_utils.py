import os
import PIL
import json
import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List
from collections import deque
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from constants import *
from point_cloud_utils import *

class TaskUtil:
    def __init__(self, folder_name,
                controller, 
                reachable_positions, 
                failure_injection, 
                index, 
                repo_path, 
                chosen_failure,
                failure_injection_params, 
                counter=0, 
                replan=False):
        self.counter = counter
        self.repo_path = repo_path
        self.folder_name = folder_name
        if failure_injection:
            self.specific_folder_name = self.get_folder_name(folder_name, index+1)
        else:
            self.specific_folder_name = folder_name
        self.controller = controller
        self.grid = self.createGraph()
        self.interact_actions = {}
        self.nav_actions = {}
        self.reachable_positions = reachable_positions
        self.reachable_points = self.get_2d_reachable_points()
        self.failure_added = False
        self.objs_w_unk_loc = []
        self.failures = ['drop', 'failed_action', 'missing_step']
        self.unity_name_map = self.get_unity_name_map()
        self.sounds = {}
        self.failure_injection_params = failure_injection_params
        if failure_injection and chosen_failure is None:
            i = index % len(self.failures)
            self.chosen_failure = self.failures[i]
            print("[INFO] chosen failure:", self.chosen_failure)
        else:
            self.chosen_failure = chosen_failure
        self.interact_action_primitives = ['put_on', 'put_in', 'pick_up', 'slice_obj', 'toggle_on', 'toggle_off', 'open_obj', 'close_obj', 'pour', 'crack_obj']
        self.gt_failure = {}
        if os.path.exists(f'{self.repo_path}/thor_tasks/{folder_name.split("/")[0]}/{folder_name.split("/")[1]}.pickle'):
            with open(f'{self.repo_path}/thor_tasks/{folder_name.split("/")[0]}/{folder_name.split("/")[1]}.pickle', 'rb') as handle:
                self.failures_already_injected = pickle.load(handle)

    def get_unity_name_map(self):
        obj_list = ['CounterTop', 'StoveBurner', 'Cabinet', 'Faucet', 'Sink']
        obj_rep_map = {}
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectType"] in obj_list:
                if obj["objectType"] in obj_rep_map:
                    obj_rep_map[obj["objectType"]] += 1
                else:
                    obj_rep_map[obj["objectType"]] = 1
        for key in obj_rep_map.keys():
            if obj_rep_map[key] == 1:
                obj_list.remove(key)
        unity_name_map = {}
        for obj_type in obj_list:
            counter = 0
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] == obj_type:
                    counter += 1
                    unity_name_map[obj['name']] = obj_type + '-' + str(counter)
        # print("unity_name_map: ", unity_name_map)
        return unity_name_map

    def get_folder_name(self, folder_name, folder_idx):
        final_folder_name = folder_name + '-' + str(folder_idx)
        return final_folder_name

    def createGraph(self, gridSize=0.25, min=-5, max=5.1):
        grid = np.mgrid[min:max:gridSize, min:max:gridSize].transpose(1,2,0)
        return grid

    def get_2d_reachable_points(self):
        reachable_points = []
        for p in self.reachable_positions:
            reachable_points.append([p['x'], p['z']])
        reachable_points = np.array(reachable_points)
        return reachable_points
    
    def get_reasoning_json(self):
        with open(f'state_summary/{self.specific_folder_name}/reasoning.json') as f:
            reasoning_json = json.load(f)
        return reasoning_json


def closest_position(
    object_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]]
) -> Dict[str, float]:
    out = reachable_positions[0]
    min_distance = float('inf')
    for pos in reachable_positions:
        # NOTE: y is the vertical direction, so only care about the x/z ground positions
        dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
        if dist < min_distance:
            min_distance = dist
            out = pos
    return out

# A queue node used in BFS
class Node:
    # (x, y) represents coordinates of a cell in the matrix
    # maintain a parent node for the printing path
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
 
    def __repr__(self):
        return str((self.x, self.y))
 
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
 
# Below lists detail all four possible movements from a cell
ROW = [-1, 0, 0, 1]
COL = [0, -1, 1, 0]
 
# The function returns false if (x, y) is not a valid position
def isValid(x, y, N, reachable_points, grid):
    # If cell lies out of bounds
    if (x < 0 or y < 0 or x >= N or y >= N):
        return False
    
    # Check if cell is in reachable cells
    val = grid[x][y]
    if val not in reachable_points.tolist():    
        return False
    
    return True
 
# Utility function to find path from source to destination
def getPath(node, path=[]):
    if node:
        getPath(node.parent, path)
        path.append(node)
 
# Find the shortest route in a matrix from source cell (x, y) to
# destination cell (N-1, N-1)
def findPath(matrix, x=0, y=0, target_pos=None, reachable_points=None):
    matrix = matrix.tolist()

    # `N × N` matrix
    N = len(matrix)
 
    # create a queue and enqueue the first node
    q = deque()
    src = Node(x, y)
    q.append(src)
 
    # set to check if the matrix cell is visited before or not
    visited = set()
 
    key = (src.x, src.y)
    visited.add(key)
 
    # loop till queue is empty
    while q:
 
        # dequeue front node and process it
        curr = q.popleft()
        i = curr.x
        j = curr.y
 
        # return if the destination is found
        if i == target_pos[0] and j == target_pos[1]:
            path = []
            getPath(curr, path)
            return path
 
        # check all four possible movements from the current cell
        # and recur for each valid movement
        for k in range(len(ROW)):
            # get next position coordinates using the value of the current cell
            x = i + ROW[k]
            y = j + COL[k]
 
            # check if it is possible to go to the next position
            # from the current position
            if isValid(x, y, N, reachable_points, matrix):
                # construct the next cell node
                next = Node(x, y, curr)
                key = (next.x, next.y)
 
                # if it isn't visited yet
                if key not in visited:
                    # enqueue it and mark it as visited
                    q.append(next)
                    visited.add(key)
 
    # return None if the path is not possible
    return


def obj_is_blocked(taskUtil, src_obj_type):
    target_obj_type = taskUtil.failure_injection_params['target_obj_type']
    src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == src_obj_type)
    src_obj_loc = src_obj['position']
    src_obj_loc = np.array([src_obj_loc['x'], src_obj_loc['y'], src_obj_loc['z']])
    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == target_obj_type)
    target_obj_loc = target_obj['position']
    target_obj_loc = np.array([target_obj_loc['x'], target_obj_loc['y'], target_obj_loc['z']])
    dist = np.linalg.norm(src_obj_loc - target_obj_loc)
    # print("dist: ", dist)
    if dist < 0.3:
        e = taskUtil.controller.last_event
        x = e.metadata['agent']['position']['x']
        y = e.metadata['agent']['position']['y']
        z = e.metadata['agent']['position']['z']
        camera_world_xyz = torch.as_tensor([x, y, z])
        rotation = e.metadata['agent']['rotation']['y']
        horizon = e.metadata['agent']['cameraHorizon']
        pos_A = world_space_xyz_to_camera_space_xyz(torch.tensor(np.expand_dims(target_obj_loc, 0)).reshape(3, 1),
                                                    camera_world_xyz, rotation, horizon).flatten()
        pos_B = world_space_xyz_to_camera_space_xyz(torch.tensor(np.expand_dims(src_obj_loc, 0)).reshape(3, 1),
                                                    camera_world_xyz, rotation, horizon).flatten()
        cam_arr = pos_B - pos_A
        norm_vector = cam_arr / np.linalg.norm(cam_arr)
        # print("norm_vector: ", norm_vector)
        if norm_vector[2] > 0.75:
            print(f"[INFO] {src_obj_type} is blocked.")
            return True
        else:
            print(f"[INFO] {src_obj_type} is not blocked.")
            return False

def save_data(task, e, replan=False):
    task.counter += 1
    if replan:
        folder = 'recovery'
    else:
        folder = 'thor_tasks'
    os.system(f"mkdir -p {task.repo_path}/{folder}/{task.specific_folder_name}/events")
    os.system(f"mkdir -p {task.repo_path}/{folder}/{task.specific_folder_name}/ego_img")
    
    with open(f'{task.repo_path}/{folder}/{task.specific_folder_name}/events/step_' + str(task.counter) + '.pickle', 'wb') as handle:
        pickle.dump(e, handle, protocol=pickle.HIGHEST_PROTOCOL)
    rgb = e.frame
    plt.imsave(f'{task.repo_path}/{folder}/{task.specific_folder_name}/ego_img/img_step_' + str(task.counter) + '.png', np.asarray(rgb, order='C'))


def generate_video(taskUtil, recovery_video):
    if recovery_video:
        folder = 'recovery'
    else:
        folder = 'thor_tasks'
    img_path = f"{taskUtil.repo_path}/{folder}/{taskUtil.specific_folder_name}/ego_img/"
    save_path = f"{taskUtil.repo_path}/{folder}/{taskUtil.specific_folder_name}/"
    SOUND_PATH = f"{taskUtil.repo_path}/assets/sounds/"
    l = os.listdir(img_path)
    lsorted = sorted(l, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    
    if len(lsorted) > 0 and not recovery_video:
        video_path = f'{save_path}original-video.mp4'
        with imageio.get_writer(video_path, mode='I', fps=1) as writer:
            for filename in lsorted:
                img = PIL.Image.open(img_path + filename).convert('RGB')
                writer.append_data(np.array(img))
        # Add sound
        sounds = taskUtil.sounds
        print("sounds: ", sounds)
        sound_lis = []
        for key, val in sounds.items():
            start_time = key
            audio = AudioFileClip(SOUND_PATH + val)
            sound_lis.append(audio.set_start(start_time))
        if len(sound_lis) > 0:
            clip = VideoFileClip(save_path + "original-video.mp4")
            audio = CompositeAudioClip(sound_lis)
            audio = audio.set_duration(clip.duration)
            clip.audio = audio
            os.system(f"rm {save_path}original-video.mp4")
            clip.write_videofile(save_path + 'original-video.mp4', audio_codec='aac')
    
    if len(lsorted) > 0 and recovery_video:
        recover_video_path = f'{save_path}recovery-video.mp4'
        with imageio.get_writer(recover_video_path, mode='I', fps=1) as writer:
            for filename in lsorted:
                img = PIL.Image.open(img_path + filename).convert('RGB')
                writer.append_data(np.array(img))
        
        sounds = taskUtil.sounds
        print("sounds: ", sounds)
        sound_lis = []
        for key, val in sounds.items():
            start_time = key
            audio = AudioFileClip(SOUND_PATH + val)
            sound_lis.append(audio.set_start(start_time))
        if len(sound_lis) > 0:
            clip = VideoFileClip(save_path + "recovery-video.mp4")
            audio = CompositeAudioClip(sound_lis)
            audio = audio.set_duration(clip.duration)
            clip.audio = audio
            os.system(f"rm {save_path}recovery-video.mp4")
            clip.write_videofile(save_path + 'recovery-video.mp4', audio_codec='aac')
