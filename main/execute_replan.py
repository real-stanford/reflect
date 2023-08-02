import os
import pickle
import json
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from utils import *
from task_utils import *
from constants import *
from action_primitives import *


def execute_correction_plan(task_idx, f_name, taskUtil):
    with open(f'state_summary/{f_name}/replan.json', 'r') as f:
        plan = json.load(f)["plan"]
        for instr in plan:
            lis = instr.split(',')
            lis = [item.strip("() ") for item in lis]
            action = lis[0]
            params = lis[1:]

            taskUtil.chosen_failure = "blocking" if taskUtil.chosen_failure == "blocking" else None
            print("action, params: ", action, params)
            func = globals()[action]
            func(taskUtil, *params, fail_execution=False, replan=True)

    is_success = check_task_success(task_idx, taskUtil.controller.last_event)
    print("Task success :-)" if is_success else "Task fail :-(")
    return is_success


def run_correction(data_path, f_name):
    with open(f'thor_tasks/{f_name}/task.json') as f:
        task = json.load(f)
    controller = Controller(
        agentMode="default",
        massThreshold=None,
        scene=task['scene'],
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=960,
        height=960,
        fieldOfView=60,
        platform=CloudRendering
    )

    events_path = 'thor_tasks/' + f_name + '/events/'
    l = os.listdir(events_path)
    lsorted = sorted(l, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    last_frame = int(lsorted[-1].split('_')[-1].split('.')[0])
    with open(events_path + lsorted[-1], 'rb') as f:
        final_event = pickle.load(f)
    objs = [obj for obj in final_event.metadata['objects']]
    final_agent = final_event.metadata['agent']
    # print("final_agent: ", final_agent)

    e = controller.step(
        action="Teleport",
        position=final_agent['position'],
        rotation=final_agent['rotation'],
        horizon=final_agent['cameraHorizon'],
        standing=final_agent['isStanding'],
        forceAction=True
    )
    # print("Setting agent pose: ", e)

    # ---------- Resetting state to final state of the failed execution ----------
    objectPoses = []
    dropped_obj_type = ""
    
    # restore dropped object
    if "Dropped" == task['gt_failure_reason'].split(" ")[0]:
        dropped_obj_type = task['gt_failure_reason'].split(" ")[1]
        dropped_step = convert_timestep_to_step(task['gt_failure_step'], video_fps=1)
        with open(f'thor_tasks/{f_name}/events/step_{dropped_step}.pickle', 'rb') as f:
            dropped_event = pickle.load(f)
            dropped_obj = next(o for o in dropped_event.metadata["objects"] if o["objectType"] == dropped_obj_type)
            temp_dict = {'objectName': dropped_obj['name'], 'position': dropped_obj['position'], 'rotation': dropped_obj['rotation']}
            objectPoses.append(temp_dict)

    # restore object states and object poses
    for obj in objs:
        if 'Sliced' in obj['objectType']:  # e.g. BreadSliced
            index = obj['objectType'].find('Sliced')
            org_obj_type = obj['objectType'][:index]  # e.g. Bread
            
            org_obj = next(o for o in controller.last_event.metadata["objects"] if o["objectType"] == org_obj_type)
            if not org_obj['isSliced']:
                e = controller.step(
                action="SliceObject",
                objectId=org_obj['objectId'],
                forceAction=True
            )

        elif 'Cracked' in obj['objectType']:
            index = obj['objectType'].find('Cracked')
            org_obj_type = obj['objectType'][:index]

            org_obj = next(o for o in controller.last_event.metadata["objects"] if o["objectType"] == org_obj_type)
            if not org_obj['isBroken']:
                e = controller.step(
                action="BreakObject",
                objectId=org_obj['objectId'],
                forceAction=True
            )

    for obj in objs:
        org_obj = next(o for o in controller.last_event.metadata["objects"] if o["name"] == obj['name'])
        if obj['isOpen']:
            e = controller.step(
                action="OpenObject",
                objectId=org_obj['objectId'],
                forceAction=True
            )
        else:
            e = controller.step(
                    action="CloseObject",
                    objectId=org_obj['objectId'],
                    forceAction=True
                ) 
        if obj['isToggled']:
            e = controller.step(
                action="ToggleObjectOn",
                objectId=org_obj['objectId'],
                forceAction=True
            )
        else:
            e = controller.step(
                action="ToggleObjectOff",
                objectId=org_obj['objectId'],
                forceAction=True
            )
        if obj['isFilledWithLiquid']:
            e = controller.step(
                action="FillObjectWithLiquid",
                objectId=org_obj['objectId'],
                fillLiquid=obj['fillLiquid'],
                forceAction=True
            )
        if obj['isDirty']:
            e = controller.step(
                action="DirtyObject",
                objectId=org_obj['objectId'],
                forceAction=True
            )
        if obj['objectType'] != dropped_obj_type:
            obj_name = obj['name']
            pos = obj['position']
            rot = obj['rotation']
            if not obj['pickupable'] and not obj['moveable']:
                continue
            temp_dict = {'objectName': obj_name, 'position': pos, 'rotation': rot}
            objectPoses.append(temp_dict)

    e = controller.step(
        action='SetObjectPoses',
        objectPoses=objectPoses
    )
    # print("SetObjectPoses: ", e)
    controller.step(action="Done")

    # restore 'in robot gripper' relation
    for obj in objs:
        if obj['isPickedUp']:
            org_obj = next(o for o in controller.last_event.metadata["objects"] if o["name"] == obj['name'])
            e = controller.step(
                action="PickupObject",
                objectId=org_obj['objectId'],
                forceAction=True
            )
    # ----------------------------------------------------------------------------

    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    if 'chosen_failure' in task:
        chosen_failure = task['chosen_failure']
    else:
        chosen_failure = None
    if 'failure_injection_params' in task:
        failure_injection_params = task['failure_injection_params']
    else:
        failure_injection_params = None
    taskUtil = TaskUtil(folder_name=f_name,
                    controller=controller,
                    reachable_positions=reachable_positions,
                    failure_injection=False,
                    index=0,
                    repo_path=data_path,
                    chosen_failure=chosen_failure,
                    failure_injection_params=failure_injection_params,
                    counter=last_frame,
                    replan=True)
    is_success = execute_correction_plan(task['task_idx'], f_name, taskUtil)

    with open(f'state_summary/{f_name}/replan.json', 'r') as f:
        replan_json = json.load(f)
    replan_json["success"] = is_success
    with open(f'state_summary/{f_name}/replan.json', 'w') as f:
        json.dump(replan_json, f)

    generate_video(taskUtil, recovery_video=True)
    controller.stop()
