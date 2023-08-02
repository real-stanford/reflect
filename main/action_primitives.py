import time
import math
import numpy as np
from task_utils import *
from constants import *

def navigate_to_obj(taskUtil, obj_type, to_drop=False, failure_injection_idx=0, obj_id=None, replan=False, fail_execution=False):
    print("[INFO] Execute action: Navigate to", obj_type)
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]
    drop_failure_injected = False

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        start_frame = taskUtil.counter + 1
        taskUtil.nav_actions[(start_frame, start_frame)] = 'Move to ' + obj_type_in_sim.lower()
        return False

    if obj_id is not None:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectId"] == obj_id)
    elif '-' in obj_type:
        for key, val in taskUtil.unity_name_map.items():
            if val == obj_type:
                obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == key)
    else:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)

    # BFS search for poth
    closest_pos = closest_position(obj["position"], taskUtil.reachable_positions)
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    target_pos_val = [closest_pos['x'], closest_pos['z']]
    # print("robot_pos, target_pos, closest_to_target_pos: ", robot_pos, target_pos_val, closest_pos)
    taskUtil.grid_size = taskUtil.grid.shape[0]
    # calculate taskUtil.grid_index from taskUtil.grid_value
    for row in range(taskUtil.grid.shape[0]):
        for col in range(taskUtil.grid.shape[1]):
            if [round(robot_pos['x'], 2), round(robot_pos['z'], 2)] == [taskUtil.grid[row, col, 0], taskUtil.grid[row, col, 1]]:
                robot_x = row
                robot_y = col
            if [round(target_pos_val[0], 2), round(target_pos_val[1], 2)] == [taskUtil.grid[row, col, 0], taskUtil.grid[row, col, 1]]:
                target_x = row
                target_y = col
    target_pos = [target_x, target_y]
    # print("*** start, goal: ", robot_x, robot_y, target_pos)
    path = findPath(taskUtil.grid, x=robot_x, y=robot_y, target_pos=target_pos, reachable_points=taskUtil.reachable_points)
    # print("path: ", path)

    if path is None:
        print("[ERROR] No valid path is found from robot to target object")
        save_data(taskUtil, taskUtil.controller.last_event, replan=replan)
        taskUtil.controller.step(action="Done")
        return

    start_frame = taskUtil.counter + 1
    for p in path:
        x = taskUtil.grid[p.x,p.y][0]
        z = taskUtil.grid[p.x,p.y][1]
        y = 0.9
        # print("p: ", p, x, z)
        e = taskUtil.controller.step(
            action="Teleport",
            position=dict(x=x, y=y, z=z),
            forceAction=True,
            # horizon=30,
            standing=True
        )
        # print("e: ", e)
        save_data(taskUtil, e, replan=replan)
        taskUtil.controller.step(action="Done")
        # dropping injection
        obj_in_hand = False
        for o in taskUtil.controller.last_event.metadata['objects']:
            if o['isPickedUp'] == True:
                obj_in_hand = True
                break
        add_failure = np.random.uniform()
        if not taskUtil.failure_added and taskUtil.chosen_failure == 'drop' and obj_in_hand and add_failure > 0.5 and to_drop:
            # drop action primitive
            print("injected drop at step", taskUtil.counter)
            drop(taskUtil, failure_injection_idx)
            drop_failure_injected = True
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=obj["position"], robot_pos=robot_pos, replan=replan)
    end_frame = taskUtil.counter
    taskUtil.nav_actions[(start_frame, end_frame)] = 'Move to ' + obj_type_in_sim.lower()
    taskUtil.controller.step(action="Done")
    if to_drop:
        return drop_failure_injected
    else:
        return True


def pick_up(taskUtil, obj_type, fail_execution=False, replan=False):
    print("[INFO] Execute action: Picking up", obj_type)
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    # if the Sliced/Cracked object does not exist, then pick up the original object
    # if the original object does not exist, then pick up the Sliced/Cracked object
    obj_types = sorted([obj["objectType"] for obj in taskUtil.controller.last_event.metadata["objects"]])
    if obj_type in OBJ_UNSLICED_MAP:
        obj_unsliced_type = OBJ_UNSLICED_MAP[obj_type]
        if obj_unsliced_type in obj_types and obj_type not in obj_types:
            obj_type = obj_unsliced_type
    elif obj_type in OBJ_SLICED_MAP:
        obj_sliced_type = OBJ_SLICED_MAP[obj_type]
        if obj_sliced_type in obj_types:
            obj_type = obj_sliced_type

    e = taskUtil.controller.last_event
    objs = [obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type]
    
    # thor specific - to avoid picking up the largest slice (e.g. Lettuce_6_Slice_8 is smallest and Lettuce_6_Slice_1 is the largest)
    if 'LettuceSliced' in obj_type or 'AppleSliced' in obj_type:
        objs = sorted(objs, key=lambda x: int(x['name'].split('_')[-1])*-1)
    if 'PotatoSliced' in obj_type:
        objs = objs[2:]
    if 'TomatoSliced' in obj_type:
        objs = objs[4:]
    if "EggCracked" in obj_type:
        objs = ""
    
    if fail_execution == True or len(objs) == 0:
        if len(objs) == 0:
            print("Cannot find the target object to pick up")
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Pick up " + obj_type_in_sim.lower()
        return

    # if navigation is required
    if not objs[0]['visible'] and objs[0]['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, objs[0]['objectType'], replan=replan)

    if (taskUtil.chosen_failure == 'blocking' and taskUtil.failure_injection_params['src_obj_type'] == obj_type) \
        and obj_is_blocked(taskUtil, obj_type) or (taskUtil.chosen_failure == "drop" and taskUtil.failure_added is True and replan is False):
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Pick up " + obj_type_in_sim.lower()
        return
       
    for obj in objs:
        obj_id = obj['objectId']
        obj_pos = obj['position']
        # look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj_pos, robot_pos=robot_pos, replan=replan)
        e = taskUtil.controller.step(
            action="PickupObject",
            objectId=obj_id,
            forceAction=False,
            manualInteract=False
        )
        #print("PickUpObject: ", e)
        if replan:
            reasoning_info = taskUtil.get_reasoning_json()
            if 'drop' in reasoning_info['pred_failure_reason'] and reasoning_info['gt_failure_step'] in reasoning_info['pred_failure_step']:
                e = taskUtil.controller.step(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=True,
                    manualInteract=False
                )
                #print("PickUpDroppedObject: ", e)
            if e.metadata['lastActionSuccess']:
                break
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = "Pick up " + obj_type_in_sim.lower()
    taskUtil.controller.step(action="Done")
    time.sleep(1)


def put_in(taskUtil, src_obj_type, target_obj_type, fail_execution=False, replan=False):
    print(f"[INFO] Execute action: Putting {src_obj_type} in {target_obj_type}")
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim= NAME_MAP[src_obj_type]
    target_obj_type_in_sim = target_obj_type
    if target_obj_type in NAME_MAP:
        target_obj_type_in_sim = NAME_MAP[target_obj_type]
    
    if taskUtil.chosen_failure == "wrong_perception":
        if src_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            src_obj_type = taskUtil.failure_injection_params['wrong_obj_type']
        elif target_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            target_obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    src_obj = None
    for obj in taskUtil.controller.last_event.metadata["objects"]:
        if obj['isPickedUp']:
            src_obj = obj
            break
    if src_obj is None:
        print("The robot is not holding anything")
    elif src_obj['objectType'] != src_obj_type:
        print(f"The robot is not holding {src_obj_type}")
    else:
        print("The robot is holding:", src_obj['objectId'], src_obj['objectType'])
    
    if fail_execution or src_obj is None:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " inside " + target_obj_type_in_sim.lower()
        return

    # thor-specific, put in sink sometimes does not work as expected
    if target_obj_type == 'Sink':
        target_obj_type = 'SinkBasin'
  
    #if there are multiple instances
    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == target_obj_type:
            target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
            found_obj = True
            break
    if not found_obj:
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == target_obj_type)
    target_obj_id = target_obj['objectId']
    target_obj_pos = target_obj['position']

    # if navigation is required
    if not target_obj['visible'] and target_obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, target_obj['objectType'], obj_id=target_obj['objectId'], replan=replan)

    # look at object
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=target_obj_pos, robot_pos=robot_pos, replan=replan)

    # can only put one object in microwave
    if target_obj_type == 'Microwave' and len(target_obj['receptacleObjectIds']) > 0:
        print("Microwave already contains an object: ", target_obj['receptacleObjectIds'])
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " inside " + target_obj_type_in_sim.lower()
        return
    
    if target_obj_type == 'Toaster' and target_obj['isToggled']:
        place_obj_in_small_receptacle(taskUtil, target_obj_pos, replan=replan)
    else:
        if src_obj:
            e = taskUtil.controller.step(
                action="PutObject",
                objectId=target_obj_id,
                forceAction=False,
                placeStationary=True
            )
            if e.metadata['lastActionSuccess']:
                save_data(taskUtil, e, replan=replan)
                taskUtil.controller.step(action="Done")
                time.sleep(1)
            else:
                print("thor put_obj did not work, try place obj in small recetacle primitive")
                if target_obj_type in ["CoffeeMachine", "Microwave"]:
                    save_data(taskUtil, e, replan=replan)
                else:
                    place_obj_in_small_receptacle(taskUtil, target_obj_pos, replan=replan)
 
    taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " inside " + target_obj_type_in_sim.lower()


def put_on(taskUtil, src_obj_type, target_obj_type, fail_execution=False, target_obj_id=None, replan=False):
    print(f"[INFO] Execute action: Putting {src_obj_type} on {target_obj_type}")
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim = NAME_MAP[src_obj_type]
    if taskUtil.chosen_failure == 'ambiguous_plan' and target_obj_type.split("-")[0] == taskUtil.failure_injection_params['ambi_obj_type']:
        target_obj_type_in_sim = target_obj_type.split('-')[0]
        if target_obj_type_in_sim in NAME_MAP:
            target_obj_type_in_sim = NAME_MAP[target_obj_type_in_sim]
    else:
        target_obj_type_in_sim = target_obj_type
        if target_obj_type in NAME_MAP:
            target_obj_type_in_sim = NAME_MAP[target_obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if src_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            src_obj_type = taskUtil.failure_injection_params['wrong_obj_type']
        elif target_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            target_obj_type = taskUtil.failure_injection_params['wrong_obj_type']
    
    src_obj = None
    for obj in taskUtil.controller.last_event.metadata["objects"]:
        if obj['isPickedUp']:
            src_obj = obj
            break
    if src_obj is None:
        print("The robot is not holding anything")
    elif src_obj['objectType'] != src_obj_type:
        print(f"The robot is not holding {src_obj_type}")
    else:
        print("The robot is holding: ", src_obj['objectId'], src_obj['objectType'])

    e = taskUtil.controller.last_event
    if fail_execution or src_obj is None or src_obj['objectType'] != src_obj_type:
        save_data(taskUtil, e, replan=replan)
        if target_obj_type_in_sim.lower() == "sink":
            taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " inside " + target_obj_type_in_sim.lower()
        else:
            taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " on " + target_obj_type_in_sim.lower()
        return

    if target_obj_id is None:
        if "-" in target_obj_type and target_obj_type.split("-")[0] in ['StoveBurner', 'CounterTop']:
            for key, val in taskUtil.unity_name_map.items():
                if val == target_obj_type:
                    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == key)
                    break
        else:
            target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == target_obj_type)
    # if the exact object instance is specified
    else:
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectId"] == target_obj_id)

    target_obj_id = target_obj['objectId']
    target_obj_pos = target_obj['position']
    if target_obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, target_obj['objectType'], replan=replan)
    
    # look at object
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=target_obj_pos, robot_pos=robot_pos, replan=replan)
    e = taskUtil.controller.step(
        action="PutObject",
        objectId=target_obj_id,
        forceAction=False,
        placeStationary=True
    )
    
    # if not successful, try standing
    if not e.metadata['lastActionSuccess']:
        taskUtil.controller.step(action="Stand")
        e = taskUtil.controller.step(
            action="PutObject",
            objectId=target_obj_id,
            forceAction=False,
            placeStationary=True
        )
    taskUtil.controller.step(action="Done")

    # print("PutObject: ", e)
    save_data(taskUtil, e, replan=replan)

    # if still not successful, try pre-defined primitives
    if not e.metadata['lastActionSuccess']:
        print("thor put_obj did not work, applying self-defined primitives")
        if taskUtil.folder_name.split("/")[1] == 'heatPotato-3':
            taskUtil.controller.step(
                action="MoveHeldObjectAhead",
                moveMagnitude=0.4,
                forceVisible=False
            )
            e = taskUtil.controller.step(
                action="DropHandObject",
                forceAction=False
            )
        if target_obj_type.split("-")[0] == 'CounterTop':
            place_obj_on_large_receptacle(taskUtil, src_obj, target_obj_type, target_obj_id=target_obj_id, replan=replan)
    else:
        new_src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectId"] == src_obj['objectId'])
        look_at(taskUtil, target_pos=new_src_obj['position'], robot_pos=robot_pos, replan=replan)
        
    if target_obj_type_in_sim.lower() == "sink":
        taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " inside " + target_obj_type_in_sim.lower()
    else:
        taskUtil.interact_actions[taskUtil.counter] = "Put " + src_obj_type_in_sim.lower() + " on " + target_obj_type_in_sim.lower()
    time.sleep(1)
    

def toggle_on(taskUtil, obj_type, fail_execution=False, replan=False):
    print("[INFO] Execute action: Toggling on", obj_type)
    e = taskUtil.controller.last_event
    if taskUtil.chosen_failure == 'ambiguous_plan' and obj_type.split("-")[0] == taskUtil.failure_injection_params['ambi_obj_type']:
        obj_type_in_sim = obj_type.split('-')[0]
        if obj_type_in_sim in NAME_MAP:
            obj_type_in_sim = NAME_MAP[obj_type_in_sim]
    else:
        obj_type_in_sim = obj_type
        if obj_type in NAME_MAP:
            obj_type_in_sim = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']
    
    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Toggle on " + obj_type_in_sim.lower()
        return

    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == obj_type:
            obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
            found_obj = True
            break
    if not found_obj:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    obj_id = obj['objectId']

    # if navigation is required
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], obj_id=obj['objectId'], replan=replan)
    else:
        # look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj['position'], robot_pos=robot_pos, replan=replan)
    
    # retrieve the stove knob corresponding to the chosen stove burner
    if "StoveBurner" in obj_type:
        for o in taskUtil.controller.last_event.metadata["objects"]:
            if 'StoveKnob' in o['objectType'] and o['controlledObjects'] is not None \
                    and o['controlledObjects'][0] == obj_id:
                obj_id = o['objectId']

    execute_action = True
    if obj_type == 'Television':
        remote_control_obj = [obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == 'RemoteControl']
        if len(remote_control_obj) > 0 and not remote_control_obj[0]['isPickedUp']:
            execute_action = False
   
    if execute_action:
        e = taskUtil.controller.step(
            action="ToggleObjectOn",
            objectId=obj_id,
            forceAction=True
        )
        # print("ToggleObjectOn: ", e)
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = "Toggle on " + obj_type_in_sim.lower()
    taskUtil.controller.step(action="Done")
    # add sound to dict if action success
    if execute_action and e.metadata["lastActionSuccess"] and ("Toggle on " + obj_type.split('-')[0]) in SOUND_PATH:
        taskUtil.sounds[taskUtil.counter-1] = SOUND_PATH["Toggle on " + obj_type.split('-')[0]]
    time.sleep(1)

    # Post-processing for cleaning and filling up with water (need to do this as thor put_in primitive does not put the object directly below the faucet)
    faucet_objs = [o for o in taskUtil.controller.last_event.metadata["objects"] if o["objectType"] == "Faucet"]
    src_obj = next(o for o in taskUtil.controller.last_event.metadata["objects"] if o["objectId"] == obj['objectId'])
    if src_obj['isToggled']:
        # if single faucet:
        if len(faucet_objs) == 1:
            for obj in taskUtil.controller.last_event.metadata["objects"]:
                parentReceptacles = obj['parentReceptacles']
                if parentReceptacles is not None:
                    for parent in parentReceptacles:
                        # change state for object in sink
                        if "Sink" in parent:
                            e = taskUtil.controller.step(
                                action="CleanObject",
                                objectId=obj['objectId'],
                                forceAction=True
                            )
                            e = taskUtil.controller.step(
                                action="FillObjectWithLiquid",
                                objectId=obj['objectId'],
                                fillLiquid='water',
                                forceAction=True
                            )
        # if multiple faucets:
        else:
            src_obj_parent_receptacle = [p for p in src_obj['parentReceptacles'] if "SinkBasin" in p] 
            for obj in taskUtil.controller.last_event.metadata["objects"]:
                parentReceptacles = obj['parentReceptacles']
                if parentReceptacles is not None:
                    for parent in parentReceptacles:
                        if parent in src_obj_parent_receptacle:
                            e = taskUtil.controller.step(
                                action="CleanObject",
                                objectId=obj['objectId'],
                                forceAction=True
                            )
                            e = taskUtil.controller.step(
                                action="FillObjectWithLiquid",
                                objectId=obj['objectId'],
                                fillLiquid='water',
                                forceAction=True
                            )


def toggle_off(taskUtil, obj_type, fail_execution=False, replan=False):
    print(f"[INFO] Execute action: Toggling off", obj_type)
    e = taskUtil.controller.last_event
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Toggle off " + obj_type_in_sim.lower()
        return
    
    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == obj_type:
            obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
            found_obj = True
            break
    if not found_obj:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    obj_id = obj['objectId']
    
    # if navigation is required
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], obj_id=obj['objectId'], replan=replan)
    else:
        # look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj['position'], robot_pos=robot_pos, replan=replan)
    
    # retrieve the stove knob corresponding to the chosen stove burner
    if "StoveBurner" in obj_type:
        for o in taskUtil.controller.last_event.metadata["objects"]:
            if 'StoveKnob' in o['objectType'] and o['controlledObjects'] is not None \
                    and o['controlledObjects'][0] == obj_id:
                obj_id = o['objectId']

    execute_action = True
    # should hold remote control to open TV
    if obj_type == 'Television':
        remote_control_obj = [obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == 'RemoteControl']
        if len(remote_control_obj) > 0 and not remote_control_obj[0]['isPickedUp']:
            execute_action = False
    
    if execute_action:
        e = taskUtil.controller.step(
            action="ToggleObjectOff",
            objectId=obj_id,
            forceAction=False
        )
        # print("ToggleObjectOff: ", e)
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = "Toggle off " + obj_type_in_sim.lower()
    taskUtil.controller.step(action="Done")
    # add sound to dict
    if execute_action and e.metadata["lastActionSuccess"] and ("Toggle off " + obj_type.split('-')[0]) in SOUND_PATH:
        taskUtil.sounds[taskUtil.counter-1] = SOUND_PATH["Toggle off " + obj_type.split('-')[0]]
    time.sleep(1)


def open_obj(taskUtil, obj_type, fail_execution=False, replan=False):
    print("[INFO] Execute action: Opening", obj_type)
    src_obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']
    
    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Open " + obj_type_in_sim.lower()
        return

    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == obj_type:
            obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
            found_obj = True
            break
    if not found_obj:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    obj_id = obj['objectId']
    
    # if navigation is required
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], obj_id=obj['objectId'], replan=replan)
    else:
        # look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj['position'], robot_pos=robot_pos, replan=replan)

    e = taskUtil.controller.step(
        action="OpenObject",
        objectId=obj_id,
        forceAction=False
    )
    # print("Open object: ", e)
        
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = "Open " + src_obj_type_in_sim.lower()
    taskUtil.controller.step(action="Done")
    # if e.metadata["lastActionSuccess"] and ("Open " + obj_type.split('-')[0]) in SOUND_PATH:
    #     taskUtil.sounds[taskUtil.counter-1] = SOUND_PATH["Open " + obj_type.split('-')[0]]
    time.sleep(1)


def close_obj(taskUtil, obj_type, fail_execution=False, replan=False):
    print("[INFO] Execute action: Closing", obj_type)
    obj_type_in_english = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_english = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Close " + obj_type_in_english.lower()
        return
    
    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == obj_type:
            obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
            found_obj = True
            break
    if not found_obj:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    obj_id = obj['objectId']
    
    # if navigation is required
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], obj_id=obj['objectId'], replan=replan)
    else:
        # look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj['position'], robot_pos=robot_pos, replan=replan)
    
    e = taskUtil.controller.step(
        action="CloseObject",
        objectId=obj_id,
        forceAction=False
    )
    # print("Close object: ", e)
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = "Close " + obj_type_in_english.lower()
    taskUtil.controller.step(action="Done")
    # if e.metadata["lastActionSuccess"] and ("Close " + obj_type.split('-')[0]) in SOUND_PATH:
    #     taskUtil.sounds[taskUtil.counter-1] = SOUND_PATH["Close " + obj_type.split('-')[0]]
    time.sleep(1)


def slice_obj(taskUtil, obj_type, fail_execution=False, replan=False):
    print("[INFO] Execute action: Slicing", obj_type)
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    e = taskUtil.controller.last_event
    if fail_execution:
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Slice " + obj_type_in_sim.lower()
        return
    
    knife_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == 'Knife')
    
    obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    obj_id = obj['objectId']
    obj_pos = obj['position']

    # if navigation is required
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], replan=replan)
    
    if knife_obj['isPickedUp']:
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj_pos, robot_pos=robot_pos, replan=replan)
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
        obj_id = obj['objectId']
        e = taskUtil.controller.step(
            action="SliceObject",
            objectId=obj_id,
            forceAction=False
        )
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Slice " + obj_type_in_sim.lower()
        taskUtil.controller.step(action="Done")
        if e.metadata["lastActionSuccess"] and ("Slice " + obj_type.split('-')[0]) in SOUND_PATH:
            taskUtil.sounds[taskUtil.counter-1] = SOUND_PATH["Slice " + obj_type.split('-')[0]]
    else:
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Slice " + obj_type_in_sim.lower()

# Primitive 10
def crack_obj(taskUtil, obj_type, fail_execution=False, replan=False):
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    # skip the action if failure is injected or the object is not picked up by the robot
    if fail_execution or not obj['isPickedUp']:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = "Crack " + obj_type_in_sim.lower()
        return

    obj_id = obj['objectId']
    e = taskUtil.controller.step(
        action="BreakObject",
        objectId=obj_id,
        forceAction=False
    )
    print("BreakObject: ", e)

    if e.metadata["lastActionSuccess"] and ("Crack " + obj_type.split('-')[0]) in SOUND_PATH:
        taskUtil.sounds[taskUtil.counter] = SOUND_PATH["Crack " + obj_type.split('-')[0]]

    # after cracking, the cracked object is dropped in thor. So, need to pick it up again
    obj_types = sorted([obj["objectType"] for obj in taskUtil.controller.last_event.metadata["objects"]])
    if obj_type in OBJ_SLICED_MAP:
        obj_slice_type = OBJ_SLICED_MAP[obj_type]
        if obj_slice_type in obj_types:
            obj_type = obj_slice_type
    obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    obj_id = obj['objectId']
    e = taskUtil.controller.step(
            action="PickupObject",
            objectId=obj_id,
            forceAction=True,
            manualInteract=False
        )
    print("PickUpObject (cracked): ", e)
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = "Crack " + obj_type_in_sim.lower()  
    taskUtil.controller.step(action="Done")
    time.sleep(1)


def pour(taskUtil, src_obj_type, target_obj_type, fail_execution=False, replan=False):
    print(f"[INFO] Execute action: Pouring liquid from {src_obj_type} to {target_obj_type}")
    liquid_type = None
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim = NAME_MAP[src_obj_type]
    target_obj_type_in_sim = target_obj_type
    if target_obj_type in NAME_MAP:
        target_obj_type_in_sim = NAME_MAP[target_obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if src_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            src_obj_type = taskUtil.failure_injection_params['wrong_obj_type']
        elif target_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            target_obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == target_obj_type)
    target_obj_id = target_obj['objectId']
    src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == src_obj_type)
    src_obj_id = src_obj['objectId']

    # if navigation is required
    if not target_obj['visible'] and target_obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, target_obj['objectType'], replan=replan)

    # if obj_in_hand is True and obj_in_hand has liquid
    obj_in_hand = None
    for obj in taskUtil.controller.last_event.metadata['objects']:
        if obj['isPickedUp'] == True:
            obj_in_hand = obj
            break
    
    if obj_in_hand is not None and obj_in_hand["isFilledWithLiquid"] and src_obj_id == obj_in_hand['objectId']:
        liquid_type = obj_in_hand['fillLiquid']

        if fail_execution:
            e = taskUtil.controller.last_event
            save_data(taskUtil, e, replan=replan)
            # if liquid_type is not None:
            #     taskUtil.interact_actions[taskUtil.counter] = f"Pour {liquid_type.lower()} from " + src_obj_type_in_sim.lower() + " into " + target_obj_type_in_sim.lower() 
            # else:
            #     taskUtil.interact_actions[taskUtil.counter] = f"Pour liquid from {src_obj_type_in_sim.lower()} into {target_obj_type_in_sim.lower()}"
            return
    
        e = taskUtil.controller.step(
            action="EmptyLiquidFromObject",
            objectId=src_obj_id,
            forceAction=False
        )
        e = taskUtil.controller.step(
            action="FillObjectWithLiquid",
            objectId=target_obj_id,
            fillLiquid=liquid_type.lower(),
            forceAction=False
        )
        # print("FillWithLiquid: ", e)
        save_data(taskUtil, e, replan=replan)
        # if liquid_type is not None:
        #     taskUtil.interact_actions[taskUtil.counter] = f"Pour {liquid_type.lower()} from " + src_obj_type_in_sim.lower() + " into " + target_obj_type_in_sim.lower() 
        # else:
        #     taskUtil.interact_actions[taskUtil.counter] = f"Pour liquid from {src_obj_type_in_sim.lower()} into {target_obj_type_in_sim.lower()}"
        taskUtil.controller.step(action="Done")

    if liquid_type is not None and e.metadata["lastActionSuccess"] and f"Pour {liquid_type.lower()} into {target_obj_type.split('-')[0]}" in SOUND_PATH:
        taskUtil.sounds[taskUtil.counter-1] = SOUND_PATH[f"Pour {liquid_type.lower()} into {target_obj_type.split('-')[0]}"]
    time.sleep(1)


def drop(taskUtil, failure_injection_idx):
    obj_in_hand = None
    for obj in taskUtil.controller.last_event.metadata['objects']:
        if obj['isPickedUp'] == True:
            obj_in_hand = obj
            break
    if obj_in_hand is not None:
        e = taskUtil.controller.step(
            action="DropHandObject",
            forceAction=True
        )
        if e.metadata["lastActionSuccess"] and ("Drop " + obj_in_hand['objectType'].split('-')[0]) in SOUND_PATH:
            taskUtil.sounds[taskUtil.counter] = SOUND_PATH["Drop " + obj_in_hand['objectType'].split('-')[0]]
        # print("DropHandObject: ", e)
        obj_id = obj_in_hand['objectId']
        if obj_in_hand['objectType'] == "Egg":
            e = taskUtil.controller.step(
                action="BreakObject",
                objectId=obj_id,
                forceAction=False
            )
        # if the object is a container and filled with liquid, empty it
        if obj_in_hand['isFilledWithLiquid']:
            e = taskUtil.controller.step(
                action="EmptyLiquidFromObject",
                objectId=obj_id,
                forceAction=True
            )
        taskUtil.gt_failure['gt_failure_reason'] = "Dropped " + obj_in_hand['objectType']
        taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter+1
        taskUtil.objs_w_unk_loc.append(obj_in_hand['objectType'])
        taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
        taskUtil.failure_added = True


def dirty_obj(taskUtil, obj_type):
    src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    e = taskUtil.controller.step(
        action="DirtyObject",
        objectId=src_obj["objectId"],
        forceAction=True
    )
    print("DirtyObject: ", e)
    taskUtil.controller.step(action="Done")


def fill_obj(taskUtil, obj_type, liquid_type):
    obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == obj_type)
    e = taskUtil.controller.step(
        action="FillObjectWithLiquid",
        objectId=obj['objectId'],
        fillLiquid=liquid_type,
        forceAction=True
    )
    print("FillWithLiquid: ", e)
    taskUtil.controller.step(action="Done")

def tilt_camera(task, final_tilt):
    if final_tilt > 0:
        e = task.controller.step(
            action="LookDown",
            degrees=final_tilt
        )
    elif final_tilt < 0:
        e = task.controller.step(
            action="LookUp",
            degrees=-final_tilt
        )
    # print("Tilt camera: ", e)

def look_at(task, robot_pos=None, target_pos=None, replan=False, center_to_camera_disp=0.6, rep=0):
    robot_y = robot_pos['y'] + center_to_camera_disp
    yaw = np.arctan2(target_pos['x']-robot_pos['x'], target_pos['z']-robot_pos['z'])
    yaw = math.degrees(yaw)

    tilt = -np.arctan2(target_pos['y']-robot_y, np.sqrt((target_pos['z']-robot_pos['z'])**2 + (target_pos['x']-robot_pos['x'])**2))
    tilt = np.round(np.math.degrees(tilt),1)
    org_tilt = task.controller.last_event.metadata['agent']['cameraHorizon']
    final_tilt = tilt - org_tilt
    if tilt > 60:
        final_tilt = 60
    if tilt < -30:
        final_tilt = -30
    final_tilt = np.round(final_tilt, 1)

    e = task.controller.step(action="Teleport", **robot_pos, rotation=dict(x=0, y=yaw, z=0), forceAction=True)
    save_data(task, e, replan=replan)
    task.controller.step(action="Done")
    # print("tilt degree: ", final_tilt)
    if final_tilt > 0:
        e = task.controller.step(
            action="LookDown",
            degrees=final_tilt
        )
    elif final_tilt < 0:
        e = task.controller.step(
            action="LookUp",
            degrees=-final_tilt
        )
    save_data(task, e, replan=replan)


def place_obj(taskUtil, failure_injection_params):
    if taskUtil.chosen_failure == "occupied_put":
        src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == failure_injection_params['src_obj_type'])
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == failure_injection_params['target_obj_type'])
        e = taskUtil.controller.step(
            action="PickupObject",
            objectId=src_obj['objectId'],
            forceAction=True,
            manualInteract=False
        )
        taskUtil.controller.step(action='Done')
        if failure_injection_params['target_obj_type'] == 'Microwave':
            taskUtil.controller.step(
                action="OpenObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
            e = taskUtil.controller.step(
                action="PutObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
            taskUtil.controller.step(
                action="CloseObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
            taskUtil.controller.step(action='Done')
        else:
            e = taskUtil.controller.step(
                action="PutObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
    elif taskUtil.chosen_failure == "occupied":
        target_obj_type = failure_injection_params['target_obj_type']
        if "-" in target_obj_type and target_obj_type.split("-")[0] in ['StoveBurner', 'CounterTop']:
            for key, val in taskUtil.unity_name_map.items():
                if val == target_obj_type:
                    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["name"] == key)
                    break
        else:
            target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == failure_injection_params['target_obj_type'])

        objectPoses = []
        place_location = target_obj['position'].copy()
        objs = taskUtil.controller.last_event.metadata["objects"]
        for obj in objs:
            obj_name = obj['name']
            obj_type = obj['objectType']
            pos = obj['position']
            rot = obj['rotation']
            if not obj['pickupable'] and not obj['moveable']:
                continue
            if obj_type == failure_injection_params['src_obj_type']:
                pos = place_location
                pos['x'] += failure_injection_params['disp_x']
                pos['z'] += failure_injection_params['disp_z']
                pos['y'] += failure_injection_params['disp_y']
            # print("object name: ", obj_name)
            temp_dict = {'objectName': obj_name, 'position': pos, 'rotation': rot}
            objectPoses.append(temp_dict)

        e = taskUtil.controller.step(
            action='SetObjectPoses',
            objectPoses=objectPoses,
            placeStationary = False
        )
        print("SetObjectPoses: ", e)
        taskUtil.controller.step(
            action="AdvancePhysicsStep",
            timeStep=0.01
        )
        taskUtil.controller.step(action='Done')
    else:
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] if obj["objectType"] == failure_injection_params['target_obj_type'])
        place_location = target_obj['position'].copy()
        objs = taskUtil.controller.last_event.metadata["objects"]
        objectPoses = []
        for obj in objs:
            obj_name = obj['name']
            obj_type = obj['objectType']
            pos = obj['position']
            rot = obj['rotation']
            if not obj['pickupable'] and not obj['moveable']:
                continue
            if obj_type == failure_injection_params['src_obj_type']:
                pos = place_location
                pos['x'] += failure_injection_params['disp_x']
                pos['z'] += failure_injection_params['disp_z']
                pos['y'] += failure_injection_params['disp_y']
            # print("object name: ", obj_name)
            temp_dict = {'objectName': obj_name, 'position': pos, 'rotation': rot}
            objectPoses.append(temp_dict)
        e = taskUtil.controller.step(
            action='SetObjectPoses',
            objectPoses=objectPoses,
            placeStationary = False
        )
        taskUtil.controller.step(action='Done')
        print("SetObjectPoses: ", e)


def place_obj_in_small_receptacle(task, place_location, replan=False):
    print("[INFO] Running primitive to place object in small receptacle")
    robot_pos = task.controller.last_event.metadata['agent']['position']
    tilt = task.controller.last_event.metadata['agent']['cameraHorizon']
    dist = np.sqrt((robot_pos['x'] - place_location['x'])**2 + (robot_pos['z'] - place_location['z'])**2)
    #print("tilt, dist: ", tilt, dist)
    tilt = np.round(tilt, 1)
    dist = np.round(dist, 1) - 0.4
    # Look straight (tilt = 0)
    if tilt > 0:
        e = task.controller.step(
            action="LookUp",
            degrees=tilt
        )
    else:
        e = task.controller.step(
            action="LookDown",
            degrees=tilt
        )
    #print("Look: ", e)
    task.controller.step(action="Done")

    # Move object over receptacle
    e = task.controller.step(
        action="MoveHeldObjectAhead",
        moveMagnitude=dist,
        forceVisible=False
    )
    task.controller.step(action='Done')
    #print("move object: ", e)
    
    # Drop object
    e = task.controller.step(
        action="DropHandObject",
        forceAction=False
    )
    task.controller.step(action='Done')
    #print("drop object: ", e)

    # Look at the receptacle again
    if tilt > 0:
        e = task.controller.step(
            action="LookDown",
            degrees=tilt
        )
    else:
        e = task.controller.step(
            action="LookUp",
            degrees=tilt
        )
    #print("Look: ", e)
    save_data(task, e, replan=replan)
    task.controller.step(action="Done")
    time.sleep(1)


# helper function to place an object on a large receptacle such as a countertop
def place_obj_on_large_receptacle(task, src_obj, target_obj_type, thresh=0.8, target_obj_id=None, replan=False):
    print("[INFO] Running primitive to place object on large receptacle")
    if target_obj_id is None:
        print("target object id is not specified.")
    else:
        print("target object id:", target_obj_id)
    target_obj_type_in_sim = target_obj_type
    if target_obj_type in NAME_MAP:
        target_obj_type_in_sim = NAME_MAP[target_obj_type]
    src_obj_type, src_obj_id = src_obj['objectType'], src_obj['objectId']
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim = NAME_MAP[src_obj_type]

    if task.chosen_failure == "wrong_perception":
        if src_obj_type == task.failure_injection_params['correct_obj_type']:
            src_obj_type = task.failure_injection_params['wrong_obj_type']
        elif target_obj_type == task.failure_injection_params['correct_obj_type']:
            target_obj_type = task.failure_injection_params['wrong_obj_type']
    
    target_objs = []
    if target_obj_id is None:
        if "-" not in target_obj_type:
            robot_pos = task.controller.last_event.metadata['agent']['position']
            for obj in task.controller.last_event.metadata["objects"]:
                if obj["objectType"] == target_obj_type:
                    temp_obj_pos = obj['position']
                    dist = np.sqrt((robot_pos['x'] - temp_obj_pos['x'])**2 + (robot_pos['z'] - temp_obj_pos['z'])**2)
                    tup = (dist, obj)
                    target_objs.append(tup)
            target_objs = sorted(target_objs, key=lambda d: d[0])
        else:
            for obj_unity_name, v in task.unity_name_map.items():
                if v == target_obj_type:
                    target_obj = next(obj for obj in task.controller.last_event.metadata["objects"] if obj["name"] == obj_unity_name)
                    target_objs.append((0, target_obj))
    else: # target_obj_id is specified
        target_obj = next(obj for obj in task.controller.last_event.metadata["objects"] if obj["objectId"] == target_obj_id)
        target_objs.append((0, target_obj))

    # check if target object is in view
    found_obj = False
    for dist, target_obj in target_objs:
        target_obj_id = target_obj['objectId']
        e = task.controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=target_obj_id,
            anywhere=False
        )
        #print("spawnPoints: ", e)
        if e.metadata['lastActionSuccess'] and len(e.metadata['actionReturn']) > 0:
            found_obj = True
            print("receptacle found in current view")
            break

    # navigate to the closest target object
    if not found_obj:
        for i in range(len(target_objs)):
            _, target_obj = target_objs[i]
            print("[INFO] Navigate to the closest target object:", target_obj['objectId'])
            navigate_to_obj(task, target_obj['objectType'], obj_id=target_obj['objectId'], replan=replan)
            e = task.controller.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=target_obj['objectId'],
                anywhere=True
            )
            # print("GetSpawnCoordinatesAboveReceptacle: ", e)
            if e.metadata['actionReturn'] is not None:
                break
        target_obj_type_in_sim = NAME_MAP[task.unity_name_map[target_obj['name']]]

    print("chosen counterTop:", target_obj_id)
    task.controller.step(action="Done")
    time.sleep(1)
    place_locations = e.metadata['actionReturn']
    # print("total potential place points: ", len(place_locations))
    
    # find valid locations on the receptacle to put object
    placed = False
    visible = False
    robot_pos = task.controller.last_event.metadata['agent']['position']

    counter = 0
    while (not placed or not visible):
        counter += 1
        # if too many trials, just drop the object
        if counter > 200:
            e = task.controller.step(
                action="DropHandObject",
                forceAction=False
            )
            print("DropHandObject: ", e)
            break
        visible = False
        placed = False
        place_location = np.random.choice(place_locations)
        # place point should be close enough to robot
        dist = np.sqrt((robot_pos['x'] - place_location['x'])**2 + (robot_pos['z'] - place_location['z'])**2)
        if dist > thresh:
            continue
        e = task.controller.step(
            action="PlaceObjectAtPoint",
            objectId=src_obj_id,
            position=place_location
        )
        # print("PlaceObjectAtPoint: ", e)
        task.controller.step(action="Done")
        if e.metadata['lastActionSuccess']:
            placed = True
        if src_obj['visible']:
            visible = True
    
    save_data(task, e, replan=replan)
    look_at(task, robot_pos, place_location, replan)
    task.interact_actions[task.counter] = "Put " + src_obj_type_in_sim.lower() + " on " + target_obj_type_in_sim.lower()