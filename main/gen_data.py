import os
import json
import numpy as np
import random
import pickle
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from action_primitives import *
from task_utils import *
from constants import *
from utils import *

def flatten_list(lis):
    output = []
    for item in lis:
        if isinstance(item, list):
            output.extend(item)
        else:
            output.append(item)
    return output

def get_failure_injection_idx(taskUtil, actions, task, action_idxs, nav_idxs, interact_cnt=0, nav_cnt=0):
    counter = 0
    print("[INFO] Injected failures:", taskUtil.failures_already_injected)
    try: 
        while True:
            if taskUtil.chosen_failure == 'missing_step':
                if "specified_missing_steps" in task:
                    cnt = 0
                    for f in taskUtil.failures_already_injected:
                        if f[0] == 'missing_step':
                            cnt += 1
                    if cnt < len(task['specified_missing_steps']):
                        failure_injection_idx = task['specified_missing_steps'][cnt] # can contain multiple indices
                        return failure_injection_idx
                
                failure_injection_idx = np.random.choice(action_idxs[interact_cnt:])
                if "toggle_off" in actions[failure_injection_idx] or "close_obj" in actions[failure_injection_idx]:
                    continue
                if len(taskUtil.failures_already_injected) == 0 or \
                    failure_injection_idx not in flatten_list([f[1] for f in taskUtil.failures_already_injected]):
                    return failure_injection_idx
            elif taskUtil.chosen_failure == 'failed_action':
                failure_injection_idx = np.random.choice(action_idxs[interact_cnt:])
                if "toggle_off" in actions[failure_injection_idx] or "close_obj" in actions[failure_injection_idx]:
                    continue
                if len(taskUtil.failures_already_injected) == 0 or \
                    failure_injection_idx not in flatten_list([f[1] for f in taskUtil.failures_already_injected]):
                    return failure_injection_idx
            elif taskUtil.chosen_failure == 'drop':
                failure_injection_idx = np.random.choice(nav_idxs[nav_cnt:])
                return failure_injection_idx

            if counter > 20:
                print(f"[INFO] Unable to inject a novel failure for failure type: {taskUtil.chosen_failure}. Choosing a new failuire type")            
                taskUtil.chosen_failure = np.random.choice(taskUtil.failures)
            if counter > 60:
                print("[INFO] Unable to inject a novel failure. Skipping this round. Maybe out of failures to inject.")
                return -1
            counter += 1
    except Exception as e:
        print("[INFO] Unable to inject a novel failure. Skipping this round. Maybe out of failures to inject:", e)
        return -1
    
def run_data_gen(data_path, task):
    np.random.seed(91)
    random.seed(91)

    # to avoid repetition in failure injection
    os.system("mkdir -p {}".format('thor_tasks/' + TASK_DICT[task["task_idx"]]))
    with open(f'thor_tasks/{TASK_DICT[task["task_idx"]]}/{task["folder_name"]}.pickle', 'wb') as handle:
        pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(int(task['num_samples'])):
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
        # print("controller.last_event.metadata: ", controller.last_event.metadata['agent'])
        reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        if 'chosen_failure' in task:
            chosen_failure = task['chosen_failure']
        else:
            chosen_failure = None
        reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        if 'failure_injection_params' in task:
            failure_injection_params = task['failure_injection_params']
        else:
            failure_injection_params = None
        taskUtil = TaskUtil(folder_name=os.path.join(TASK_DICT[task["task_idx"]], task['folder_name']),
                            controller=controller,
                            reachable_positions=reachable_positions,
                            failure_injection=task['failure_injection'],
                            index=i,
                            repo_path=data_path,
                            chosen_failure=chosen_failure,
                            failure_injection_params=failure_injection_params)

        # for injecting blocking failure, explicitly changing the location of relevant objects
        if taskUtil.chosen_failure in ['blocking', 'occupied', 'occupied_put'] and 'failure_injection_params' in task:
            place_obj(taskUtil, task['failure_injection_params'])
        if taskUtil.chosen_failure == 'wrong perception' and 'disp_x' in taskUtil.failure_injection_params:
            place_obj(taskUtil, task['failure_injection_params'])

        # Add preaction steps (e.g. set mug to be dirty)
        if "preactions" in task:
            for preaction_instr in task['preactions']:
                lis = preaction_instr.split(',')
                lis = [item.strip("() ") for item in lis]
                preaction = lis[0]
                params = lis[1:]
                func = globals()[preaction]
                retval = func(taskUtil, *params)

        instrs, new_instrs = [], []
        action_idxs, nav_idxs = [], []
        for i, instr in enumerate(task['actions']):
            instrs.append(instr)
            lis = instr.split(',')
            lis = [item.strip("() ") for item in lis]
            action = lis[0]
            if action in taskUtil.interact_action_primitives:
                action_idxs.append(i)
            if 'navigate_to_obj' == action:
                nav_idxs.append(i)

        if task['failure_injection']:
            failure_injection_idx = get_failure_injection_idx(taskUtil, instrs, task, action_idxs, nav_idxs)
            if failure_injection_idx == -1:
                continue
            print("failure_injection_idx: ", failure_injection_idx)

        nav_counter = 0
        interact_counter = 0
        for i, instr in enumerate(instrs):
            lis = instr.split(',')
            lis = [item.strip("() ") for item in lis]
            action = lis[0]
            params = lis[1:]
            # print("action, params: ", action, params)
            func = globals()[action]
            if action in taskUtil.interact_action_primitives:
                interact_counter += 1
            if 'navigate_to_obj' == action:
                nav_counter += 1
            
            # dropping injection
            to_drop = False
            if not taskUtil.failure_added and taskUtil.chosen_failure == 'drop' and i == failure_injection_idx:
                to_drop = True
                params.append(to_drop)
                params.append(failure_injection_idx)

            # missing step(s) injection (remove from interact action)
            if not taskUtil.failure_added and taskUtil.chosen_failure == 'missing_step' and \
                action in taskUtil.interact_action_primitives:
                if not isinstance(failure_injection_idx, list):
                    failure_injection_idx = [failure_injection_idx]
                if i in failure_injection_idx:
                    if 'gt_failure_reason' in taskUtil.gt_failure:
                        taskUtil.gt_failure['gt_failure_reason'] += ', ' + instr
                    else:
                        taskUtil.gt_failure['gt_failure_reason'] = 'Missing ' + instr
                    taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
                    if i == failure_injection_idx[-1]:
                        taskUtil.failure_added = True
                        taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
                    else:
                        taskUtil.failure_added = False
                    continue

            # failed execution injection 
            fail_execution = False
            if not taskUtil.failure_added and taskUtil.chosen_failure == 'failed_action' and \
                action in taskUtil.interact_action_primitives and i == failure_injection_idx:
                    print("Injecting failed action...")
                    fail_execution = True
                    taskUtil.gt_failure['gt_failure_reason'] = 'Failed to successfully execute ' + instr
                    taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
                    taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
                    taskUtil.failure_added = True
                    params.append(fail_execution)

            new_instrs.append(instr)
            do_action = True
            if do_action:
                retval = func(taskUtil, *params)
            else:
                retval = func(taskUtil, *params, fail_execution=True)

            # if dropped failure was not successfully injected, find a different failure instance
            if retval == False:
                failure_injection_idx = get_failure_injection_idx(taskUtil, instrs, task, action_idxs, nav_idxs, 
                                                                    interact_cnt=interact_counter, nav_cnt=nav_counter)
                if failure_injection_idx == -1:
                    break

        # adding two buffer frames in the end
        for _ in range(2):
            e = controller.step(action="Done")
            save_data(taskUtil, e)

        print("[INFO] interact_actions:", taskUtil.interact_actions)
        print("[INFO] nav_actions:", taskUtil.nav_actions)

        with open(f'thor_tasks/{taskUtil.specific_folder_name}/interact_actions.pickle', 'wb') as handle:
            pickle.dump(taskUtil.interact_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'thor_tasks/{taskUtil.specific_folder_name}/nav_actions.pickle', 'wb') as handle:
            pickle.dump(taskUtil.nav_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'thor_tasks/{TASK_DICT[task["task_idx"]]}/{task["folder_name"]}.pickle', 'wb') as handle:
            pickle.dump(taskUtil.failures_already_injected, handle, protocol=pickle.HIGHEST_PROTOCOL)

        updated_task = task.copy()
        updated_task['specific_folder_name'] = taskUtil.specific_folder_name
        # in case no failure was added
        if 'gt_failure_reason' not in taskUtil.gt_failure:
            taskUtil.gt_failure['gt_failure_reason'] = 'No failure added'
            taskUtil.gt_failure['gt_failure_step'] = 0
        
        if 'gt_failure_reason' not in updated_task:
            updated_task['gt_failure_reason'] = taskUtil.gt_failure['gt_failure_reason']
            updated_task['gt_failure_step'] = convert_step_to_timestep(taskUtil.gt_failure['gt_failure_step'], video_fps=1)
        updated_task['unity_name_map'] = taskUtil.unity_name_map
        updated_task['sounds'] = taskUtil.sounds
        updated_task['actions'] = new_instrs
        with open(f'thor_tasks/{taskUtil.specific_folder_name}/task.json', 'w') as f:
            json.dump(updated_task, f)
        
        # print("[INFO] interact actions: ", taskUtil.interact_actions)
        # print("[INFO] nav_actions: ", taskUtil.nav_actions)

        # save video of the executed task
        generate_video(taskUtil, recovery_video=False)

        # end of task steps
        controller.stop()
        