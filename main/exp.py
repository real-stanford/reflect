import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import pickle
from constants import *
from scene_graph import SceneGraph
from scene_graph import Node as SceneGraphNode
from get_local_sg import get_scene_graph, save_pcd
import numpy as np
from data import *
from utils import *
from clip_utils import *
from audio import process_sound, audio2label
from point_cloud_utils import *

def run_sound_module(folder_name, object_list):
    detected_sounds = []
    try:
        detected_sounds = process_sound(f'thor_tasks/{folder_name}', object_list)
        print("detected sounds:", detected_sounds)
    except Exception as e:
        print(e)
    return detected_sounds


def get_scene_text(scene_graph):
    output = ""
    visited = []
    for node in set(scene_graph.nodes):
        output += (node.get_name() + ", ")

    if len(output) != 0:
        output = output[:-2] + ". "
    for edge_key, edge in scene_graph.edges.items():
        start_name, end_name = edge_key
        edge_key_2 = (end_name, start_name)
        if (edge_key not in visited and edge_key_2 not in visited):
            output += edge.start.name + " is " + edge.edge_type + " " + edge.end.name
            output += ". "
        visited.append(edge_key)
    output = output[:-1]

    return output


def get_held_object(folder_name, step_idx):
    found = False
    while not found:
        if os.path.exists(f'state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl'):
            found = True
            with open(f'state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl', 'rb') as f:
                local_sg = pickle.load(f)
            for key in local_sg.edges:
                if "robot gripper" in key and key[0] != "nothing":
                    return key[0]
        else:
            step_idx -= 1


def generate_scene_graphs(folder_name, events, object_list, nav_actions, interact_actions, WITH_AUDIO, detected_sounds):
    with open(f'thor_tasks/{folder_name}/task.json') as f:
        task = json.load(f)

    if not os.path.exists(f'state_summary/{folder_name}/global_sg.pkl'):
        # sensory-input summary
        os.system(f'mkdir -p state_summary/{folder_name}/local_graphs')
        # os.system("mkdir -p scene/{}".format(folder_name))
        key_frames = []
        prev_graph = SceneGraph(event=None, task=task)
        total_points_dict, bbox3d_dict = {}, {}
        obj_held_prev = None
        cnt, interval = 0, 2
        nav_actions_end_indices = [idx[1] for idx in nav_actions.keys()]
        for step_idx, event in enumerate(events):
            # uniformly drop intermediate navigation frames with no sound
            if (step_idx+1) not in interact_actions and ((step_idx+1) not in nav_actions_end_indices):
                cnt += 1
                if WITH_AUDIO == 1:
                    if step_idx not in detected_sounds and cnt % interval == 0:
                        continue
                elif WITH_AUDIO == 0:
                    if str(step_idx) not in task['sounds'] and cnt % interval == 0:
                        continue

            print("[Frame] " + str(step_idx+1))

            local_sg, total_points_dict, obj_held_prev, bbox3d_dict = get_scene_graph(step_idx, event, object_list, 
                                                                                      total_points_dict, bbox3d_dict, 
                                                                                      obj_held_prev, task)
            print("========================[Current Graph]=====================")
            print(local_sg)

            # 1. Select keyframe based on scene graph difference
            if local_sg != prev_graph:
                if (step_idx+1) not in key_frames:
                    key_frames.append(step_idx+1)
                    prev_graph = local_sg

            # 2. Select keyframe based on actions
            if (step_idx+1) in interact_actions or (step_idx+1) in nav_actions_end_indices:
                if (step_idx+1) not in key_frames:
                    key_frames.append(step_idx+1)

            # 3. Select keyframe based on audio
            if WITH_AUDIO == 0:
                if str(step_idx) in task['sounds']:
                    if (step_idx+1) not in key_frames:
                        key_frames.append(step_idx+1)
            elif WITH_AUDIO == 1:
                if step_idx in detected_sounds:
                    if (step_idx+1) not in key_frames:
                        key_frames.append(step_idx+1)

            with open(f'state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl', 'wb') as f:
                pickle.dump(local_sg, f)

        with open('state_summary/{}/L1_key_frames.txt'.format(folder_name), 'w') as f:
            for frame in key_frames:
                f.write("%i\n" % frame)

        # save_pcd(folder_name, total_points_dict)

        # ======================Get global graph========================
        global_sg = SceneGraph(events[-1], task)
        for label in total_points_dict.keys():
            name = get_label_from_object_id(label, events, task)
            if name is not None:
                new_node = SceneGraphNode(name=name, object_id=label, pos3d=bbox3d_dict[label].get_center(), 
                        corner_pts=np.array(bbox3d_dict[label].get_box_points()),
                        pcd=total_points_dict[label], global_node=True)
                global_sg.add_node_wo_edge(new_node)

        for label in total_points_dict.keys():
            object_name = label.split("|")[0]
            if object_name in object_list:
                name = get_label_from_object_id(label, events, task)
                if name is not None:
                    for node in global_sg.total_nodes:
                        if node.name == name:
                            global_sg.add_node(node)
            
        global_sg.add_agent()
        with open(f'state_summary/{folder_name}/global_sg.pkl', 'wb') as f:
            pickle.dump(global_sg, f)
        # ===============================================================


def generate_summary(folder_name, events, nav_actions, interact_actions, WITH_AUDIO, detected_sounds):
    with open(f'thor_tasks/{folder_name}/task.json') as f:
        task = json.load(f)

    key_frames = []
    with open('state_summary/{}/L1_key_frames.txt'.format(folder_name), 'r') as f:
        frames = f.readlines()
        key_frames = [int(frame) for frame in frames]

    # event-based summary
    if not os.path.exists(f'state_summary/{folder_name}/state_summary_L1.txt'):
    # if True:
        print("[INFO] Start generating event-based summary")
        state_summary_L1 = ""
        L1_captions = []
        for step_idx, event in enumerate(events):
            if not os.path.exists(f'state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl'):
                continue
            if (step_idx+1) in key_frames:
                caption = ""

                # add action
                if (step_idx+1) in interact_actions:
                    caption += f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. Action: {interact_actions[step_idx+1]}."
                    # action = interact_actions[step_idx+1]
                else:
                    for key in nav_actions:
                        min_step, max_step = key
                        if min_step <= (step_idx+1) <= max_step:
                            caption += f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. Action: {nav_actions[key]}."
                            # action = nav_actions[key]

                if len(caption) == 0:
                    continue
                
                with open(f'state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl', 'rb') as f:
                    local_sg = pickle.load(f)
                    scene_text = get_scene_text(local_sg)
                    caption += f" Visual observation: {scene_text}"

                # Add audio info.
                if WITH_AUDIO == 0:
                    if str(step_idx) in task['sounds']:
                        if 'drop' in task['sounds'][str(step_idx)] and get_held_object(folder_name, step_idx-1) is not None:
                            caption += f" Auditory observation: something drops."
                        else:
                            caption += f" Auditory observation: {audio2label[task['sounds'][str(step_idx)]]}."
                elif WITH_AUDIO == 1:
                    if (step_idx+1) in detected_sounds:
                        caption += f" Auditory observation: {detected_sounds[step_idx+1]}."

                caption += "\n"

                state_summary_L1 += caption
                L1_captions.append(caption)
        with open('state_summary/{}/state_summary_L1.txt'.format(folder_name), 'w') as f:
            f.write(state_summary_L1)
            print("[INFO] Write event-based summary")
    else:
        print("[INFO] Event-based summary already generated")
        L1_captions = []
        state_summary_L1 = ""
        with open('state_summary/{}/state_summary_L1.txt'.format(folder_name), 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)

    # subgoal-based summary
    if not os.path.exists(f'state_summary/{folder_name}/state_summary_L2.txt'):
    # if True:
        print("[INFO] Start generating subgoal-based summary")
        L2_captions = []
        for caption in L1_captions:
            step_num = convert_timestep_to_step(caption.split(".")[0], video_fps=1)
            if step_num in interact_actions:
                L2_captions.append(caption.replace("Action", "Goal"))

        state_summary_L2 = "".join(L2_captions)
        with open('state_summary/{}/state_summary_L2.txt'.format(folder_name), 'w') as f:
            f.write(state_summary_L2)
            print("[INFO] Write subgoal-based summary")
    else:
        print("[INFO] Subgoal-based summary already generated")
        L2_captions = []
        L2_file_name = 'state_summary/{}/state_summary_L2.txt'.format(folder_name)
        if os.path.exists(L2_file_name):
            with open(L2_file_name, 'r') as f:
                L2_captions = f.readlines()
            state_summary_L2 = "".join(L2_captions)


def run_reasoning(folder_name, llm_prompter, global_sg):
    with open(f'thor_tasks/{folder_name}/task.json') as f:
        task = json.load(f)
    
    if os.path.exists(f'state_summary/{folder_name}/reasoning.json'):
        print("[INFO] Reasoning already generated")
        with open(f'state_summary/{folder_name}/reasoning.json', 'r') as f:
            reasoning_dict = json.load(f)
        return
    else:
        reasoning_dict = {}

    save_dir = f'../LLM/{folder_name}'
    # os.system("mkdir -p {}".format(save_dir))

    with open('../LLM/prompts.json', 'r') as f:
        prompt_info = json.load(f)

    # Load L2 captions from state_summary_L2.txt
    with open('state_summary/{}/state_summary_L2.txt'.format(folder_name), 'r') as f:
        L2_captions = f.readlines()

    # Load L1 captions from state_summary_L1.txt
    with open('state_summary/{}/state_summary_L1.txt'.format(folder_name), 'r') as f:
        L1_captions = f.readlines()
    
    # Loop through each subgoal and check for post-condition
    print(">>> Run step-by-step subgoal-level analysis...")
    selected_caption = ""
    prompt = {}

    for caption in L2_captions:
        print(">>> Verify subgoal...")
        subgoal = caption.split(". ")[1].split(": ")[1].lower()

        prompt['system'] = prompt_info['subgoal-verifier']['template-system']
        prompt['user'] = prompt_info['subgoal-verifier']['template-user'].replace("[SUBGOAL]", subgoal).replace("[OBSERVATION]", caption[caption.find("Visual observation"):])

        ans, _  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['subgoal-verifier']['params'], 
                                    save=prompt_info['subgoal-verifier']['save'], save_dir=save_dir)
        is_success = int(ans.split(", ")[0] == "Yes")
        if is_success == 0:
            selected_caption = caption
            print(f"[INFO] Failure identified in subgoal [{subgoal}] at {caption.split('.')[0]}")
            break
        else:
            print(f"[INFO] Subgoal [{subgoal}] succeeded!")

    if len(selected_caption) != 0:
            print(">>> Get detailed reasoning from L1...")
            step_name = selected_caption.split(".")[0]
            for _, caption in enumerate(L1_captions):
                if step_name in caption:
                    action = caption.split(". ")[1].split(": ")[1].lower()
                    prev_observations = get_robot_plan(folder_name, step=step_name, with_obs=True)
                    if len(prev_observations) != 0:
                        prompt_name = 'reasoning-execution'
                    else:
                        prompt_name = 'reasoning-execution-no-history'
                    prompt['system'] = prompt_info[prompt_name]['template-system']
                    prompt['user'] = prompt_info[prompt_name]['template-user'].replace("[ACTION]", action)
                    prompt['user'] = prompt['user'].replace("[TASK_NAME]", task['name'])
                    prompt['user'] = prompt['user'].replace("[STEP]", step_name)
                    prompt['user'] = prompt['user'].replace("[SUMMARY]", prev_observations)
                    prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Action"):])
                    ans, _  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info[prompt_name]['params'], 
                                                save=prompt_info[prompt_name]['save'], save_dir=save_dir)

                    print("[INFO] Predicted failure reason:", ans)
                    reasoning_dict['pred_failure_reason'] = ans

                    prompt = {}
                    prompt['system'] = prompt_info['reasoning-execution-steps']['template-system']
                    prompt['user'] = prompt_info['reasoning-execution-steps']['template-user'].replace("[FAILURE_REASON]", ans)
                    time_steps, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['reasoning-execution-steps']['params'],
                                                            save=prompt_info['reasoning-execution-steps']['save'], save_dir=save_dir)
                    
                    print("[INFO] Predicted failure time steps:", time_steps, time_steps.split(", "))
                    reasoning_dict['pred_failure_step'] = [time_step.replace(",", "") for time_step in time_steps.split(", ")]
                    break
    else:
        print(">>> All actions are executed successfully, run plan-level analysis...")

        prompt['system'] = prompt_info['reasoning-plan']['template-system']
        prompt['user'] = prompt_info['reasoning-plan']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt['user'] = prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])
        prompt['user'] = prompt['user'].replace("[CURRENT_STATE]", get_scene_text(global_sg))
        prompt['user'] = prompt['user'].replace("[OBSERVATION]", get_robot_plan(folder_name, step=None, with_obs=False))
        ans, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['reasoning-plan']['params'], 
                                    save=prompt_info['reasoning-plan']['save'], save_dir=save_dir)
        
        print("[INFO] Predicted failure reason:", ans)
        reasoning_dict['pred_failure_reason'] = ans

        prompt['system'] = prompt_info['reasoning-plan-steps']['template-system']
        prompt['user'] = prompt_info['reasoning-plan-steps']['template-user'].replace("[PREV_PROMPT]", prompt['user'] + " " + ans)
        step, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['reasoning-plan-steps']['params'], 
                                    save=prompt_info['reasoning-plan-steps']['save'], save_dir=save_dir)
        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]

        print("[INFO] Predicted failure time steps:", step_str)
        reasoning_dict['pred_failure_step'] = step_str

    reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
    reasoning_dict['gt_failure_step'] = task['gt_failure_step']
    
    with open('state_summary/{}/{}'.format(folder_name, 'reasoning.json'), 'w') as f:
        json.dump(reasoning_dict, f)


def generate_replan(folder_name, llm_prompter, global_sg, last_event, task_object_list):
    with open(f'thor_tasks/{folder_name}/task.json') as f:
        task = json.load(f)
    curr_state = get_scene_text(global_sg)
    print("[INFO] Current state:", curr_state)
    global_object_list = list(set([obj["objectType"] for obj in last_event.metadata["objects"]]) | set(task_object_list))

    with open('state_summary/{}/reasoning.json'.format(folder_name), 'r') as f:
        data = json.load(f)
        reason = data["pred_failure_reason"]

    if os.path.exists(f'state_summary/{folder_name}/replan.json'):
        print("[INFO] Skipping replan generation")
        with open(f'state_summary/{folder_name}/replan.json', 'r') as f:
            plan = json.load(f)["original_plan"]
            plan = "\n".join(plan)
    else:
        with open('../LLM/prompts.json', 'r') as f:
            prompt_info = json.load(f)

        prompt = {}
        prompt['system'] = prompt_info['correction']['template-system'].replace("[PREFIX]", get_replan_prefix())
        prompt['user'] = prompt_info['correction']['template-user'].replace("[TASK_NAME]", task['name']).replace("[PLAN]", get_initial_plan(task['actions']))
        prompt['user'] = prompt['user'].replace("[FAILURE_REASON]", reason)
        prompt['user'] = prompt['user'].replace("[CURRENT_STATE]", curr_state).replace("[SUCCESS_CONDITION]", task['success_condition'])
    
        # print("=====================RE-PLAN PROMPT START========================")
        # print(prompt['system'])
        # print(prompt['user'])
        # print("=====================RE-PLAN PROMPT END==========================")

        plan, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['correction']['params'], 
                                save=prompt_info['correction']['save'], save_dir=f'../LLM/{folder_name}')

    translated_plan = translate_plan(plan, global_object_list, last_event)
    print("========================Translated plan===========================")
    print(translated_plan)

    replan_dict = {}

    with open(f'state_summary/{folder_name}/replan.json', 'w') as f:
        replan_dict["original_plan"] = plan.split("\n")
        replan_dict["plan"] = translated_plan.split("\n")[:-1]
        replan_dict["num_steps"] = len(replan_dict["plan"])
        json_object = json.dumps(replan_dict, indent=4)
        f.write(json_object)
