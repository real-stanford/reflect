import os
import pickle

def get_object_list_from_actions(actions):
    object_list = set()
    for action in actions:
        params = action[1:-1].split(", ")
        for obj in params[1:]:
            if "CounterTop" not in obj: # ignore CounterTop
                object_list.add(obj.split("-")[0])

    return list(object_list)

def load_data(task_path, task):
    object_list = get_object_list_from_actions(task['actions']) # task['object_list']

    if os.path.exists(task_path):
        num_events = len([f for f in os.listdir(os.path.join(task_path, "events")) if f.endswith(".pickle")])
        events = []
        for event_idx in range(num_events):
            with open(os.path.join(task_path, "events", "step_{}.pickle".format(event_idx+1)), 'rb') as f:
                event = pickle.load(f)
                events.append(event)
        with open(os.path.join(task_path, "interact_actions.pickle"), 'rb') as f:
            interact_actions = pickle.load(f)
        with open(os.path.join(task_path, "nav_actions.pickle"), 'rb') as f:
            nav_actions = pickle.load(f)

        print("interact_actions: ", interact_actions)
        print("nav_actions: ", nav_actions)
        print("length of events: ", len(events))

        return events, task, object_list, interact_actions, nav_actions
    else:
        raise ValueError("Queried folder does not exist.")
