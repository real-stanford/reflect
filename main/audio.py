import os
import wav2clip
import numpy as np
import itertools
import pickle
from PIL import Image
from moviepy.editor import VideoFileClip

from clip_utils import get_text_feats, get_nn_text_w_audio, get_img_feats
from constants import NAME_MAP

audio2label = {
    "toggle-on-faucet.wav": "water runs in sink",
    "toggle-on-toaster.wav": "toaster turns on",
    "toggle-on-stoveburner.wav": "stove burner turns on",
    "drop-pot.wav": "object drops or cracks on hard surface",
    "drop-plastic-bowl.wav": "object drops or cracks on hard surface",
    "drop-egg.wav": "object drops or cracks on hard surface",
    "slice-bread.wav": "slice bread",
    "toggle-on-microwave.wav": "microwave turns on",
    "toggle-on-coffeemachine.wav": "coffee machine turns on",
    "pour-water-in-sink.wav": "water runs in sink",
    "open-fridge.wav": "fridge opens",
    "close-fridge.wav": "fridge closes",
    "open-microwave.wav": "microwave opens",
    "close-microwave.wav": "microwave closes",
    "crack-egg.wav": "egg cracks"
}

TEXT_LIST = list(set([item for item in audio2label.values()]))

model = wav2clip.get_model()
model.eval()

def process_sound(data_path, object_list=None):
    def to_ranges(iterable):
        iterable = sorted(set(iterable))
        for key, group in itertools.groupby(enumerate(iterable),
                                            lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]

    with open(os.path.join(data_path, "interact_actions.pickle"), 'rb') as f:
        interact_actions = pickle.load(f)
    interact_steps = interact_actions.keys()

    pred_sounds = {}
    clip = VideoFileClip(os.path.join(data_path, "original-video.mp4"))
    total_frames = int(clip.fps * clip.duration)

    frames_w_sound = []
    for cur_frame in range(0, total_frames):
        subclip = clip.subclip(cur_frame, cur_frame+1)
        max_volume = subclip.audio.max_volume()

        if max_volume > 0.01:
            frames_w_sound.append(cur_frame)

    frame_ranges = list(to_ranges(frames_w_sound))
    # print("frame ranges:", frame_ranges)

    text_list = []
    for label in TEXT_LIST:
        if "drops" in label:
            text_list.append(label)
        for obj_class in object_list:
            if obj_class in NAME_MAP:
                obj_name = NAME_MAP[obj_class]
            else:
                obj_name = obj_class.lower()
            if obj_name in label:
                text_list.append(label)
                break
    # print("text list:", text_list)
    text_feats = get_text_feats(text_list)
    # print("text embedding shape", text_feats.shape)

    for frame_range in frame_ranges:
        # print(f"FRAME {start_frame}")
        subclip = clip.subclip(frame_range[0], frame_range[1]+1)
        signal = subclip.audio.to_soundarray().astype(np.float32)

        if len(signal.shape) == 2 and signal.shape[1] == 2:
            signal = np.mean(signal, axis=1)

        norm_signal = signal / np.max(np.abs(signal))
        audio_feats = wav2clip.embed_audio(norm_signal, model)[0]
        audio_feats /= np.linalg.norm(audio_feats)

        img = np.array(Image.open(os.path.join(data_path, 'ego_img/img_step_{}.png'.format(frame_range[0]+1))).convert("RGB"))
        img_feats = get_img_feats(img)[0]
        # print("image embedding shape", img_feats.shape)

        top_k, weight = 3, 0
        for frame in range(frame_range[0], frame_range[1]+1):
            if frame in interact_steps:
                weight = 2
        sorted_texts, sorted_scores = get_nn_text_w_audio(text_list, text_feats, img_feats, audio_feats, weight=weight)
        # for text, score in zip(sorted_texts[:top_k], sorted_scores[:top_k]):
        #     print(f"weight: {weight}, score: {score}, event: {text}")
        # print("=========================================")
        
        for frame in range(frame_range[0], frame_range[1]+1):
            pred_sounds[frame] = sorted_texts[0]

    return pred_sounds
