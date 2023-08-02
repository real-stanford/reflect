import math
import torch


def make_pinhole_camera_matrix(hfov_deg, width_px, height_px):
    f_x = width_px * 0.5 / math.tan(math.radians(hfov_deg) * 0.5)
    f_y = f_x
    K = torch.tensor([
        [f_y, 0, height_px / 2],
        [0, f_x, width_px / 2],
        [0, 0, 1]], dtype=torch.float32)
    return K


def make_pinhole_camera_matrix_4f(hfov_deg, width_px, height_px):
    f_x = width_px * 0.5 / math.tan(hfov_deg * 0.5 * math.pi / 180)
    f_y = f_x
    K = torch.tensor([
        [f_y, 0, 0, height_px / 2],
        [0, f_x, 0, width_px / 2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    return K