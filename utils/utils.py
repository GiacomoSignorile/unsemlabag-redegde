import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.hough import Hough

LABELS = {
    "soil": {"color": (0, 0, 255), "id": 0},
    "crop": {"color": (0, 255, 0), "id": 1},
    "weed": {"color": (255, 0, 0), "id": 2},
    "unknown": {"color": (255, 255, 255), "id": 3},
}


def imap2rgb(imap, channel_order):
    """
    Convert a label image into an RGB image using predefined label colors.

    Arguments:
        imap {np.ndarray} -- Label image (H, W)
        channel_order {str} -- "hwc" or "chw"
        
    Returns:
        np.ndarray -- RGB label image.
    """
    assert channel_order in ["hwc", "chw"]
    assert imap.ndim == 2

    rgb = np.zeros((imap.shape[0], imap.shape[1], 3), np.float32)
    for _, cl in LABELS.items():
        if cl["color"] == (0, 0, 0):
            continue
        mask = np.where(imap == cl["id"], 1, 0).reshape((imap.shape[0], imap.shape[1], 1))
        rgb += mask * cl["color"]
    if channel_order == "chw":
        rgb = np.moveaxis(rgb, -1, 0)
    return rgb


def get_fov(pose: np.array, sensor_angle: List, gsd: float, world_range: List):
    half_fov_size = pose[2] * np.tan(np.deg2rad(sensor_angle))

    lu = [pose[0] - half_fov_size[0], pose[1] - half_fov_size[1]]
    ru = [pose[0] + half_fov_size[0], pose[1] - half_fov_size[1]]
    rd = [pose[0] + half_fov_size[0], pose[1] + half_fov_size[1]]
    ld = [pose[0] - half_fov_size[0], pose[1] + half_fov_size[1]]
    corner_list = np.array([lu, ru, rd, ld])

    lu_index = [np.floor(lu[0] / gsd).astype(int), np.floor(lu[1] / gsd).astype(int)]
    ru_index = [np.ceil(ru[0] / gsd).astype(int), np.floor(ru[1] / gsd).astype(int)]
    rd_index = [np.ceil(rd[0] / gsd).astype(int), np.ceil(rd[1] / gsd).astype(int)]
    ld_index = [np.floor(ld[0] / gsd).astype(int), np.ceil(ld[1] / gsd).astype(int)]

    index_list = np.array([lu_index, ru_index, rd_index, ld_index])
    min_x = np.min(index_list[:, 0])
    max_x = np.max(index_list[:, 0])
    min_y = np.min(index_list[:, 1])
    max_y = np.max(index_list[:, 1])

    if np.any(np.array([min_x, min_y]) < 0) or np.any(np.array([max_x, max_y]) > np.array(world_range[:2])):
        raise ValueError("Measurement out of environment bounds.")

    return corner_list, [min_x, max_x, min_y, max_y]


def get_hough_labels(image: np.array, rho_old, theta_old, x_old, hor_line_propag, one_hot_encoded: bool = False):
    hough = Hough()
    with torch.no_grad():
        labels, horizontal_exc, rho, theta, x, lines = hough.forward(
            image, rho_old, theta_old, x_old, hor_line_propag
        )

    if one_hot_encoded:
        labels = torch.nn.functional.one_hot(
            torch.tensor(labels).long(), num_classes=4
        ).movedim(-1, 0).float()

    return labels.cpu().numpy(), rho, theta, x, horizontal_exc, lines


def save_preds(sem, preds, unc, rgb, names):
    results_dir = "./results/"

    for b in range(preds.shape[0]):
        network_pred = sem[b].squeeze().cpu().numpy()
        corrected_pred = preds[b].squeeze().cpu().numpy()
        current_rgb = rgb[b].squeeze().permute(1, 2, 0).cpu().numpy()
        uncertainty = unc[b].squeeze().cpu().numpy()

        output_filename = os.path.join(results_dir, names[b])
        output_directory = os.path.dirname(output_filename)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        
        if current_rgb.max() > 1.0:
            current_rgb = current_rgb / 255.0
        current_rgb = np.clip(current_rgb, 0, 1)

        ax[0, 0].imshow(current_rgb)
        ax[0, 0].set_title("RGB Image")
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])

        ax[0, 1].imshow(network_pred)
        ax[0, 1].set_title("Network Prediction")
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])

        ax[1, 0].imshow(uncertainty)
        ax[1, 0].set_title("Uncertainty")
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])

        ax[1, 1].imshow(corrected_pred)
        ax[1, 1].set_title("Post-Processed Prediction")
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])

        plt.tight_layout()
        print(f"Saving prediction visualization to: {output_filename}")
        plt.savefig(output_filename)
        plt.close(fig)


def save_images(name, map_rgb, img_rgb):
    dir_name = os.path.dirname(name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(map_rgb.shape[1] / 300, map_rgb.shape[0] / 300)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(map_rgb)
    plt.savefig(name, dpi=300)
    plt.close()
