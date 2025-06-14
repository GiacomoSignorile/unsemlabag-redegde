import cv2
import numpy as np

from utils.utils import get_hough_labels, imap2rgb


def generate_single_pose_label(
    simulator, mapper, pose, rho_old, theta_old, x_old, prev_col_exceeding_lines, turning_point
):
    measurement = simulator.get_measurement(pose)

    # Check if prev_col_exceeding_lines has enough elements; if not, use default values.
    if len(prev_col_exceeding_lines) > len(x_old) and prev_col_exceeding_lines[len(x_old)] != 0:
        all_predictions, all_rho, all_theta, all_x, all_exceed_horiz, all_lines = get_hough_labels(
            measurement["image"], rho_old, theta_old, x_old, prev_col_exceeding_lines[len(x_old)], one_hot_encoded=True
        )
    else:
        all_predictions, all_rho, all_theta, all_x, all_exceed_horiz, all_lines = get_hough_labels(
            measurement["image"], rho_old, theta_old, x_old, {"start": 0, "end": 0, "size": 0}, one_hot_encoded=True
        )

    # Process each line's results
    for prediction, rho, theta, x, exceed_horiz, lines in zip(
        all_predictions, all_rho, all_theta, all_x, all_exceed_horiz, all_lines
    ):
        theta_old, x_old, rho_old, prev_col_exceeding_lines = update(
            theta_old, theta, x_old, x, rho_old, rho, prev_col_exceeding_lines, exceed_horiz
        )

        prediction = label_weeds(prediction, lines)
        mapper.update_map({"semantics": prediction, "fov": measurement["fov"], "gsd": measurement["gsd"]})

    # Reset if turning point is reached
    if len(x_old) == turning_point:
        x_old, rho_old = reset()

    return rho_old, theta_old, x_old, prev_col_exceeding_lines


def generate_poses(rows, cols):
    zero_step = 0.512
    add_step = 1.536 - 0.512

    poses = []
    for i in range(0, rows):
        for j in range(0, cols):
            poses.append(np.array([0.1 + zero_step + (add_step / 2) * i, 0.1 + zero_step + add_step * j, zero_step]))
    return poses


def reset():
    return [], []


def update(theta_old, theta, x_old, x, rho_old, rho, prev_col_exceeding_lines, exceed_horiz):
    if theta_old == -1:
        theta_old = theta
    else:
        theta_old = (0.5 * theta_old + 1.5 * theta) / 2

    # Ensure these are lists
    if not isinstance(rho_old, list):
        rho_old = list(rho_old)
    if not isinstance(x_old, list):
        x_old = list(x_old)
    if not isinstance(prev_col_exceeding_lines, list):
        prev_col_exceeding_lines = list(prev_col_exceeding_lines)

    rho_old.append(rho)
    x_old.append(x)

    index = len(x_old) - 1
    if index >= len(prev_col_exceeding_lines):
        prev_col_exceeding_lines.append(0)

    if exceed_horiz == 1:
        prev_col_exceeding_lines[index] = exceed_horiz
    elif prev_col_exceeding_lines[index] != 0:
        prev_col_exceeding_lines[index] = 0

    return theta_old, x_old, rho_old, prev_col_exceeding_lines


def label_weeds(prediction, line_mask):
    """
    Refines the prediction by re-labeling 'unknown' components as 'weed' if they
    are sufficiently far from any detected crop row line.

    Args:
        prediction (np.array): A 4-channel one-hot encoded prediction map (C, H, W).
                               Channel 1: crop, Channel 2: weed, Channel 3: unknown.
        line_mask (np.array): A binary mask (H, W) where detected crop rows are non-zero.

    Returns:
        np.array: The modified 4-channel prediction map.
    """
    # Get the single-channel mask for unknown components (class 3)
    unknown_mask = prediction[3].astype(np.uint8)
    if np.sum(unknown_mask) == 0:
        return prediction # No unknown components to process, return early

    # Find all connected components in the 'unknown' mask
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(unknown_mask)

    # --- Calculate a robust threshold for distance ---
    # Get stats for components already labeled as 'crop' (class 1)
    # We use the width of the crop components as a reference for our threshold
    crop_mask = prediction[1].astype(np.uint8)
    num_crop_labels, _, crop_stats, _ = cv2.connectedComponentsWithStats(crop_mask)
    
    # Use the median width of crop components as a robust measure.
    # We ignore the background component (stat[0]).
    if num_crop_labels > 1:
        crop_widths = crop_stats[1:, cv2.CC_STAT_WIDTH]
        # Use median instead of mean/std to be robust to outliers
        median_crop_width = np.median(crop_widths)
    else:
        # If no crops are found, fall back to a reasonable default pixel value
        median_crop_width = 15 # A reasonable default width in pixels

    # Define the distance threshold. A component must be further than ~1.5 crop widths away.
    distance_threshold = 1.5 * median_crop_width

    # --- Use Distance Transform for efficient distance calculation ---
    # It's much faster than manual numpy calculations in a loop.
    # The transform calculates for each pixel the distance to the nearest zero pixel.
    # We need to invert the line_mask so that crop rows are zero.
    
    # Ensure line_mask is binary (0 or 255) for the transform
    line_mask_binary = (line_mask > 0).astype(np.uint8) * 255
    
    dist_map = cv2.distanceTransform(255 - line_mask_binary, cv2.DIST_L2, 5)

    # --- Iterate through 'unknown' components and re-label if they are far away ---
    # We start from 1 to skip the background component (label 0)
    for k in range(1, num_labels):
        # Get the centroid (center) of the current component
        centroid_x, centroid_y = int(centroids[k][0]), int(centroids[k][1])
        
        # Look up the distance from the centroid to the nearest crop row
        # using our pre-calculated distance map.
        distance_to_row = dist_map[centroid_y, centroid_x]

        if distance_to_row > distance_threshold:
            # This component is far from any crop row, re-label it as a weed.
            component_mask = (labels_im == k)
            
            # Set 'unknown' channel (3) to 0 for these pixels
            prediction[3][component_mask] = 0
            # Set 'weed' channel (2) to 1 for these pixels
            prediction[2][component_mask] = 1

    return prediction