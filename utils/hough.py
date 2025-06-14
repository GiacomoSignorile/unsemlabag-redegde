import cv2
import numpy as np
from PIL import Image


class Hough:
    def __init__(self, pixel_res=1, angle_res=180, min_theta=1.45, max_theta=1.70, min_voting=300):
        self.pixel_res = pixel_res
        self.angle_res = angle_res
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_voting = min_voting

    def forward(self, image, rho_old=[], theta_old=-1, x_old=[], horizontal_exceed={}):
        self.extract_vegetation_mask(image)
        if self.binary_mask.sum() == 0:
            print("DEBUG: No vegetation found (binary_mask empty).")
            return [(self.binary_mask, horizontal_exceed, -1, -1, -1, -1)]
        
        lines = cv2.HoughLines(
            self.binary_mask,
            self.pixel_res,
            self.angle_res,
            self.min_voting,
            min_theta=self.min_theta,
            max_theta=self.max_theta,
        )
        if lines is None:
            print("DEBUG: No lines detected by Hough transform.")
            # --- FIX: Return a list containing one "failure" tuple ---
            # All vegetation is marked as 'unknown' (class 3) when no row is found
            fallback_mask = np.zeros_like(self.binary_mask, dtype=np.uint8)
            fallback_mask[self.binary_mask == 1] = 3 
            return [(fallback_mask, {}, -1, -1, -1, (self.binary_mask*0).astype(bool))]
        print(f"DEBUG: Detected {len(lines)} lines.")
        line_results = []
        for idx, line in enumerate(lines):
            rho, theta = line[0]
            x1 = int(rho / np.cos(theta))
            x2 = int((rho - image.shape[1] * np.sin(theta)) / np.cos(theta))
            print(f"DEBUG: Line {idx}: rho={rho:.2f}, theta={theta:.2f}, x1={x1}, x2={x2}")
            # Create separate mask for the line
            line_mask = np.zeros_like(image, dtype=np.uint8)
            cv2.line(line_mask, (x1, 0), (x2, image.shape[1]), (255, 255, 255), 15)

            # Propagate vertical rows
            line_mask = self.propagate_vertical_rows(line_mask, x1, x_old, rho_old, theta_old)

            # Include lines propagated from the x-axis
            line_mask = cv2.line(
                line_mask,
                (5, horizontal_exceed.get("start", 0)),
                (5, image.shape[1] - horizontal_exceed.get("end", 0)),
                (255, 255, 255),
                15,
            )
            line_mask = line_mask[:, :, 0].astype(bool)
            final_mask, horizontal_dict = self.generate_hough_line_label(line_mask)
            print(f"DEBUG: Line {idx}: final_mask unique values: {np.unique(final_mask)}")
            line_results.append((final_mask, horizontal_dict, rho, theta, x2, line_mask))
        
        return line_results

        # x0 = rho * np.cos(theta)
        # y0 = rho * np.sin(theta)
        # y1 = int(0)
        # x1 = int(rho / np.cos(theta))
        # y2 = int(image.shape[1])
        # x2 = int((rho - y2 * np.sin(theta)) / np.cos(theta))

        # line_mask = np.zeros_like(image, dtype=np.uint8)
        # cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 15)

        # # include lines propagated from the y axis
        # line_mask = self.propagate_vertical_rows(line_mask, x1, x_old, rho_old, theta_old)

        # # include lines propagated from the x axis
        # line_mask = cv2.line(
        #     line_mask,
        #     (5, horizontal_exceed["start"]),
        #     (5, image.shape[1] - horizontal_exceed["end"]),
        #     (255, 255, 255),
        #     15,
        # )
        # line_mask = line_mask[:, :, 0].astype(bool)
        # final_mask, horizontal_dict = self.generate_hough_line_label(line_mask)
        # return final_mask, horizontal_dict, rho, theta, x2, line_mask

    def propagate_vertical_rows(self, line_mask, x1, x_old, rho_old, theta_old):
        if x_old != []:
            for it in range(len(x_old)):
                if x_old[it] == -1 or abs(x_old[it] - x1) < 200:
                    continue
                x_new = int(
                    (rho_old[it] - line_mask.shape[0] * (it + 2) * np.sin(theta_old)) / np.cos(theta_old)
                    + (150 + (len(x_old) - it) * 25)
                )
                line_mask = cv2.line(
                    line_mask,
                    (x_old[it] + (150 + (len(x_old) - it) * 25), 0),
                    (x_new, line_mask.shape[1]),
                    (255, 255, 255),
                    15,
                )
        return line_mask

    def generate_hough_line_label(self, mask):  # , real_mask):
        # mask has one line that is my "row crop line"
        # we need to define which pixels in this line are vegetation using self.binary_mask
        line_crop = self.binary_mask * mask
        line_soil = ~self.binary_mask

        # create a label mask where we will store the labels
        label_mask = np.ones_like(mask) * (3)  # 3 is the ignore index
        veg_components = cv2.connectedComponentsWithStats(self.binary_mask)

        # identify crops as veg components intersecting the line
        for _id in range(1, veg_components[0]):
            if (mask * (veg_components[1] == _id)).sum() != 0:
                label_mask[(veg_components[1] == _id)] = 1

        label_mask[(line_soil == 255)] = 0  # soil
        label_mask, horizontal_dict = self.check_horizontal(label_mask)
        return label_mask, horizontal_dict

    def check_horizontal(self, mask):
        return mask, {
            "start": mask[-int(mask.shape[0] / 2), :].argmax(),
            "end": np.flip(mask[-int(mask.shape[0] / 2), :]).argmax(),
            "size": mask[-int(mask.shape[0] / 2), :].sum(),
        }

    def check_vertical(self, mask):
        # For horizontal rows, we check a vertical slice through the middle
        middle_col = int(mask.shape[1] / 2)  # Middle column instead of middle row
        vertical_slice = mask[:, middle_col]  # Vertical slice instead of horizontal
        
        return mask, {
            "start": vertical_slice.argmax(),  # First non-zero element from top
            "end": np.flip(vertical_slice).argmax(),  # First non-zero element from bottom (flipped)
            "size": vertical_slice.sum(),  # Total sum of vertical slice
        }

    def weakly_supervised_mask(self, name):
        mask = np.array(Image.open(name))
        self.binary_mask = (mask != 0).astype(np.uint8)
        return mask

    def extract_vegetation_mask(self, image):
        # Convert the RGB image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # --- Define a robust range for the color green in HSV ---
        # These values typically work well for a wide range of lighting conditions
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create a mask where green pixels are white (1) and others are black (0)
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # --- Use Morphological Operations to clean the mask ---
        # This is crucial for connecting broken plant parts and removing noise
        kernel = np.ones((5, 5), np.uint8)
        
        # Closing fills small holes within plant blobs
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Opening removes small, isolated noise pixels
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        self.binary_mask = mask_opened.astype(np.uint8)
        # Ensure the final mask is strictly 0 or 1
        self.binary_mask[self.binary_mask > 0] = 1

    def propagate_horizontal_rows(self, line_mask, y1, y_old, rho_old, theta_old):
        if y_old != []:
            for it in range(len(y_old)):
                # Skip if no previous line detected or lines are too close
                if y_old[it] == -1 or abs(y_old[it] - y1) < 200:
                    continue
                
                # Calculate new x-coordinate for the propagated horizontal line
                # For horizontal lines: x = (rho - y * sin(theta)) / cos(theta)
                y_new = int(
                    (rho_old[it] - line_mask.shape[1] * (it + 2) * np.cos(theta_old)) / np.sin(theta_old)
                    + (150 + (len(y_old) - it) * 25)
                )
                
                # Draw horizontal line from left to right edge of image
                line_mask = cv2.line(
                    line_mask,
                    (0, y_old[it] + (150 + (len(y_old) - it) * 25)),  # Start point (left edge)
                    (line_mask.shape[1], y_new),  # End point (right edge)
                    (255, 255, 255),
                    15,
                )
        return line_mask