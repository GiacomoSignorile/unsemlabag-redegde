import cv2
import numpy as np
from PIL import Image


class Hough:
    def __init__(self, pixel_res=1, angle_res=np.pi / 180, min_theta=-0.02, max_theta=0.02, min_voting=600):
        self.pixel_res = pixel_res
        self.angle_res = angle_res
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_voting = min_voting

    def forward(self, image, rho_old=[], theta_old=-1, x_old=[], horizontal_exceed={}):
        self.extract_vegetation_mask(image)

        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
    
        if self.debug_counter < 5: # Save first 5 patches
            cv2.imwrite(f"debug_patch_{self.debug_counter:03d}_input.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"debug_patch_{self.debug_counter:03d}_binary_mask.png", self.binary_mask * 255)
        # END DEBUG
        if self.binary_mask.sum() == 0:
            return self.binary_mask, horizontal_exceed, -1, -1, -1, -1
        try:
            rho, theta = cv2.HoughLines(
                self.binary_mask,
                self.pixel_res,
                self.angle_res,
                self.min_voting,
                min_theta=self.min_theta,
                max_theta=self.max_theta,
            )[0][0]
        except:
            try:
                rho, theta = cv2.HoughLines(
                    self.binary_mask,
                    self.pixel_res,
                    self.angle_res,
                    int(self.min_voting / 10),
                    min_theta=self.min_theta,
                    max_theta=self.max_theta,
                )[0][0]
            except:
                # no line has been found
                return np.zeros_like(image, dtype=np.uint8)[:, :, 0], horizontal_exceed, -1, -1, -1, -1

        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)
        y1 = int(0)
        x1 = int(rho / np.cos(theta))
        y2 = int(image.shape[1])
        x2 = int((rho - y2 * np.sin(theta)) / np.cos(theta))

        line_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 15)

        # include lines propagated from the y axis
        line_mask = self.propagate_vertical_rows(line_mask, x1, x_old, rho_old, theta_old)

        scaled_x_offset = int(round(5 * (image.shape[0] / 1024.0))) if image.shape[0] != 1024 else 5 # Dynamic scaling for '5'
        scaled_x_offset = max(1, scaled_x_offset) # Ensure at least 1
        # include lines propagated from the x axis
        line_mask = cv2.line(
            line_mask,
            (5, horizontal_exceed["start"]),
            (5, image.shape[1] - horizontal_exceed["end"]),
            (255, 255, 255),
            15,
        )

        if self.debug_counter < 5: # Match the counter for binary_mask
            # line_mask is likely (H, W, 3) and uint8 at this point
            cv2.imwrite(f"debug_patch_{self.debug_counter:03d}_hough_lines.png", line_mask)
        line_mask = line_mask[:, :, 0].astype(bool)
        final_mask, horizontal_dict = self.generate_hough_line_label(line_mask)
        return final_mask, horizontal_dict, rho, theta, x2, line_mask

    def propagate_vertical_rows(self, line_mask, x1, x_old, rho_old, theta_old):
        if x_old != []:
            # scale_factor = line_mask.shape[0] / 1024.0 # line_mask has same shape as image patch
            # scaled_offset_base = int(round(150 * scale_factor))
            # scaled_offset_increment = int(round(25 * scale_factor))
            # # Ensure they don't become too small, e.g., min 1
            # scaled_offset_base = max(1, scaled_offset_base)
            # scaled_offset_increment = max(1, scaled_offset_increment)
            for it in range(len(x_old)):
                if x_old[it] == -1 or abs(x_old[it] - x1) < 200: # <--- REVERTED
                    continue

                current_offset = 150 + (len(x_old) - it) * 25 # <--- REVERTED
                
                x_new = int(
                    (rho_old[it] - line_mask.shape[0] * (it + 2) * np.sin(theta_old)) / np.cos(theta_old)
                    + current_offset # Using scaled offset
                )
                line_mask = cv2.line(
                    line_mask,
                    (x_old[it] + current_offset, 0), # Using scaled offset
                    (x_new, line_mask.shape[1]),
                    (255, 255, 255),
                    15, # <--- SCALED THICKNESS
                )
        return line_mask

    def generate_hough_line_label(self, mask):  # , real_mask):
        # mask has one line that is my "row crop line"
        # we need to define which pixels in this line are vegetation using self.binary_mask
        line_crop = self.binary_mask * mask
        line_soil = ~self.binary_mask
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

    def weakly_supervised_mask(self, name):
        mask = np.array(Image.open(name))
        self.binary_mask = (mask != 0).astype(np.uint8)
        return mask

    def extract_vegetation_mask(self, image):
        # paper method
        # cv2.imwrite('geometric_segmentation/fig.pnm', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # os.system("cd geometric_segmentation; ./segment 1 500 50 fig.pnm fig.pnm" )
        # self.binary_mask = np.array(Image.open("geometric_segmentation/fig.pnm")).sum(-1)
        # colors = np.unique(self.binary_mask)
        # for color in colors:
        #    current = image[ self.binary_mask == color ].sum(0)/(self.binary_mask == color).sum()
        #    if (2*current[1] - current[0] - current[2] < 100) and (2*current[1] - current[0] - current[2] > 35):
        #        self.binary_mask[ self.binary_mask == color ] = 0

        # faster but requires a lot of manual finetuning
        # r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        # r = (r - r.mean()) / (r.std() + 1e-15)
        # g = (g - g.mean()) / (g.std() + 1e-15)
        # b = (b - b.mean()) / (b.std() + 1e-15)
        # exg = 2 * g - r - b
        # self.binary_mask = (exg > 0.3) * (r < g) * (b < g) * (r * 0.5 > b)
        # comp = cv2.connectedComponentsWithStats(self.binary_mask.astype(np.uint8))
        # for num in range(comp[0]):
        #     if comp[2][num][-1] < 11:  # fitler our very small components, usually this veg mask has noise
        #         self.binary_mask[comp[1] == num] = 0

        # self.binary_mask[self.binary_mask != 0] = 1
        # self.binary_mask = self.binary_mask.astype(np.uint8)
        # New method
        r_ch, g_ch, b_ch = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        valid_data_mask = (image.sum(axis=2) > 15) # Sum of RGB > 15 (threshold for "not black")
                                               # Using 15 instead of 10 for a bit more margin.

        if not np.any(valid_data_mask):
            self.binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            return

        r_mean_valid = r_ch[valid_data_mask].mean()
        r_std_valid = r_ch[valid_data_mask].std() + 1e-15
        g_mean_valid = g_ch[valid_data_mask].mean()
        g_std_valid = g_ch[valid_data_mask].std() + 1e-15
        b_mean_valid = b_ch[valid_data_mask].mean()
        b_std_valid = b_ch[valid_data_mask].std() + 1e-15

        r_norm = (r_ch.astype(np.float32) - r_mean_valid) / r_std_valid
        g_norm = (g_ch.astype(np.float32) - g_mean_valid) / g_std_valid
        b_norm = (b_ch.astype(np.float32) - b_mean_valid) / b_std_valid
    
        exg_value = 2 * g_norm - r_norm - b_norm
        exg_threshold = -0.05 # Start with a lenient threshold
        self.binary_mask = (exg_value > exg_threshold) & valid_data_mask
    
        # Convert to uint8 for morphological operations
        binary_mask_uint8_for_morph = self.binary_mask.astype(np.uint8) * 255 # 0 or 255

        # Define a kernel (structuring element)
        # Kernel size depends on the scale of noise vs. features in your PATCH
        # For a 341x341 patch, start with a small kernel. For 1024x1024, maybe larger.
        # Let's assume patch size is around 341x341 (scaled from 1024 with factor ~1/3)
        # Original kernel for 1024px might be 5x5 or 7x7. Scaled: 3x3 or 2x2.
        kernel_size = 3 # Try 3 or 5 for 341px patch; try 5 or 7 or 9 for 1024px patch
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply Opening to remove small white noise specks
        opened_mask = cv2.morphologyEx(binary_mask_uint8_for_morph, cv2.MORPH_OPEN, kernel, iterations=1)

        # Apply Closing to fill small holes in vegetation and connect nearby parts
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # You can try more iterations

        # Update self.binary_mask with the cleaned version (back to boolean or keep as uint8 0/1 for next step)
        self.binary_mask = (closed_mask > 0) # Back to boolean for the existing area filter logic

        # --- Now apply your existing area filter ---
        area_filter_threshold = 100 # Your scaled value for 341x341 patches

        # Convert boolean mask to uint8 for connected components
        binary_mask_uint8_for_cc = self.binary_mask.astype(np.uint8)
        if binary_mask_uint8_for_cc.sum() > 0:
            comp = cv2.connectedComponentsWithStats(binary_mask_uint8_for_cc)
            for num in range(comp[0]):
                if num == 0: continue
                if comp[2][num][-1] < area_filter_threshold:
                    self.binary_mask[comp[1] == num] = False

        self.binary_mask = self.binary_mask.astype(np.uint8) # Final 0/1 uint8 mask
