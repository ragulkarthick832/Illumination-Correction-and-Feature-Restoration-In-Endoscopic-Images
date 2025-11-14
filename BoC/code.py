import cv2
import numpy as np
import os
import random
#edit from lab
def generate_composite_mask(height, width, scenario='radial', center=None, corner=0, direction='left-to-right',
                            sharpness_params={'gaussian': 0.4, 'sigmoid': 0.2, 'power': 0.5},
                            weights={'gaussian': 0.34, 'sigmoid': 0.33, 'power': 0.33},
                            random_focus=True):
    
    y, x = np.mgrid[0:height, 0:width] #first it creates a grid for the mask

    if scenario == 'radial':
        if center is None:
            if random_focus:
                center = (random.randint(0, width - 1), random.randint(0, height - 1))
            else:
                center = (width // 2, height // 2)
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2) # finds the distance of all pixels from the ligth source

    elif scenario == 'corner':
        corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        if random_focus:
            corner = random.randint(0, 3)
        corner_x, corner_y = corners[corner]
        if random_focus:
            jitter_x = random.randint(-width // 10, width // 10)
            jitter_y = random.randint(-height // 10, height // 10)
            corner_x = np.clip(corner_x + jitter_x, 0, width - 1)
            corner_y = np.clip(corner_y + jitter_y, 0, height - 1)
        dist = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)

    elif scenario == 'linear':
        if random_focus:
            direction = random.choice(['left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top'])
        if direction == 'left-to-right':
            dist = x
        elif direction == 'right-to-left':
            dist = width - 1 - x
        elif direction == 'top-to-bottom':
            dist = y
        else:
            dist = height - 1 - y
    else:
        raise ValueError("Unknown scenario.")

    dist_normalized = dist / np.max(dist) if np.max(dist) != 0 else np.zeros_like(dist)

    s_g = max(sharpness_params['gaussian'], 1e-5)
    mask_g = np.exp(-dist_normalized**2 / (2 * s_g**2))

    s_s = 1 / max(sharpness_params['sigmoid'], 1e-5)
    mask_s = 1 - (1 / (1 + np.exp(-s_s * (dist_normalized - 0.5))))

    s_p = 2 / max(sharpness_params['power'], 1e-5)
    mask_p = 1 / (1 + (dist_normalized * s_p)**2)

    total_weight = sum(weights.values())
    composite_mask = ((weights['gaussian'] * mask_g) +
                      (weights['sigmoid'] * mask_s) +
                      (weights['power'] * mask_p)) / total_weight

    min_val, max_val = np.min(composite_mask), np.max(composite_mask)
    composite_mask = (composite_mask - min_val) / (max_val - min_val + 1e-8)

    # nonlinear boost
    composite_mask = composite_mask ** 1.2
    composite_mask = 0.2 + 1.8 * composite_mask
    composite_mask = np.clip(composite_mask, 0, 2.0)

    return composite_mask


def apply_lighting_effect(image, mask, gamma=0.7):
    mask_3d = mask[:, :, np.newaxis]
    image_float = image.astype(np.float32) / 255.0
    lit_image = np.clip(image_float * (mask_3d ** 1.2), 0, 1)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    lit_image_uint8 = np.clip(lit_image * 255, 0, 255).astype("uint8")
    gamma_corrected_image = cv2.LUT(lit_image_uint8, table)
    return gamma_corrected_image


# --- Main pipeline ---
input_folder = "../Dataset/images"
output_base = "../Dataset/output"
os.makedirs(output_base, exist_ok=True)

scenarios = ["radial", "corner", "linear"]
for scenario in scenarios:
    os.makedirs(os.path.join(output_base, scenario), exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
image_files = sorted(image_files)  # process all images

for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Could not read {filename}, skipping...")
        continue
    H, W, _ = image.shape

    base_name, ext = os.path.splitext(filename)
    for scenario in scenarios:
        mask = generate_composite_mask(H, W, scenario=scenario)
        output_img = apply_lighting_effect(image, mask, gamma=0.7)

        out_path = os.path.join(output_base, scenario, f"{base_name}_{scenario}.jpg")
        cv2.imwrite(out_path, output_img)
        print(f"✅ Saved {out_path}")

















































