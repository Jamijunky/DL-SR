import os
import glob
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf

from models.DFCAN16 import DFCAN


# ---------------- CONFIG ----------------
input_wf_dir = "../dataset/test/F-actin/input_wide_field_images"
input_sim_dir = "../dataset/test/F-actin/input_raw_sim_images"

model_weights_wf = "../trained_models/DFCAN-SISR_F-actin/weights.best"
model_weights_sim = "../trained_models/DFCAN-SIM_F-actin/weights.best"

input_height = 502
input_width = 502
n_channel = 1
scale_factor = 2
# ---------------------------------------


def load_images_from_dir(input_dir):
    img_paths = glob.glob(os.path.join(input_dir, "*"))
    img_paths = [p for p in img_paths if os.path.isfile(p)]

    images = []
    for p in img_paths:
        img = imageio.imread(p).astype(np.float32)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        images.append(img)

    return images


def run_inference(input_dir, weights_path):
    print("Processing", input_dir)

    model = DFCAN((input_height, input_width, n_channel), scale=scale_factor)

    try:
        model.load_weights(weights_path)
    except Exception as e:
        print("WARNING: could not load weights:", e)

    images = load_images_from_dir(input_dir)

    for img in images:
        img = np.expand_dims(img, axis=0)
        _ = model.predict(img)


# ---------------- RUN ----------------
run_inference(input_wf_dir, model_weights_wf)
run_inference(input_sim_dir, model_weights_sim)
