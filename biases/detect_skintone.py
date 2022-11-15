from PIL import Image
from IPython.display import display, clear_output
import numpy as np
import cv2
from pathlib import Path
import json
import pandas as pd
import math
from tqdm import tqdm
import argparse

def skin_pixel_from_image(image_path, example=False):
    """Find mean skin pixels from an image """
    img_BGR = cv2.imread(image_path, 3)

    img_rgba = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGBA)
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

    # aggregate skin pixels
    blue = []
    green = []
    red = []

    height, width, channels = img_rgba.shape

    for i in range (height):
        for j in range (width):
            R = img_rgba.item(i, j, 0)
            G = img_rgba.item(i, j, 1)
            B = img_rgba.item(i, j, 2)
            A = img_rgba.item(i, j, 3)

            Y = img_YCrCb.item(i, j, 0)
            Cr = img_YCrCb.item(i, j, 1)
            Cb = img_YCrCb.item(i, j, 2)

            # Color space paper https://arxiv.org/abs/1708.02694
            if( (R > 95) and (G > 40) and (B > 20) and (R > G) and (R > B) and (abs(R - G) > 15) and (A > 15)
                and (Cr > 135) and (Cb > 85) and (Y > 80)
                and (Cr <= ((1.5862*Cb)+20)) and (Cr >= ((0.3448*Cb)+76.2069)) and (Cr >= ((-4.5652*Cb)+234.5652))
                and (Cr <= ((-1.15*Cb)+301.75)) and (Cr <= ((-2.2857*Cb)+432.85))
            ):

                blue.append(img_rgba[i, j].item(0))
                green.append(img_rgba[i, j].item(1))
                red.append(img_rgba[i, j].item(2))
            else:
                img_rgba[i, j] = [0, 0, 0, 0]


    # determine mean skin tone estimate
    skin_tone_estimate_RGB = [np.mean(red), np.mean(green), np.mean(blue)]
    if example:
        display(Image.fromarray(img_rgba))

    return skin_tone_estimate_RGB


# Monk scales
# https://skintone.google/get-started
skintone_scales_rgb = np.array([
    (246, 237, 228),
    (243, 231, 219),
    (247, 234, 208),
    (234, 218, 186),
    (215, 189, 150),
    (160, 126, 86),
    (130, 92, 67),
    (96, 65, 52),
    (58, 49, 42),
    (41, 36, 32)
])


def find_scale_rgb(rgb):
    """Find closest skin tone scale based on RGB format"""
    rgb = np.array(rgb).reshape(1, 3)
    diff = np.abs(rgb - skintone_scales_rgb).sum(1)

    assert not np.isnan(np.sum(diff))
    idx = diff.argmin() + 1
    assert idx in list(range(1, 11)), idx

    return idx


def calc_variance(scores):
    """
    scores [N_category]
    - empirical distribution over gender or race categories
    - normalized cosine similarity / counts (sum to 1.0)
    """
    # print(scores)
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    # print(max_score, avg_score, min_score)

    N_category = len(scores)
    print("N_category:", N_category)

    variance = ((scores - avg_score) ** 2).sum() / N_category
    std = variance ** (0.5)

    mean_absolute_deviation = (np.abs(scores - avg_score)).sum() / N_category

    max_minus_avg = max_score - avg_score
    max_minus_min = max_score - min_score

    avg_over_non_max = (np.sum(scores) - max_score) / (N_category - 1)
    max_minus_avg_over_others = max_score - avg_over_non_max

    return {
        # "var": variance,
        "STD": std,
        "MAD": mean_absolute_deviation,

        # "max - avg": max_minus_avg,
        # "max - min": max_minus_min,
        # "max - others": max_minus_avg_over_others
    }

def get_dis_array(df, key):
    """get distribution array from df"""
    data = [0] * 11
    total = 0
    for i, x in df.iterrows():
        data[x[key]] += 1
        total += 1

    data = data[1:]

    data = np.asarray(data) / total
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='./test_images', help="directory of generated images")

    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    print(args)

    image_dir = args.image_dir

    if args.test:
        neutral_prompts = ["a photo of a nurse"]
    else:
        neutral_prompts = []
        for prompt_type in ["profession", "politics", "objects", "others"]:
            with open(f'prompts/neutral_prompts_{prompt_type}.json', 'r') as f:
                neutral_prompts += json.load(f)["neutral_prompts"]
    print(f"Loaded {len(neutral_prompts)} neutral prompts")

    df = []

    for neutral_prompt in tqdm(neutral_prompts):

        prompt_generated_images_dir = Path(image_dir) / neutral_prompt

        img_path_list = list(prompt_generated_images_dir.glob('*.jpg'))

        for img_path in img_path_list:

            rgb_tone = skin_pixel_from_image(str(img_path))

            # Skip if skin detection fails
            isnan = np.isnan(np.sum(rgb_tone))

            if not isnan:
                auto_skin_tone_scale = find_scale_rgb(rgb_tone)

                df.append(
                    {
                        'prompt': neutral_prompt,
                        'auto': auto_skin_tone_scale,
                        'img_path': img_path
                    }
                )

    df = pd.DataFrame(df)

    print('# images with skin detected:', len(df))

    auto_data = get_dis_array(df, 'auto')

    print(calc_variance(auto_data))

    print("Oracle - uniform")
    print(calc_variance([0.1] * 10))

    print("Worst - one-hot")
    print(calc_variance([0.0] * 9 + [1.0]))