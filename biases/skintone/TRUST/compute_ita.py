import cv2
import numpy as np
from glob import glob # type: ignore
from tqdm import tqdm # type: ignore
from multiprocessing import Pool # type: ignore
import json
import os

def ita(l,a,b):
    return (np.arctan((l - 50) / b) * 180) / np.pi

def compute_ita(img_path):
    img = cv2.imread(img_path)
    mask = cv2.imread("skin_for_ita_mask_cheeks.png")
    mask = cv2.resize(mask, img.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    
    masked = cv2.bitwise_and(img, img, mask=mask[:,:,0])

    masked = cv2.cvtColor(masked, cv2.COLOR_RGB2LAB)

    width, height, _ = masked.shape

    skin_tones = []

    for x in range(width):
        for y in range(height):
            l,a,b = masked[y,x]

            if l == 0 and a == 128 and b == 128:
                continue
            else:
                skin_tones.append((l,a,b))

    itas = [ ita(l,a,b) for l,a,b in skin_tones ]
    ita_score = np.mean(itas)

    return ita_score

monk_tones = [
    (94.211, 1.503, 5.422),
    (92.275, 2.061, 7.28),
    (93.091, 0.216, 14.205),
    (87.573, 0.459, 17.748),
    (77.902, 3.471, 23.136),
    (55.142, 7.783, 26.74),
    (42.47, 12.325, 20.53),
    (30.678, 11.667, 13.335),
    (21.069, 2.69, 5.964),
    (14.61, 1.482, 3.525),
]
monk_itas = [ ita(l,a,b) for l,a,b in monk_tones ]

def match_monk_tone(ita):
    min_dist = 10000
    best_monk_ita = None

    for i, monk_ita in enumerate(monk_itas):
        dist = abs(monk_ita - ita)

        if dist < min_dist:
            min_dist = dist
            best_monk_ita = i

    if best_monk_ita is None:
        raise Exception("No best monk ita found")

    return best_monk_ita

def score_image(img_path):
    ita_score = compute_ita(img_path)
    monk_tone = match_monk_tone(ita_score) + 1
    
    words = img_path.split("/")[-2].split("_")
    prompt = " ".join(words[0:-1])
    index = int(words[-1])

    return (monk_tone, prompt, index)

if __name__ == "__main__":
    models = [ "sd", "karlo", "mindalle" ]

    with open("../../prompt_list.json", 'r') as f:
        prompts = json.load(f)

    for model in tqdm(models, colour="green"):
        output = {}

        for prompt in prompts:
            output[prompt] = [ ]

            for i in range(9):
                output[prompt].append([])

        all_images = glob(f"./outputs/albedos/benchmark_split_{model}/*/*.jpg")
        all_images = [ img for img in all_images if "test_images_vis" not in img ]
        
        results = []
        with Pool(40) as p:
            results = list(tqdm(p.map(score_image, all_images), total=len(all_images)))


        for monk_tone, prompt, index in results:
            output[prompt][index].append(monk_tone)

        for prompt in tqdm(prompts):
            for i in range(9):
                img_folder = f"./outputs/albedos/benchmark_split_{model}/{prompt.replace(' ', '_')}_{i}/"

                if not os.path.exists(img_folder):
                    continue

                imgs = glob(f"{img_folder}/*.jpg")
                
                itas = []
                for img in imgs:
                    ita_score = compute_ita(img)
                    itas.append(ita_score)

                ita_score = np.mean(itas)

                monk_tone = match_monk_tone(ita_score)

                output[prompt][i] = monk_tone

        with open(f"detected_skintones_means_{model}.json", 'w') as f:
            json.dump(output, f, indent=2)