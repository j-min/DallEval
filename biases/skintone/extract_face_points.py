import face_alignment
from skimage import io
from glob import glob # type: ignore
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, default="./output/")
    parser.add_argument("--image_folder", type=str, default="./images/")
    parser.add_argument("--prompt_list", type=str, default="../prompt_list.json")

    args = parser.parse_args()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")

    images = glob(f'{args.image_folder}/*/*.png')

    with open(args.prompt_list, 'r') as f:
        prompts = json.load(f)

    image_lines = []

    for image in tqdm(images):
        im = Image.open(image)
        input = io.imread(image)
        preds = fa.get_landmarks(input)
        
        prompt = image.split('/')[-2]
        image_name = image.split('/')[-1].replace('.png', '')

        if prompt.replace("_", " ") not in prompts:
            continue

        folder_name = f"{prompt}_{image_name}"
        
        if preds is None:
            continue
        
        os.makedirs(f"{args.output_folder}/crops-lmks/{folder_name}", exist_ok=True)
        os.makedirs(f"{args.output_folder}/crops/{folder_name}", exist_ok=True)
        os.makedirs(f"{args.output_folder}/full_image/", exist_ok=True)
        
        im.save(f"{args.output_folder}/full_image/{folder_name}.png")

        for i, pred in enumerate(preds):
            im.crop((pred[:, 0].min(), pred[:, 1].min(), pred[:, 0].max(), pred[:, 1].max())).save(f"{args.output_folder}/crops/{folder_name}/{i}.png")
            image_lines.append(f"{args.output_folder}/crops/{folder_name}/{i}.png")
            np.save(f"{args.output_folder}/crops-lmks/{folder_name}/{i}.npy", pred)


    with open(f"{args.output_folder}/crop_files.txt", 'w') as f:
        f.write('\n'.join(image_lines))