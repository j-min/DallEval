import json
import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

import clip


def clip_forward(model, image, text):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    return logits_per_image, logits_per_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='./test_images', help="directory of generated images")
    parser.add_argument('--out', type=str, default='./bias_scores_clip.json', help="directory of save CLIP-based prediction results")

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

    classifier_prompts = []
    for prompt_type in ["gender", "race"]:
        with open(f'prompts/classifier_prompts_{prompt_type}.json', 'r') as f:
            classifier_prompts += json.load(f)["classifier_prompts"]
    print(f"Loaded {len(classifier_prompts)} classifier prompts")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f'Loaded CLIP at {device}')

    gender_classifier_prompts = classifier_prompts[:2]
    race_classifier_prompts = classifier_prompts[2:]

    gender_classifier_prompts = clip.tokenize(gender_classifier_prompts).to(device)
    race_classifier_prompts = clip.tokenize(race_classifier_prompts).to(device)

    prompt2scores = {}

    for neutral_prompt in tqdm(neutral_prompts):
        prompt2scores[neutral_prompt] = []

        prompt_generated_images_dir = Path(image_dir).joinpath(
            neutral_prompt
        )

        img_path_list = list(prompt_generated_images_dir.glob('*.jpg'))

        for img_path in img_path_list:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits_per_image, logits_per_text = clip_forward(model, image, gender_classifier_prompts)
                logits_per_image2, logits_per_text2 = clip_forward(model, image, race_classifier_prompts)

                classifier_probs_gender = logits_per_image.softmax(dim=-1).cpu().numpy()
                classifier_probs_race = logits_per_image2.softmax(dim=-1).cpu().numpy()

            classifier_probs = np.append(
                classifier_probs_gender[0],
                classifier_probs_race[0]
            )
            prompt2scores[neutral_prompt].append(classifier_probs)

        n_images_per_prompt = len(img_path_list)
        assert n_images_per_prompt == 9, (neutral_prompt, n_images_per_prompt)

    # merge classifier probs
    for neutral_prompt in neutral_prompts:
        classifier_probs = np.zeros(shape=(n_images_per_prompt, len(classifier_prompts)))

        for i, ith_classifier_probs in enumerate(prompt2scores[neutral_prompt]):
            classifier_probs[i] = ith_classifier_probs

        classifier_probs = classifier_probs.mean(axis=0)
        assert classifier_probs.shape == (len(classifier_prompts),)

        prompt2scores[neutral_prompt] = classifier_probs.tolist()


    output = {
        "results": prompt2scores,
        "classifier_prompts": classifier_prompts
    }

    with open(f"{args.out}", 'w') as f:
        json.dump(output, f)
