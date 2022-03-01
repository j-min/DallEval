# Social Bias Evaluation

We provide the scripts for CLIP-based social bias evaluation. This is done in two steps.

* First, we generate images with a text-to-image generation model from a set of gender/race `neutral prompts`.
* Then, we classify the generated images with a set of gender/race `classifier prompts`.
Prompt files are located in the `./prompts/` folder.


## Prompts

For `neutral prompts`, we provide four files:
```bash
# Total 145 prompts
prompts/neutral_prompts_profession.json   # 85 prompts
prompts/neutral_prompts_objects.json      # 39 prompts
prompts/neutral_prompts_politics.json     # 6 prompts
prompts/neutral_prompts_others.json       # 15 prompts
```
Each file has following structure:
```json
{
    "neutral_prompts": [
        ...
    ],
​}
```

For `classifier prompts`, we provide two files:
```bash
prompts/classifier_prompts_gender.json # 2 prompts
prompts/classifier_prompts_race.json   # 4 prompts
```
Each file has following structure:
```json
{
    "classifier_prompts": [
        ...
    ],

}
```

## CLIP-based evaluation

### 0) Generate images from prompts
In `image_dir` directory, generate 9 images for a neutral prompt with name format: `image_dir/{prompt}/{0-9}.jpg`.
We provide [`./test_images/`](./test_images/) for example.

### 1) Classify gender/race category with CLIP.

```bash
python score_clip.py --image_dir image_dir --out bias_scores_clip.json
```

For [`./test_images/`](./test_images/), you can calculate score with
```bash
python score_clip.py --test
```


### 2) Calculate statistics from the CLIP prediction results

```bash
python calculate_variance.py --score_path bias_scores_clip.json
```

The result would be similar to:
```bash
╰─ python calculate_variance.py --score_path bias_scores_clip.json
Gender Bias
male_votes: 90
female_votes: 55
total: 145
N_category: 2
{'STD': 0.12068965517241381, 'MAD': 0.12068965517241381}
Oracle - uniform
N_category: 2
{'STD': 0.0, 'MAD': 0.0}
Worst - one-hot
N_category: 2
{'STD': 0.5, 'MAD': 0.5}

Race Bias
white votes: 20
black votes: 14
hispanic votes: 85
asian votes: 26
total: 145
N_category: 4
{'STD': 0.19630201924234167, 'MAD': 0.16810344827586204}
Oracle - uniform
N_category: 4
{'STD': 0.0, 'MAD': 0.0}
Worst - one-hot
N_category: 4
{'STD': 0.4330127018922193, 'MAD': 0.375}
```