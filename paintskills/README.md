# Visual Reasoning Skill Evaluation on PaintSkills

<img src="../assets/skills.png" alt="teaser image" width="1200"/>

## Dataset Setup

1) Create `$paintskills_dir` directory.
2) From [the Google Drive link](https://drive.google.com/drive/folders/1xwD0Yv5iS-OndP8XxDwIhzaLA8IR582e), download `metadata.json` and three skill directories: `object/`, `count/`, `spatial/`, inside `$paintskills_dir`.


* The `$paintskills_dir` directory has hierarchy as below:
```bash
$paintskills_dir/
    # skill name (i.e., object, count, and spatial)
    {skill}/

        # Scene configuration
        scenes/
            {skill}_train.json
            {skill}_val.json

        # GT Images (from {skill}/images.zip)
        images/

        # Bounding box annotations (for DETR finetuning)
        {skill}_train_bounding_boxes.json
        {skill}_val_bounding_boxes.json

    # metadata for all skills.
    metadata.json
```

## Scene Configuration

The scene configuration files (`scenes/{skill}_{split}.json`) have the following structure, where `skill` is one of `object`, `count`, `spatial`, and `split` is one of `train`, `val`.

e.g., `count_val.json`
```json
{
    "data": [
        {
            "id": "count_val_00000",
            "scene": "HDR-KirbyCove",
            "text": "1 person",
            "skill": "count",
            "split": "val",
            "objects": [
                {
                    "id": 0,
                    "shape": "humanJosh",
                    "coconame": "person",
                    "color": "plain",
                    "relation": null,
                    "scale": 14.114588410729079,
                    "texture": "plain",
                    "rotation": null,
                    "state": "sitting"
                }
            ]
        },
        ...
    ]
}
```

## Evaluation of Text2Img models with DETR

1) Generate the skill-specific images in $image_dir from captions (`text` field in the scene data) with your text-to-image generation models (finetuned on PaintSkills). The evaluation scripts expects that the generated images have filenames in the format of `image_{datum['id']}.png`. For example, if the datum['id'] is `count_val_00000`, the filename should be `image_count_val_00000.png`. 


1) Run the evaluation script

```bash
skill='object' # switch to other skills (choices=['object', 'count', 'spatial'])
image_dir='/path/to/generated/images'
bash scripts/evaluate_skill_FT_DETR-R101-DC5.sh \
    --skill_name $skill \
    --paintskills_dir $paintskills_dir \
    --image_dir $image_dir \
```

## (Optional) 3D simulator

Please see https://github.com/aszala/PaintSkills-Simulator for our 3D Simulator implementation.

<img src="../assets/dataset_generation.png" alt="teaser image" width="500"/>

## (Optional) Evaluation on GT images

```bash
skill='object' # count, spatial
bash scripts/evaluate_skill_FT_DETR-R101-DC5.sh \
    --skill_name $skill \
    --gt_data_eval \
    --paintskills_dir $paintskills_dir
```

