# Visual Reasoning Skill Evaluation on PaintSkills

<img src="../assets/skills.png" alt="teaser image" width="1200"/>

## Dataset Setup

* Create `$paintskills_dir` directory.
* From [the Google Drive link](https://drive.google.com/drive/folders/1Bza2zyvHLvComohZ9PAGyykY7sm7JoIH), download `metadata.json` and three skill data: `object.zip`, `count.zip`, `spatial.zip`.
* Unzip the zipfiles inside `$paintskills_dir`.
```bash
cd $paintskills_dir
unzip object.zip
unzip count.zip
unzip spatial.zip
```


* The `$paintskills_dir` directory has hierarchy as below:
```bash
$paintskills_dir/
    # skill name (i.e.., object, count, and spatial)
    {skill}/

        # Images
        images/

        # Scene configuration
        scenes/
            {skill}_train.json
            {skill}_val.json

        # Bounding box annotations - only needed for DETR
        {skill}_train_bounding_boxes.json
        {skill}_val_bounding_boxes.json

    # metadata for all skills.
    metadata.json
```

## (Optional) 3D simulator

Please see https://github.com/aszala/PaintSkills-Simulator for our 3D Simulator implementation.

<img src="../assets/dataset_generation.png" alt="teaser image" width="500"/>

## Evaluation on GT images

```bash
bash scripts/evaluate_skill.sh \
    --skill_name object --gt_data_eval --paintskills_dir $paintskills_dir
bash scripts/evaluate_skill.sh \
    --skill_name count --gt_data_eval --paintskills_dir $paintskills_dir
bash scripts/evaluate_skill.sh \
    --skill_name spatial --gt_data_eval --paintskills_dir $paintskills_dir
```

## Download pretrained DETR checkpoints

Download checkpoint for each skill from [the Google Drive link](https://drive.google.com/drive/folders/1qZr0biroWR5WV6wXD0BMZHiO1sgsqlCD) at `detr/output/$skill/checkpoint.pth`.

## Evaluation of Text2Img models with DETR

Coming soon.