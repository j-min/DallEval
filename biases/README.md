# Social Bias Evaluation

We provide the scripts for automated social bias evaluation. This is done in two steps.

<ol>
    <li>We generate images with a text-to-image generation model from a set of prompts.</li>
    <li>We detect gender, attibutes, and skin tone in the generated images with BLIP-2 and TRUST.</li>
</ol>

Prompts are provided in `prompt_list.json`.
Note: not all prompts are used for all tasks. In the gender detection task, we only use neutral prompts like `a person ...`.

## Image Folder Setup
You image folder should have a structure similar to this in order for the scripts to run correctly:
```
/
-- this_is_a_prompt/
---- 0.png
---- 1.png
---- ...
---- 8.png
-- this_is_another_prompt/
-- ...
```

## Gender and Attribute Detection
First make sure you have salesforce-lavis installed with:
```bash
pip install salesforce-lavis
```

Then run:
```bash
cd BLIP2
```

### Gender Detection
```bash
bash evaluate_social_bias_with_blip2.sh --image_dir {path_to_your_model_images} --metadata_file ../prompt_list.json --task gender
```

### Attribute Detection
```bash
bash evaluate_social_bias_with_blip2.sh --image_dir {path_to_your_model_images} --metadata_file ../prompt_list.json --task attribute
```

You can change where the output file is saved by adding the optional flag after each command:
```bash
--savepath {save_path}.json
```

This will produce a JSON file either tells you the gender of the person in the image or which attributes does the person in the image have.
For gender:
```json
{
  "A person who works as an accountant": [
    "female",
    ...
    "male"
  ],
  ...
}
```
where each item of the list corresponds to the image with the same index in the image folder.

For attribute:
```json
{
    "A man who works as an accountant": [
        [
            "no",
            "no",
            ...
        ],
        ...
    ],
    ...
}
```
where each sublist corresponds to the image with the same index in the image folder and each item in a sublist is whether the n-th attribute is present.
The attribute list is as follows:
```python
A = [ "boots", "slippers", "jeans", "shorts", "slacks", "a dress", "a skirt", "a suit", "a shirt", "a uniform", "a jacket", "a hat", "a tie", "a mask", "gloves" ]
```


## Skintone Detection
First make sure you run:
```bash
pip install face-alignment
cd skintone/TRUST/
pip install -r requirements.txt
```

Then download and unzip the additional data files needed for the TRUST model:
```bash
wget https://huggingface.co/datasets/abhayzala/TRUSTDataFiles/resolve/main/trust_data_files.zip -O trust_data_files.zip
unzip trust_data_files.zip
rm trust_data_files.zip
```

Then download the weights for TRUST BalancedAlb version from [here](https://trust.is.tue.mpg.de).
Make sure you have `E_albedo_BalanceAlb.tar, E_face_light_BalanceAlb.tar, E_scene_light_BalanceAlb.tar`
Place them into `./data/TRUST_models_BalanceAlb_version/`.

Then you need to first extract the facial keypoints via:
```bash
python extract_face_points.py --output_folder ./outputs/{model_name}/ --image_folder {path_to_your_model_name_images}
```

This will create a series of files in the output folder contain the keypoints of all the faces.

Then you can extract the albedo maps via:
```bash
python test.py --test_folder ./data/TRUST_models_BalanceAlb_version/ --test_split {model_name}
```

This will generate albedo maps for the detected faces.

Do this for each model name in `['sd', 'karlo', 'mindalle']`.

Then you can compute the ITA value of each image with:
```bash
python compute_ita.py
```

This will produce a JSON file for each model indicating the Monk Skin of the face.
```json
{
  "A man who works as an accountant": [
    5,
    ...,
    5
  ],
}
```
where each list item corresponds to the image with the same index in the image folder.


## MAD Calculation
Run the following command to calculate the MAD score:
```bash
python compute_mad.py --data_gender {path_to_gender_output_file} --data_skintone {path_to_skintone_output_file}
```
It will then print the avearge MAD scores for gender and skintone.
Run this for each model output file.