
# DETR-based Visual Reasoning Skill Evaluation on PaintSkills


## Training DETR on 4 skills

```bash
bash scripts/train_detr_skill.sh 4 --skill_name object
bash scripts/train_detr_skill.sh 4 --skill_name count
bash scripts/train_detr_skill.sh 4 --skill_name color
bash scripts/train_detr_skill.sh 4 --skill_name spatial
```

## Evaluation

### Evaluation on GT images

```bash
bash scripts/evaluate_skill.sh --skill_name object --gt_data_eval
bash scripts/evaluate_skill.sh --skill_name count --gt_data_eval
bash scripts/evaluate_skill.sh --skill_name color --gt_data_eval
bash scripts/evaluate_skill.sh --skill_name spatial --gt_data_eval
```