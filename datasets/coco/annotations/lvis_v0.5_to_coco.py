"""
Script transforming LVIS v0.5 annotations to COCO annotations.
"""
from copy import deepcopy
import json


# Load LVIS training and validation jsons
lvis_train_path = '../../lvis/annotations/lvis_v0.5_train.json'
lvis_val_path = '../../lvis/annotations/lvis_v0.5_val.json'

with open(lvis_train_path, 'r') as json_file:
    lvis_train_json = json.load(json_file)

with open(lvis_val_path, 'r') as json_file:
    lvis_val_json = json.load(json_file)

# Get dictionary transforming LVIS category ids to COCO category ids
synset_to_lvis = {cat['synset']: cat['id'] for cat in lvis_train_json['categories']}

with open('coco_to_synset.json', 'r') as json_file:
    coco_to_synset_json = json.load(json_file)

synset_to_coco = {v['synset']: v['coco_cat_id'] for v in coco_to_synset_json.values()}
lvis_to_coco = {synset_to_lvis[k]: v for k, v in synset_to_coco.items() if k in synset_to_lvis}

# Get COCO training and validation jsons
coco_train_json = deepcopy(lvis_train_json)
coco_val_json = deepcopy(lvis_val_json)

coco_train_json['info']['description'] = "Subset of COCO 2017 training images with LVIS v0.5 annotations"
coco_val_json['info']['description'] = "COCO 2017 validation images with LVIS v0.5 annotations"

coco_ann_id = 1
coco_train_json['annotations'] = []

for lvis_ann in lvis_train_json['annotations']:
    lvis_cat_id = lvis_ann['category_id']

    if lvis_cat_id not in lvis_to_coco:
        continue

    coco_ann = {k: v for k, v in lvis_ann.items() if k not in ('id', 'category_id')}
    coco_ann['id'] = coco_ann_id
    coco_ann['category_id'] = lvis_to_coco[lvis_cat_id]
    coco_ann['is_crowd'] = 0

    coco_ann_id += 1
    coco_train_json['annotations'].append(coco_ann)

coco_ann_id = 1
coco_val_json['annotations'] = []

for lvis_ann in lvis_val_json['annotations']:
    lvis_cat_id = lvis_ann['category_id']

    if lvis_cat_id not in lvis_to_coco:
        continue

    coco_ann = {k: v for k, v in lvis_ann.items() if k not in ('id', 'category_id')}
    coco_ann['id'] = coco_ann_id
    coco_ann['category_id'] = lvis_to_coco[lvis_cat_id]
    coco_ann['is_crowd'] = 0

    coco_ann_id += 1
    coco_val_json['annotations'].append(coco_ann)

# Save COCO training and validation jsons
coco_train_path = 'instances_train_lvis_v0.5.json'
coco_val_path = 'instances_val_lvis_v0.5.json'

with open(coco_train_path, 'w') as json_file:
    json.dump(coco_train_json, json_file)

with open(coco_val_path, 'w') as json_file:
    json.dump(coco_val_json, json_file)
