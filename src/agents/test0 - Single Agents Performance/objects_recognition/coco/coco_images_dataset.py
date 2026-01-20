from pycocotools.coco import COCO
import requests
import os
import json

# paths configuration
ANNOTATIONS_PATH = 'annotations_trainval2017/annotations/instances_val2017.json'
SAVE_DIR = 'sample_images/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Loading COCO datasets
coco = COCO(ANNOTATIONS_PATH)

# Obtaining 10 random images' IDs with at least 3 objets
image_ids = []
for img_id in coco.imgs:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    if len(ann_ids) >= 3:
        image_ids.append(img_id)
    if len(image_ids) == 10:
        break

# Dict for saving annotations
dataset_info = []

for img_id in image_ids:
    # image information
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    url = img_info['coco_url']

    # download image
    img_data = requests.get(url).content
    with open(os.path.join(SAVE_DIR, file_name), 'wb') as f:
        f.write(img_data)

    # obtaining annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Saving minimal annotations
    annotations = [{
        'category': coco.loadCats(ann['category_id'])[0]['name'],
        'bbox': ann['bbox']
    } for ann in anns]

    dataset_info.append({
        'image_id': img_id,
        'file_name': file_name,
        'url': url,
        'annotations': annotations
    })

# Saving dataset information
with open(os.path.join(SAVE_DIR, 'annotations.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)