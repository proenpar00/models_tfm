import os
import json
import shutil
from sklearn.model_selection import KFold

# --- Configuración ---
NUM_FOLDS = 10
DATA_DIR = r'C:\Users\cabel\Desktop\TFM_v2\MODELS_V2\MODELS\data_kfold_coco'
OUT_DIR = r'C:\Users\cabel\Desktop\TFM_v2\MODELS_V2\MODELS\data_kfold_coco_v2'
FULL_IMAGE_PATH = r'C:\Users\cabel\Desktop\TFM_v2\MODELS_V2\MODELS\data_kfold_coco\images'
ANNOT_PATH = r'C:\Users\cabel\Desktop\TFM_v2\MODELS_V2\MODELS\data_kfold_coco\annotations.json'
BASE_YAML_PATH = r'C:\Users\cabel\Desktop\TFM_v2\MODELS\data_kfold_coco_v2'

# --- Cargar anotaciones COCO ---
with open(ANNOT_PATH, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# Mapear imágenes a sus anotaciones
image_id_to_anns = {}
for ann in annotations:
    image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

# Crear folds
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f'Procesando fold {fold_idx}...')

    fold_path = os.path.join(OUT_DIR, f'fold_{fold_idx}')
    for split, idx in zip(['train', 'val'], [train_idx, val_idx]):
        split_path = os.path.join(fold_path, split)
        images_out_path = os.path.join(split_path, 'images')
        os.makedirs(images_out_path, exist_ok=True)

        split_images = [images[i] for i in idx]
        split_image_ids = {img['id'] for img in split_images}

        split_annotations = [ann for ann in annotations if ann['image_id'] in split_image_ids]

        # Copiar imágenes
        for img in split_images:
            original_file = os.path.join(FULL_IMAGE_PATH, img['file_name'])
            shutil.copy(original_file, os.path.join(images_out_path, img['file_name']))

        # Guardar anotaciones COCO
        split_annot = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": categories
        }

        with open(os.path.join(split_path, 'annotations.json'), 'w') as f:
            json.dump(split_annot, f)

    # Crear YAML para Detectron2
    yaml_content = f"""train: {BASE_YAML_PATH}/fold_{fold_idx}/train/images
val: {BASE_YAML_PATH}/fold_{fold_idx}/val/images

nc: {len(categories)}
names: {[cat['name'] for cat in categories]}
"""
    with open(f'data_fold_{fold_idx}.yaml', 'w') as f:
        f.write(yaml_content)

print("✅ K-Fold COCO estructura generada correctamente.")
