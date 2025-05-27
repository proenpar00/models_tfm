import os
import shutil
from sklearn.model_selection import KFold

# Configuración
SOURCE_DIR = "data_yolo/train"
OUTPUT_DIR = "data_kfold_yolo"
NUM_FOLDS = 10
RANDOM_STATE = 42

image_dir = os.path.join(SOURCE_DIR, "images")
label_dir = os.path.join(SOURCE_DIR, "labels")

# Nombres de clases
class_names = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'NILM', 'SCC']

# Obtener todas las imágenes
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
image_files.sort()

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
    print(f"\n🔁 Generando fold {fold}...")

    for split_name, split_idx in [("train", train_idx), ("val", val_idx)]:
        images_split = [image_files[i] for i in split_idx]

        split_img_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}", split_name, "images")
        split_lbl_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}", split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_file in images_split:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"

            shutil.copy2(os.path.join(image_dir, img_file), os.path.join(split_img_dir, img_file))
            lbl_path = os.path.join(label_dir, label_file)
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(split_lbl_dir, label_file))
            else:
                print(f"⚠️ Etiqueta no encontrada: {label_file}")

    # Crear YAML por fold
    yaml_path = os.path.join(OUTPUT_DIR, f"data_fold_{fold}.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(OUTPUT_DIR)}/fold_{fold}/train/images\n")
        f.write(f"val: {os.path.abspath(OUTPUT_DIR)}/fold_{fold}/val/images\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

print("\n✅ División K-Fold completada y YAMLs generados.")
