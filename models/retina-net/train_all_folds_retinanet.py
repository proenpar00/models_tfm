import os
import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
import pandas as pd
import shutil

# Configuración
NUM_CLASSES = 6  # Número de clases (sin contar el fondo)
NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'models/retina-net/data_kfold_coco_retinanet'
RESULTS_DIR = 'models/retina-net/results'

# Función para registrar los datasets en formato COCO
def register_datasets():
    for fold in range(10):
        fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
        train_img_dir = os.path.join(fold_dir, 'train', 'images')
        val_img_dir = os.path.join(fold_dir, 'val', 'images')
        train_ann_file = os.path.join(fold_dir, 'train', 'annotations.json')
        val_ann_file = os.path.join(fold_dir, 'val', 'annotations.json')

        register_coco_instances(f"fold_{fold}_train", {}, train_ann_file, train_img_dir)
        register_coco_instances(f"fold_{fold}_val", {}, val_ann_file, val_img_dir)

# Función para configurar el modelo
def get_cfg_for_fold(fold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"fold_{fold}_train",)
    cfg.DATASETS.TEST = (f"fold_{fold}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = LEARNING_RATE
    cfg.SOLVER.MAX_ITER = NUM_EPOCHS * 100  
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE = "cpu"  # forzar CPU
    cfg.OUTPUT_DIR = os.path.join(RESULTS_DIR, f'fold_{fold}')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

# Entrenamiento por fold
def train_all_folds():
    register_datasets()
    summary = []

    for fold in range(10):
        print(f'Entrenando Fold {fold}')
        cfg = get_cfg_for_fold(fold)

        # Inicializar entrenador
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Evaluación
        evaluator = COCOEvaluator(f"fold_{fold}_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = detectron2.data.build_detection_test_loader(cfg, f"fold_{fold}_val")
        metrics = detectron2.engine.DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])

        # Guardar métricas
        metrics['Fold'] = fold
        summary.append(metrics)

        # Copiar el mejor modelo
        best_model_src = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        best_model_dst = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
        shutil.copyfile(best_model_src, best_model_dst)

    # Guardar resumen de métricas
    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(os.path.join(RESULTS_DIR, 'metrics_by_fold.xlsx'), index=False)

if __name__ == "__main__":
    train_all_folds()
