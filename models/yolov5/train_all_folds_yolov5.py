import os
import csv
import pandas as pd
from ultralytics import YOLO

# Configuración
DATA_DIR = "data_kfold_yolo"
NUM_FOLDS = 10
RESULTS_DIR = "models/yolov5/results"
MODEL_DIR = "models/yolov5"
EPOCHS = 100
DEVICE = 'cpu' 

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "charts"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# CSV global por fold (una fila por fold)
metrics_summary_csv = os.path.join(RESULTS_DIR, "metrics.csv")
with open(metrics_summary_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "fold", "precision", "recall", "mAP50", "mAP50-95", "f1", "best_model_path"
    ])

# CSV global por época (muchas filas por fold)
metrics_per_epoch_path = os.path.join(RESULTS_DIR, "metrics_per_epoch.csv")
with open(metrics_per_epoch_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "fold", "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ])

# Entrenamiento por fold
for fold in range(NUM_FOLDS):
    print(f"\n Entrenando Fold {fold}")

    data_yaml = os.path.join(DATA_DIR, f"data_fold_{fold}.yaml")
    model = YOLO("yolov5n.pt")

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=640,
        project=MODEL_DIR,
        name=f"fold_{fold}",
        device=DEVICE,
        save=True,
        verbose=True
    )

    # Guardar métricas por época
    results_csv_path = os.path.join(MODEL_DIR, f"fold_{fold}", "results.csv")
    if os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        df["fold"] = fold
        df["epoch"] = df.index
        df = df[["fold", "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                 "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]]
        df.to_csv(metrics_per_epoch_path, mode="a", header=False, index=False)
    else:
        print(f" No se encontró {results_csv_path} para el fold {fold}")

    # Evaluar y guardar resumen
    val_metrics = model.val()
    metrics = val_metrics.results_dict

    best_model_path = os.path.join(MODEL_DIR, f"fold_{fold}/weights/best.pt")
    f1 = (2 * metrics["metrics/precision(B)"] * metrics["metrics/recall(B)"]) / \
         (metrics["metrics/precision(B)"] + metrics["metrics/recall(B)"] + 1e-8)

    with open(metrics_summary_csv, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            fold,
            round(metrics["metrics/precision(B)"], 4),
            round(metrics["metrics/recall(B)"], 4),
            round(metrics["metrics/mAP50(B)"], 4),
            round(metrics["metrics/mAP50-95(B)"], 4),
            round(f1, 4),
            best_model_path
        ])

print("\n ¡Todo listo! Métricas por fold y por época guardadas en 'results/'")
