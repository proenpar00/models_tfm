import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.datasets import CocoDetection
import pandas as pd
from tqdm import tqdm

# Configuración
NUM_CLASSES = 6  # Número de clases (sin contar el fondo)
NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DATA_DIR = 'models/retina-net/data_kfold_coco_retinanet'
RESULTS_DIR = 'models/retina-net/results'

# Transformaciones
transform = T.Compose([
    T.ToTensor()
])

# Función para cargar el conjunto de datos COCO
def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(img_dir, ann_file, transform=transform)

# Conversión de anotaciones COCO
def convert_coco_annotations(coco_targets):
    new_targets = []
    for anns in coco_targets:
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        target_dict = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        new_targets.append(target_dict)
    return new_targets

# Placeholder de métricas
def compute_metrics(outputs, targets):
    return {
        'precision': 0.0,
        'recall': 0.0,
        'map50': 0.0,
        'map50_95': 0.0,
        'f1_score': 0.0
    }

# Entrenamiento por fold
for fold in range(10):
    print(f'Entrenando Fold {fold}')
    fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
    train_dir = os.path.join(fold_dir, 'train', 'images')
    val_dir = os.path.join(fold_dir, 'val', 'images')
    train_ann = os.path.join(fold_dir, 'train', 'annotations.json')
    val_ann = os.path.join(fold_dir, 'val', 'annotations.json')

    # Cargar datasets
    train_dataset = get_coco_dataset(train_dir, train_ann)
    val_dataset = get_coco_dataset(val_dir, val_ann)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Inicializar modelo RetinaNet
    model = retinanet_resnet50_fpn(pretrained=True)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES + 1  # +1 por clase de fondo
    )
    model.to(DEVICE)

    # Optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Crear directorio de resultados
    fold_results_dir = os.path.join(RESULTS_DIR, f'fold_{fold}')
    os.makedirs(fold_results_dir, exist_ok=True)
    metrics_list = []
    best_map50 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            images = [img.to(DEVICE) for img in images]
            targets = convert_coco_annotations(targets)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        # Evaluación
        model.eval()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                outputs = model(images)
                all_outputs.extend(outputs)
                all_targets.extend(targets)

            metrics = compute_metrics(all_outputs, all_targets)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = epoch_loss / len(train_loader)
            metrics_list.append(metrics)

            # Guardar mejor modelo
            if metrics['map50'] > best_map50:
                best_map50 = metrics['map50']
                torch.save(model.state_dict(), os.path.join(fold_results_dir, 'best_model.pth'))

    # Guardar métricas por época
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(fold_results_dir, 'metrics.csv'), index=False)

# Resumen de métricas por fold
summary = []
for fold in range(10):
    df = pd.read_csv(os.path.join(RESULTS_DIR, f'fold_{fold}', 'metrics.csv'))
    best_epoch = df.loc[df['map50'].idxmax()]
    summary.append({
        'Fold': fold,
        'Precision': best_epoch['precision'],
        'Recall': best_epoch['recall'],
        'mAP@50': best_epoch['map50'],
        'mAP@50-95': best_epoch['map50_95'],
        'F1-score': best_epoch['f1_score']
    })

df_summary = pd.DataFrame(summary)
df_summary.to_excel(os.path.join(RESULTS_DIR, 'metrics_by_fold.xlsx'), index=False)
