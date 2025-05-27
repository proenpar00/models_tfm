import json



def fix_categories(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    max_cat_id = max(ann['category_id'] for ann in annotations)
    
    # Crear categorías automáticas con nombres genéricos
    data['categories'] = [{"id": i, "name": f"class_{i}"} for i in range(1, max_cat_id + 1)]
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Categorías corregidas en {json_path}, máximo category_id: {max_cat_id}")

# Aplica para cada fold, train y val
for fold in range(10):
    for split in ['train', 'val']:
        path = f"models/retina-net/data_kfold_coco_retinanet/fold_{fold}/{split}/annotations.json"
        fix_categories(path)