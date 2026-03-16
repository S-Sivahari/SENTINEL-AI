from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8m.pt")

image_folder = Path("pics/val_dataset")
images = list(image_folder.glob("*.jpg")) + \
         list(image_folder.glob("*.png")) + \
         list(image_folder.glob("*.jpeg"))

print(f"Found {len(images)} images\n")

for img_path in images:
    print(f"{'='*50}")
    print(f"Image: {img_path.name}")
    print(f"{'='*50}")
    
    results = model(str(img_path), verbose=False)
    
    for r in results:
        if len(r.boxes) == 0:
            print("  No objects detected")
            continue
            
        for box in r.boxes:
            cls_id      = int(box.cls)
            label       = model.names[cls_id]
            conf        = float(box.conf)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            
            print(f"  {label:<15} {conf:.0%} confidence   bbox: ({x1},{y1}) → ({x2},{y2})")
    
    print()