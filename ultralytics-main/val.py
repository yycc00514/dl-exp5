# 验证val

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/exp4/prune1/weights/prune.pt') # 剪枝后的模型权重
    model.val(data='equation.yaml', split='val', imgsz=640, batch=8, save_json=True, workers=0,
              project='runs/exp4/prune1', name='val')