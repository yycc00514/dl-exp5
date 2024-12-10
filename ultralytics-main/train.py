# 剪枝
# =================================约束训练=================================
# import os
# from ultralytics import YOLO
# import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
 
# def main():
#     model = YOLO(r'ultralytics/cfg/models/v8/yolov8m.yaml').load('runs/train/exp22/weights/best.pt')
#     model.train(data="equation.yaml", amp=False, imgsz=640, epochs=100, batch=16, device=0, workers=0, 
#                 project='runs/exp4', name='trial')
 
 
# if __name__ == '__main__':
#     main()
# ==========================================================================



# =================================回调训练=================================
# import os
# from ultralytics import YOLO
# import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
# def main():
#     model = YOLO('runs/exp4/prune/weights/prune.pt')
#     model.train(data="equation.yaml", amp=False, imgsz=640, epochs=400, batch=8, device=0, workers=0, 
#                 project='runs/exp4', name='trial')
 
# if __name__ == '__main__':
#     main()
# ==========================================================================


# # 蒸馏
# =================================教师模型=================================
# import os
# from ultralytics import YOLO
# import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
 
# def main():
#     model = YOLO("yolov8x.pt")
#     model.train(data="equation.yaml", amp=False, imgsz=640, epochs=600, batch=8, device=0, workers=0, 
#                project='runs/exp4', name='model_t',)
 
 
# if __name__ == '__main__':
#     main()
# ==========================================================================


# =================================回调训练=================================
# import os
# from ultralytics import YOLO
# import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
# def main():
#     model = YOLO('runs/exp4/distillation/weights/best.pt')
#     model.train(data="equation.yaml", amp=False, imgsz=640, epochs=400, batch=8, device=0, workers=0, 
#                 project='runs/exp4', name='trial')
 
# if __name__ == '__main__':
#     main()
# ==========================================================================