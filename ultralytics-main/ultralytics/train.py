import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO(r"F:/DL/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml")
 
    # model.load('yolov8n.pt') # 是否加载预训练权重，可以加载也可以不加载
 
    model.train(data=r'F:/DL/ultralytics-main/equation.yaml',
                cache=False, # 是否生成告诉缓存文件，生成以后训练的速度是会提高的
                imgsz=640, # 图片输入模型的尺寸
                epochs=150, # 训练的轮数
                single_cls=False,  # 是否是单类别检测
                batch=4, # batch_size 的大小
                close_mosaic=10, # 最后10个epoch关闭马赛克增强
                workers=0, # 线程数
                device='cpu', # 选择哪个卡进行训练， 没有GPU卡的话可以修改成 device="cpu"
                optimizer='SGD', # 使用 SGD 优化器
                resume='runs/train/exp/weights/last.pt', # 如过想继续训就设置last.pt的地址
                amp=True,  # 开启自动混合精度训练
                project='runs/train',
                name='exp_20241103_1', #这个name建议大家修改，我一般修改为数据集的名称+时间戳，这样你保存的实验结果可知道是什么适合做的哪个实验
                )