# from ultralytics import YOLO
# import os

# # 加载预训练的 YOLO 模型
# model = YOLO('runs/exp4/trial6/weights/best.pt')

# # 定义包含图像文件用于推理的目录路径
# source = r'datasets/dataset_200/images/test'

# # 确保保存结果的目录存在
# save_dir = r'runs/exp4/trial7/test'
# os.makedirs(save_dir, exist_ok=True)

# # 获取文件夹中所有图片的路径
# image_files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.jpg')]


# for img_path in image_files:
#     result = model.predict(img_path, conf=0.8)  # 进行推理

#     # 遍历每个结果
#     for r in result:  # r是每个预测结果
#         # 获取每个结果的图像  
#         save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + '.jpg')  # 保存为jpg格式
        
#         # 保存绘制了预测结果的图像
#         r.plot(line_width=1, save=True, filename=save_path)

#         # 保存预测结果的文本文件
#         txt_save_path = save_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt') # 替换文件扩展名为 .txt
#         r.save_txt(txt_save_path)  # 保存文本预测结果

# # 打印或处理预测结果
# print("\n成功处理" + str(len(image_files)) + "张图片\t", "预测结果已保存至：", save_dir)  


from ultralytics import YOLO
import os

# 加载预训练的 YOLO 模型
model = YOLO('runs/exp4/weights/best.pt')

# 定义包含图像文件用于推理的目录路径
source = r'datasets/dataset_200/images/test/163.jpg'

# 确保保存结果的目录存在
save_dir = r'runs/'
os.makedirs(save_dir, exist_ok=True)


result = model.predict(source, conf=0.8)  # 进行推理

# 遍历每个结果
for r in result:  # r是每个预测结果
    print(type(r))
    # 获取每个结果的图像  
    save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(source))[0] + '.jpg')  # 保存为jpg格式
    
    # 保存绘制了预测结果的图像
    r.plot(line_width=1, save=True, filename=save_path)

    # 保存预测结果的文本文件
    txt_save_path = save_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt') # 替换文件扩展名为 .txt
    r.save_txt(txt_save_path)  # 保存文本预测结果



