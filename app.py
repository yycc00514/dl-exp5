import sys
import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
from streamlit.web import cli as stcli
import streamlit as st
from streamlit import runtime
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import streamlit
import atexit


# 进行yolo检测，呈现在web页面上
def detect(image, tempFolder):
    # st.subheader("Detected Image")
    # st.write("Just a second ...")

    # 加载模型
    model = YOLO('./best.pt')
    
    progressBar = st.progress(0)    # 显示进度条
    # img = image.copy()
    startTime = time.time()
    result = model.predict(os.path.join(tempFolder, image.name), conf=0.8)
    endTime = time.time()
    
    # 遍历每个结果
    for r in result:  # r是每个预测结果
        # 获取每个结果的图像  
        tempImagePath = os.path.join(tempFolder, image.name.replace('.jpg', '_detect.jpg'))  # 保存为jpg格式
        r.plot(line_width=1, save=True, filename=tempImagePath) # 保存绘制了预测结果的图像

        # 保存预测结果的文本文件
        tempLabelPath = tempImagePath.replace('.jpg', '.txt')   # 替换文件扩展名为 .txt
        r.save_txt(tempLabelPath)  # 保存文本预测结果

    for percent_complete in range(100):
        progressBar.progress(percent_complete + 1)
    
    st.write("检测结果: ")
    # 显示检测结果
    st.image(tempImagePath, caption=f"检测结果(用时{endTime - startTime: .2f}s)", width=300)


def detectDone():
    st.session_state.detectDone = True


def detectRes():
    st.session_state.detectRes = True


# 删除文件夹的函数
def delTempFolder():
    if os.path.exists('./temp'):
        shutil.rmtree('./temp')


def runStreamlit():
    print(os.listdir(os.getcwd()))
    # 初始化 session_state
    if 'detectDone' not in st.session_state:
        st.session_state.detectDone = False
    if 'detectRes' not in st.session_state:
        st.session_state.detectRes = None
    
    # 注册程序退出时调用的清理函数
    atexit.register(delTempFolder)

    st.title("口算题识别系统")
    st.write("")
    image = st.file_uploader("请在此处上传一张.jpg格式的图片", type="jpg")
    
    # 创建临时文件夹
    tempFolder = "./temp"
    saveFolder = "./res"
    if not os.path.exists(tempFolder):
        os.makedirs(tempFolder)
    
    if image is not None:
        imagePath = os.path.join(tempFolder, image.name)

        # 将图片写入文件
        with open(imagePath, "wb") as f:
            f.write(image.getbuffer())
        
        # image = Image.open(image)
        st.subheader("上传的图片: ")
        st.image(image, caption="上传的图片", width=300)    # 显示图片
        st.write("")

        # 开始模型预测
        st.button('点此开始口算题识别', on_click=detectDone)
        if st.session_state.detectDone:
            st.subheader("口算题识别")
            tempImagePath, tempLabelPath = detect(image, tempFolder)

            # 保存预测结果
            st.button('点此保存预测结果', on_click=detectRes)
            if st.session_state.detectRes:
                if not os.path.exists(saveFolder):  # 检查文件夹是否存在，如果不存在则创建
                    os.makedirs(saveFolder)

                # 生成预测结果图和标签的完整路径
                saveImagePath = os.path.join(saveFolder, image.name.replace('.jpg', '_detect.jpg'))
                saveLabelPath = saveImagePath.replace('.jpg', '.txt')

                # 将文件从源路径复制到目标文件夹
                shutil.copy(tempImagePath, saveImagePath)
                shutil.copy(tempLabelPath, saveLabelPath)
                st.write("已将预测结果保存至./res文件夹中! ")



if __name__ == '__main__':
    runStreamlit()
