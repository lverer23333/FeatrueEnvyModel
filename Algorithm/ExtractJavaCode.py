# -*- encoding: utf-8 -*-
"""
@File    : ExtractJavaCode.py
@Time    : 2019/9/11 20:31
@Author  : knight
# code is far away from bugs with the god animal protecting
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃        ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import os
import re

# project_path = "D:/桌面/源码JavaProject/7个训练集/Areca/areca-7.4.7-src"
project_path = "C:/Users/AAAA/Desktop/1/ArtOfIllusion-master"
output_path = "E:/PycharmProjects/FeatureEnvy-master/Algorithm/allcode.txt"

def findAll(dirs):
    for a_dir in os.listdir(dirs):
        print(a_dir)
        new_dir = dirs+'/'+a_dir
        if a_dir.endswith('.java'):
            read_file = open(new_dir,'rb')
            write_file = open(output_path,'ab+')
            data = read_file.read()
            # print(data)
            write_file.write(data)
            read_file.close()
            write_file.close()

        if os.path.isdir(new_dir):
            findAll(new_dir)
            os.chdir(new_dir)


if __name__ == "__main__":
    findAll(project_path)

