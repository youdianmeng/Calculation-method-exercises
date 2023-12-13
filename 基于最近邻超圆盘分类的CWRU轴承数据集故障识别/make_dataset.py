import numpy as np
from scipy.io import loadmat
import glob
import re
import os

def make_dataset(data_src, num_samples, class_):
    
    # 获取目标文件夹数据
    req_key = "_DE_time" 
    pattern = re.compile(req_key)
    files = glob.glob(data_src)
    files = np.sort(files)
    data = loadmat(files[0])
    keysList = [key for key in data]
    for key in keysList:
        if pattern.search(key):
            my_key = key
    drive_end_data = data[my_key]
    drive_end_data = drive_end_data.reshape(-1) # 变一维
    # 分割数据块
    num_segments = np.floor(len(drive_end_data)/num_samples)
    #slices = np.split(drive_end_data[0:int(num_segments*num_samples)], num_samples)
    slices = drive_end_data[0:int(num_segments*num_samples)]
    silces = slices.reshape(int(num_segments), num_samples)
    segmented_data = silces
    # 预防性操作
    files = files[1:]
    for file in files:
        data = loadmat(file)
        keysList = [key for key in data]
        for key in keysList:
            if pattern.search(key):
                my_key = key
        drive_end_data = data[my_key]
        drive_end_data = drive_end_data.reshape(-1)
        num_segments = np.floor(len(drive_end_data)/num_samples)
        #slices = np.split(drive_end_data[0:int(num_segments*num_samples)], num_samples)
        slices = drive_end_data[0:int(num_segments*num_samples)]
        silces = slices.reshape(int(num_segments), num_samples)
        segmented_data = np.concatenate( (segmented_data, silces) , axis=0, out=None)
    
    segmented_data = np.unique(segmented_data, axis= 0) # remove duplicates
    np.random.shuffle( segmented_data) # suffule the data
    Class_ = np.ones(len(segmented_data)).astype(int)*class_
    
    return segmented_data, Class_