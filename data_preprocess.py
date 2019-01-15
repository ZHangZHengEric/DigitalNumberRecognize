import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import tensorflow as tf

def loadCSVfile2():
    data = pd.read_csv('data/train.csv')
    images = data.iloc[:,1:].values
    labels = data.iloc[:,:1].values.ravel()
    # images = np.multiply(images,1.0/255.0)
    images = images.reshape(images.shape[0],1,28,28).transpose(0, 2, 3, 1)
    print( images.shape)
    print( labels.shape)
    return images,labels

def save_csv_as_image_files(path , images ,labels):
    print("save_csv_as_image_files")
    for i in range(0,len(labels)):
        folder = path+"/"+ str(labels[i])
        isExists=os.path.exists(folder)
        if not isExists:
            os.makedirs(folder) 
            print(path+' 创建成功')
        file_path = folder+"/"+str(i)+".jpg"
        cv2.imwrite(file_path,images[i])

def rotate_image(images ,labels):
    labels_len  = len(labels)
    for i in range(0,labels_len):
        print(i)
        image = Image.fromarray(np.uint8(images[i])[:,:,0])
        for angle in range(-10,11,20):
            # plt.imshow(image)
            after_rotate = np.array(image.rotate(angle)) 
            # plt.imshow(after_rotate)
            # plt.show()
            np.append(images,after_rotate) 
            np.append(labels,labels[i])
    return images,labels

def get_batch_data(images,label,batch_size=3,num_epochs =10,w=100,h=100,c=100,use_shuffle = True):
    image=tf.cast(images,tf.string)
    label=tf.cast(label,tf.int32)
    input_queue = tf.train.slice_input_producer([image, label],num_epochs= num_epochs, shuffle=use_shuffle)

    print (input_queue[0])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])

    image_contents=tf.image.decode_jpeg(image_contents,channels=c)

    image_contents = tf.image.resize_images(image_contents,[w,h])
    # image_contents=tf.image.resize_image_with_crop_or_pad(image_contents,w,h)
    image_contents=tf.image.per_image_standardization(image_contents)

    min_after_dequeue=10
    capacity=min_after_dequeue+2*batch_size
    image_batch,label_batch=tf.train.shuffle_batch([image_contents,label],batch_size=batch_size,num_threads=8,capacity=capacity,min_after_dequeue=min_after_dequeue)

    image_batch = tf.reshape(image_batch,[batch_size,w,h,c]) 
    image_batch=tf.cast(image_batch,np.float32) 
    return image_batch, label_batch

def find_all_data(path):
    img_paths = []
    labels = []
    classes = os.listdir(path)
    names=[]
    for name in classes:
        names.append(name)
    names.sort()
    for idx, folder in enumerate(names):
        cate = os.path.join(path,folder)
        
        print(idx,folder)
        for im in os.listdir(cate):
            img_path = os.path.join(cate,im)
            img_paths.append(img_path)
            labels.append(idx)
            # print(img_path , idx)
    print(len(img_paths),len(labels))
    return img_paths,labels 