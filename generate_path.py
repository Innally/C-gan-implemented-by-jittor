#####需要先对所有的图片生成对应的标签以及路经#####
import os
import random
from sklearn.model_selection import train_test_split
def get_path_label(dir, relative_path, id_labels,id_path, labels):
    global img_id,category
    ###采用递归的模式  如果当前这个文件夹里面都没有其他文件夹那么此必定属于一个新的标签
    list_d = os.listdir(os.path.join(dir,relative_path))
    ok = True
    for f in list_d:
        if os.path.isdir(os.path.join(dir,relative_path,f)):
            ok = False
            break
    ####如果ok那么必定全部是图片，则产生新标签，并且将所有有的属于该标签的图片加入####
    if ok == True:
        ###添加相对于主文件夹的路径####
        for f in list_d:
            ##每张图##
            id_labels.append(str(img_id)+' '+str(category)+'\n')
            labels.append(category)
            id_path.append(str(img_id)+' '+os.path.join(relative_path,f)+'\n')
            img_id+=1
        category += 1
    else:
        ###否则就继续迭代###
        for f in list_d:
            if os.path.isdir(os.path.join(dir,relative_path,f)):
                get_path_label(dir,os.path.join(relative_path,f),id_labels,id_path, labels)

def save(obj, path):
    with open(path, 'w', encoding='utf-8') as fp:
        fp.writelines(obj)
        fp.close()

def save_train(count, path):
    ##生成下标0-count-1的训练随机数！！！###
    ###训练集：测试集或者验证集= 4：1
    train_count = int(0.8*count)
    rand_nums = list(range(0,count))
    random.shuffle(rand_nums)
    dic_train_ids = {}
    for ii in range(train_count):
        dic_train_ids[rand_nums[ii]] = 1
    with open(path, 'w', encoding='utf-8') as fp:
        for every_id in range(count):
            if every_id in dic_train_ids.keys():
                fp.write(str(every_id) + ' ' + '1\n')
            else:
                fp.write(str(every_id) + ' ' + '0\n')
        fp.close()

def save_model_train_test(count, labels, path):
    X = list(range(count))
    train_X, test_X, train_y,test_y = train_test_split(X, labels, stratify=labels, test_size=0.2, random_state=1)
    dic_id_train = {}
    for x_id in train_X:
        dic_id_train[x_id] = 1
    with open(path, 'w', encoding='utf-8') as fp:
        for every_id in range(count):
            if every_id in dic_id_train.keys():
                fp.write(str(every_id) + ' ' + '1\n')
            else:
                ### 全部都是1 no validate###
                fp.write(str(every_id) + ' ' + '1\n')
        fp.close()


if __name__ == '__main__':
    img_dir = '/home/innally/images/'
    img_id = 0
    category = 0
    id_path = []
    id_labels = []
    labels = []
    get_path_label(os.path.join(img_dir,'flower'), '', id_labels, id_path, labels)



    save(id_path,os.path.join('/home/innally/Documents/flower','images.txt'))
    save(id_labels, os.path.join('/home/innally/Documents/flower','image_class_labels.txt'))
    save_model_train_test(len(id_path), labels, os.path.join('/home/innally/Documents/flower','train_test_split.txt'))

    # save_train(len(id_path), os.path.join('/home/ubuntu7/SD_Dataset/sd_train_v1','train_test_split.txt'))

# import torch
# print(torch.__version__)