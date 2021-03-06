import os
import errno
import numpy as np
import scipy
import scipy.misc
import pickle
import random

import tensorflow as tf
import cv2
import imageio

from keras.datasets import mnist

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):       
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def transform(image, npx=64, is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image,
                                            [resize_w, resize_w])
    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        #image = scipy.misc.imread(path).astype(np.float)
        #if np.image.shape
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return ((image + 1) * 127.5).astype(np.uint8)

class CelebA(object):
    def __init__(self, images_path):

        self.dataname = "CelebA"
        self.dims = 64 * 64
        self.shape = [64, 64, 3]
        self.image_size = 64
        self.channel = 3
        self.images_path = images_path
        self.train_data_list, self.train_lab_list = self.load_celebA()

    def load_celebA(self):

        # get the list of image path
        return read_image_list_file(self.images_path, is_test=False)

    def load_test_celebA(self):

        # get the list of image path
        return read_image_list_file(self.images_path, is_test=True)

class InputData(object):
    def __init__(self, train_images_path, valid_images_path, anomaly_images_path, test_images_path):

        self.image_size = 0
        self.channel = 0
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path
        self.train_data_list = self.load_input_pickle(train_images_path)
        self.valid_data_list = self.load_input_pickle(valid_images_path)
        self.anomaly_data_list = self.load_input_pickle(anomaly_images_path)
        self.test_data_list = self.load_input_pickle(test_images_path)

    def load_input(self, images_path):

        # get the list of image path
        return read_image_path(images_path)

    def load_input_pickle(self, pickle_path):
            
        if pickle_path==None:
            return None

        print("Load pickle: {}".format(pickle_path))

        features, labels = pickle.load(open(pickle_path, mode='rb'))

        print("==shape: {}".format(features.shape))
                
        self.image_size = np.shape(features)[1]
        self.channel = np.shape(features)[-1]

        return (features, labels)

class InputMNISTData(object):
    def __init__(self):

        self.image_size = 32
        self.channel = 1
        self.train_images_path = None
        self.test_images_path = None
        self.train_data_list = []
        self.valid_data_list = []
        self.anomaly_data_list = []
        self.test_data_list = []
        self.anoCls = []

    def reshape_x(self, x):

        new_x = np.empty((len(x), 32, 32))
        for i, e in enumerate(x):
            new_x[i] = cv2.resize(e, (32, 32))
        #return np.expand_dims(new_x, axis=-1) / 127 - 1
        return new_x

    def one_hot_encode(self, x):

        output = np.zeros((len(x), 10))
        
        for i, j in enumerate(x):
            output[i,j] = 1
               
        return output
    
    def load_input_data(self, anoCls):
        
        self.anoCls = anoCls
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
                       
        x_train = self.reshape_x(x_train) / 255.0
        x_test = self.reshape_x(x_test) / 255.0
               
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)       
        
        x_train_normal = x_train[y_train != anoCls] 
        y_train_normal = y_train[y_train != anoCls] 
        x_train_ano = x_train[y_train == anoCls] 
        y_train_ano = y_train[y_train == anoCls] 
        x_test_normal = x_test[y_test != anoCls] 
        y_test_normal = y_test[y_test != anoCls] 

        y_train_normal = self.one_hot_encode(y_train_normal)
        y_train_ano = self.one_hot_encode(y_train_ano)
        y_test_normal = self.one_hot_encode(y_test_normal)
        #y_test = self.one_hot_encode(y_test)
                       
        self.train_data_list = (x_train_normal, y_train_normal)
        print("==[MNIST] Train data shape: {}, {}".format(x_train_normal.shape, y_train_normal.shape))
        self.valid_data_list = (x_test_normal, y_test_normal)
        print("==[MNIST] Validation data shape: {}, {}".format(x_test_normal.shape, y_test_normal.shape))
        self.anomaly_data_list = (x_train_ano, y_train_ano)
        print("==[MNIST] Anomaly data shape: {}, {}".format(x_train_ano.shape, y_train_ano.shape))
        #self.test_data_list = (x_test, y_test)
        #print("==[MNIST] Test data shape: {}, {}".format(x_test.shape, y_test.shape))
    
    def load_test_data(self, anoCls):
    
        self.anoCls = anoCls
        
        _, (x_test, y_test) = mnist.load_data()
        x_test = self.reshape_x(x_test) / 255.0
        x_test = np.expand_dims(x_test, axis=-1)       
        
        x_test_ano = x_test[y_test == anoCls] 
        y_test_ano = y_test[y_test == anoCls] 
        y_test_ano = self.one_hot_encode(y_test_ano)

        self.test_data_list = (x_test_ano, y_test_ano)
        print("==[MNIST] Test data shape: {}, {}".format(x_test_ano.shape, y_test_ano.shape))
        
class InputAllClassData(object):
    def __init__(self, anoCls, train_images_path, valid_images_path, anomaly_images_path, test_images_path):

        self.dataname = "InputData"
        self.image_size = 0
        self.channel = 0
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path
        self.train_data_dict = self.load_class_data(anoCls, train_images_path, 'train')
        self.valid_data_dict = self.load_class_data(anoCls, valid_images_path, 'test')
        self.anomaly_data_list = self.load_input_pickle(anomaly_images_path)
        self.test_data_list = self.load_input_pickle(test_images_path)

    def load_input(self, images_path):

        # get the list of image path
        return read_image_path(images_path)

    def load_input_pickle(self, pickle_path):
            
        if pickle_path == None:
            return None

        features, labels = pickle.load(open(pickle_path, mode='rb'))
        
        self.image_size = np.shape(features)[1]
        self.channel = np.shape(features)[-1]

        return (features, labels)
    
    def load_class_data(self, anoCls, dir_path, data_type):
        
        if dir_path == None:
            return None

        cls_dict = {}
                
        for i in range(0, 10):
            
            if i in anoCls:
                continue
            
            pickle_path = dir_path + "/pr_" + data_type + "_class_" + str(i) + ".p"
            cls_dict[i] = self.load_input_pickle(pickle_path)
            
        return cls_dict

def read_image_list_file(category, is_test):
    end_num = 0
    if is_test == False:

        start_num = 1202
        path = category + "celebA/"

    else:

        start_num = 4
        path = category + "celeba_test/"
        end_num = 1202

    list_image = []
    list_label = []

    lines = open(category + "list_attr_celeba.txt")
    li_num = 0
    for line in lines:

        if li_num < start_num:
            li_num += 1
            continue

        if li_num >= end_num and is_test == True:
            break

        flag = line.split('1 ', 41)[20]  # get the label for gender
        file_name = line.split(' ', 1)[0]

        # print flag
        if flag == ' ':

            list_label.append(1)

        else:

            list_label.append(0)

        list_image.append(path + file_name)

        li_num += 1

    lines.close()

    return list_image, list_label
    
def read_image_path(path):
    list_image = []

    file_name = os.listdir(path)

    list_image = [os.path.join(path, file_name[i]) for i in range(len(file_name))]

    return list_image
    
def reshape_image(input, size):
    
    width, height, channel = size
    
    temp_x = np.empty((len(input), width, height, channel))
    
    for i, e in enumerate(input):
        temp_x[i] = cv2.resize(e, (width, height))

    return temp_x

def get_batch(data, batch_size):
    
    #idx = np.array(list(range(0, len(data[0]))))
    
    #random.shuffle(idx)
    
    idx = np.random.randint(0, len(data[0]), batch_size)
    
    next_x_images = data[0][idx[0:batch_size]]
    next_y = data[1][idx[0:batch_size]]     
    
    return next_x_images, next_y
    
def get_batch_by_class(data, batch_size, anoCls, random_order=False):
    
    class_num = 10 - len(anoCls)
    sample_num = batch_size // class_num 
    batch_data = np.array([])
    batch_label = np.array([])
    
    if random_order == True:
        class_idx = np.array(list(range(0, 10)))
        rdm_class_idx = np.array(list(range(0, 10)))
        while True:
            random.shuffle(rdm_class_idx)
            if np.sum(rdm_class_idx-class_idx) == 0:
                break

    for i in range(0, 10):
        
        if random_order == True:
            idx = rdm_class_idx[i]
        else:
            idx = i
        
        if idx in anoCls:
            continue

        rdm_idx = np.array(list(range(0, len(data[idx][0]))))        
        random.shuffle(rdm_idx)
        
        temp_data = data[idx][0][rdm_idx[0:sample_num]]        
        temp_label = data[idx][1][rdm_idx[0:sample_num]]        

        batch_data = np.append(batch_data, temp_data)
        batch_label = np.append(batch_label, temp_label)

    batch_data = np.reshape(batch_data, tuple([-1, np.shape(temp_data)[1], np.shape(temp_data)[2], np.shape(temp_data)[3]]))
    batch_label = np.reshape(batch_label, tuple([-1, np.shape(temp_label)[1]]))
    
    rest_sample = batch_size % (class_num)
    if rest_sample != 0:
           
        while True:
            rdm_cls = random.randint(0, class_num-1)
            if rdm_cls not in anoCls:
                break

        rdm_idx = np.array(list(range(0, len(data[rdm_cls][0]))))     
        random.shuffle(rdm_idx)
        
        temp_data = data[rdm_cls][0][rdm_idx[0:rest_sample]]        
        temp_label = data[rdm_cls][1][rdm_idx[0:rest_sample]]             

        batch_data = np.append(batch_data, temp_data, axis=0)
        batch_label = np.append(batch_label, temp_label, axis=0)
        
    return batch_data, batch_label        

def img_merge(images, output_size):
    
    h = images.shape[1]
    w = images.shape[2]
    
    output_image = np.zeros((int(h*output_size[0]), int(w*output_size[1]), 3))
    
    for idx, curr_image in enumerate(images):
        
        i = idx % output_size[1]
        j = idx // output_size[1]
        output_image[j * h:j * h + h, i * w: i * w + w, :] = curr_image
    
    return output_image

def img_output(output_image, output_path):

    if len(output_image.shape) <= 3:
        output_image = np.expand_dims(output_image, axis=-1)
    
    print(output_image.shape)
    
    output_image = img_merge(output_image[0:256], [16, 16])

    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    
    imageio.imwrite(output_path, output_image)
