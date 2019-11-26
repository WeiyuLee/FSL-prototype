import scipy
import scipy.io as sio
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import os
import preprocess as preprc

def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_SVHN_mat(SVHN_dataset_folder_path, mat_name):
    """
    Load a mat of the dataset
    """
    mat = sio.loadmat(SVHN_dataset_folder_path + mat_name)
    
    data = mat['X']
    data = np.moveaxis(data, -1, 0)   
    label = mat['y'] - 1

    return data, label

def preprocess_and_save_data(SVHN_dataset_folder_path, output_path, rm_class, aug_enable=False, reshape_enable=False):
    """
    Preprocess Training and Validation Data
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load training data ======================================================
    train_data, train_label = load_SVHN_mat(SVHN_dataset_folder_path, "train_32x32.mat")
    
    # Preprocess training & validation data
    if aug_enable==True:
        train_data_ud = preprc.vertical_flip(train_data)
        train_data_lr = preprc.horizontal_flip(train_data)
        train_data = np.concatenate((train_data, train_data_ud, train_data_lr))
        train_label = np.concatenate((train_label, train_label, train_label))

    if reshape_enable==True:
        train_data = preprc.reshape_image(train_data, (64, 64, 3))    
         
    print("[Training data] Removing No.{} Class...".format(rm_class))
    print("\t[Before] train_data shape: ", np.shape(train_data))
    print("\t[Before] train_label shape: ", np.shape(train_label))    

    idx = np.squeeze(train_label != rm_class)
    train_data = train_data[idx]
    train_label = train_label[idx]

    train_data, _, _ = preprc.normalize(train_data, mean=[], std=[])
    train_label = preprc.one_hot_encode(train_label)

    print("\t[After] train_data shape: ", np.shape(train_data))
    print("\t[After] train_label shape: ", np.shape(train_label))
    
    # Save training data
    pickle.dump((train_data, train_label), open(os.path.join(output_path, 'preprocess_train_{}.p'.format(rm_class)), 'wb'), protocol=4)

    # Load Testing data =======================================================
    test_data, test_label = load_SVHN_mat(SVHN_dataset_folder_path, "test_32x32.mat")

    if reshape_enable==True:
        test_data = preprc.reshape_image(test_data, (64, 64, 3))    
    
    print("[Testing data] Removing No.{} Class...".format(rm_class))
    print("\t[Before] test_data shape: ", np.shape(test_data))
    print("\t[Before] test_label shape: ", np.shape(test_label))    

    idx = np.squeeze(test_label != rm_class)
    test_data_rm = test_data[idx]
    test_label_rm = test_label[idx]

    print("\t[After] test_data shape: ", np.shape(test_data_rm))
    print("\t[After] test_label shape: ", np.shape(test_label_rm))

    # Preprocess training & validation data
    test_data, _, _ = preprc.normalize(test_data, mean=[], std=[])
    test_label = preprc.one_hot_encode(test_label)
    test_data_rm, _, _ = preprc.normalize(test_data_rm, mean=[], std=[])
    test_label_rm = preprc.one_hot_encode(test_label_rm)

    # Save original test data
    pickle.dump((np.array(test_data), np.array(test_label)), open(os.path.join(output_path, 'test.p'), 'wb'))

    # Save test data
    pickle.dump((np.array(test_data_rm), np.array(test_label_rm)), open(os.path.join(output_path, 'preprocess_test_{}.p'.format(rm_class)), 'wb'), protocol=4)

def preprocess_and_save_single_class_data(SVHN_dataset_folder_path, output_path, aug_enable=False, reshape_enable=False):
    """
    Preprocess Training and Validation Data
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load training data ======================================================
    train_data, train_label = load_SVHN_mat(SVHN_dataset_folder_path, "train_32x32.mat")
        
    # Preprocess training & validation data
    if aug_enable==True:
        train_data_ud = preprc.vertical_flip(train_data)
        train_data_lr = preprc.horizontal_flip(train_data)
        train_data = np.concatenate((train_data, train_data_ud, train_data_lr))
        train_label = np.concatenate((train_label, train_label, train_label))

    if reshape_enable==True:
        train_data = preprc.reshape_image(train_data, (64, 64, 3))   

    train_data, _, _ = preprc.normalize(train_data, mean=[], std=[])
    train_label = preprc.one_hot_encode(train_label)
    
    for reserved_class in range(10):
          
        print("[Training data] Extracting No.{} Class...".format(reserved_class))

        curr_features = train_data[train_label[:, reserved_class] == 1]
        curr_lables = train_label[train_label[:, reserved_class] == 1]
    
        print("\t[Class {}] feature shape: ".format(reserved_class), np.shape(curr_features))
        print(np.min(curr_features))
        print(np.max(curr_features))        
        # Save training data
        pickle.dump((curr_features, curr_lables), open(os.path.join(output_path, 'pr_train_class_{}.p'.format(reserved_class)), 'wb'))

    # Load Testing data =======================================================
    test_data, test_label = load_SVHN_mat(SVHN_dataset_folder_path, "test_32x32.mat")

    if reshape_enable==True:
        test_data = preprc.reshape_image(test_data, (64, 64, 3))    

    # Preprocess training & validation data
    test_data, _, _ = preprc.normalize(test_data, mean=[], std=[])
    test_label = preprc.one_hot_encode(test_label)

    # Save original test data
    pickle.dump((np.array(test_data), np.array(test_label)), open(os.path.join(output_path, 'test.p'), 'wb'))
      
    for reserved_class in range(10):
        
        print("[Testing data] Extracting No.{} Class...".format(reserved_class))

        curr_features = test_data[test_label[:, reserved_class] == 1]
        curr_lables = test_label[test_label[:, reserved_class] == 1]
    
        print("\t[After] feature shape: ", np.shape(curr_features))
        print(np.min(curr_features))
        print(np.max(curr_features))
        # Save test data
        pickle.dump((np.array(curr_features), np.array(curr_lables)), open(os.path.join(output_path, 'pr_test_class_{}.p'.format(reserved_class)), 'wb'))
        
# -----------------------------------------------------------------------------

SVHN_dataset_folder_path = "/data/wei/dataset/FSL/SVHN/"
output_path = "/data/wei/dataset/FSL/SVHN/preprocessed/"

for i in range(0, 10):
    preprocess_and_save_data(SVHN_dataset_folder_path, output_path, rm_class=i)

#preprocess_and_save_single_class_data(SVHN_dataset_folder_path, output_path)