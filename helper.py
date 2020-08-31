import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import os
import download_cifar_10 as DL
import preprocess as preprc

# cifar-10
#mean = np.array([125.3, 123.0, 113.9])
#std = np.array([63.0, 62.1, 66.7])

mean = []
std = []

def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    """
    Display Stats of the the dataset
    """
    batch_ids = list(range(1, 6))

    if batch_id not in batch_ids:
        print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
        return None

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch {}:'.format(batch_id))
    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    label_names = _load_label_names()

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.axis('off')
    plt.imshow(sample_image)


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features, m, sd = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, output_path, rm_class, aug_enable, reshape_enable):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    features = []
    labels = []
    for batch_i in range(1, n_batches + 1):
        curr_features, curr_labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        
        if len(features) is 0:
            features = curr_features
            labels = curr_labels
        else:
            features = np.concatenate((features, curr_features))
            labels = np.concatenate((labels, curr_labels))
    
    # Preprocess training & validation data
    if aug_enable==True:
        features_ud = preprc.vertical_flip(features)
        features_lr = preprc.horizontal_flip(features)
        features_rot90 = preprc.rot90(features)
        features_rot270 = preprc.rot270(features)
        features = np.concatenate((features, features_ud, features_lr, features_rot90, features_rot270))
        labels = np.concatenate((labels, labels, labels, labels, labels))

    if reshape_enable==True:
        features = preprc.reshape_image(features, (64, 64, 3))    

    features, _, _ = preprc.normalize(features, mean=mean, std=std)
    labels = preprc.one_hot_encode(labels)
          
    print("[Training data] Removing No.{} Class...".format(rm_class))
    print("\t[Before] feature shape: ", np.shape(features))
    print("\t[Before] label shape: ", np.shape(labels))    
    count = 0
    remove_class = []
    for i in range(len(features)):
        if labels[i, rm_class] == 1:
            count = count + 1
            remove_class.append(i)
    print("\tCount: {}".format(count))            
    features = np.delete(features, remove_class, axis=0)
    labels = np.delete(labels, remove_class, axis=0)

    print("\t[After] feature shape: ", np.shape(features))
    print("\t[After] label shape: ", np.shape(labels))
    
    # Save training data
    pickle.dump((features, labels), open(os.path.join(output_path, 'preprocess_train_{}.p'.format(rm_class)), 'wb'), protocol=4)

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    if reshape_enable==True:
        test_features = preprc.reshape_image(test_features, (64, 64, 3))    

    # Preprocess training & validation data
    test_features, _, _ = preprc.normalize(test_features, mean=mean, std=std)
    test_labels = preprc.one_hot_encode(test_labels)

    # Save original test data
    pickle.dump((np.array(test_features), np.array(test_labels)), open(os.path.join(output_path, 'test.p'), 'wb'))
      
    print("[Testing data] Removing No.{} Class...".format(rm_class))
    print("\t[Before] feature shape: ", np.shape(test_features))
    print("\t[Before] label shape: ", np.shape(test_labels))    
    count = 0
    remove_class = []
    for i in range(len(test_features)):
        if test_labels[i, rm_class] == 1:
            count = count + 1
            remove_class.append(i)
    print("\tCount: {}".format(count))            
    test_features = np.delete(test_features, remove_class, axis=0)
    test_labels = np.delete(test_labels, remove_class, axis=0)

    print("\t[After] feature shape: ", np.shape(test_features))
    print("\t[After] label shape: ", np.shape(test_labels))

    # Save test data
    pickle.dump((np.array(test_features), np.array(test_labels)), open(os.path.join(output_path, 'preprocess_test_{}.p'.format(rm_class)), 'wb'), protocol=4)

def preprocess_and_save_single_class_data(cifar10_dataset_folder_path, output_path, aug_enable, reshape_enable):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    features = []
    labels = []
    for batch_i in range(1, n_batches + 1):
        curr_features, curr_labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        
        if len(features) is 0:
            features = curr_features
            labels = curr_labels
        else:
            features = np.concatenate((features, curr_features))
            labels = np.concatenate((labels, curr_labels))
    
    # Preprocess training & validation data
    if aug_enable==True:
        features_ud = preprc.vertical_flip(features)
        features_lr = preprc.horizontal_flip(features)
        features_rot90 = preprc.rot90(features)
        features_rot270 = preprc.rot270(features)
        features = np.concatenate((features, features_ud, features_lr, features_rot90, features_rot270))
        labels = np.concatenate((labels, labels, labels, labels, labels))

    if reshape_enable==True:
        features = preprc.reshape_image(features, (64, 64, 3)) 

    features, _, _ = preprc.normalize(features, mean=mean, std=std)
    labels = preprc.one_hot_encode(labels)
    
    for reserved_class in range(10):
          
        print("[Training data] Extracting No.{} Class...".format(reserved_class))

        curr_features = features[labels[:, reserved_class] == 1]
        curr_lables = labels[labels[:, reserved_class] == 1]
    
        print("\t[Class {}] feature shape: ".format(reserved_class), np.shape(curr_features))
        
        # Save training data
        pickle.dump((curr_features, curr_lables), open(os.path.join(output_path, 'pr_train_class_{}.p'.format(reserved_class)), 'wb'))

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    if reshape_enable==True:
        test_features = preprc.reshape_image(test_features, (64, 64, 3))    

    # Preprocess training & validation data
    test_features, _, _ = preprc.normalize(test_features, mean=mean, std=std)
    test_labels = preprc.one_hot_encode(test_labels)

    # Save original test data
    pickle.dump((np.array(test_features), np.array(test_labels)), open(os.path.join(output_path, 'test.p'), 'wb'))
      
    for reserved_class in range(10):
        
        print("[Testing data] Extracting No.{} Class...".format(reserved_class))

        curr_features = test_features[test_labels[:, reserved_class] == 1]
        curr_lables = test_labels[test_labels[:, reserved_class] == 1]
    
        print("\t[After] feature shape: ", np.shape(curr_features))
    
        # Save test data
        pickle.dump((np.array(curr_features), np.array(curr_lables)), open(os.path.join(output_path, 'pr_test_class_{}.p'.format(reserved_class)), 'wb'))
    
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    
    return features, labels

def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = _load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]

        axies[image_i][0].imshow(feature)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
        
# -----------------------------------------------------------------------------

cifar10_dataset_folder_path = "/data/wei/dataset/FSL/Cifar-10/cifar-10-batches-py"
tar_gz_path = "/data/wei/dataset/FSL/Cifar-10/cifar-10-python.tar.gz"
single_class_output_path = "/data/wei/dataset/FSL/Cifar-10/pr_single_class_aug/temp/"
all_class_output_path = "/data/wei/dataset/FSL/Cifar-10/preprocessed_aug/temp/"

# Download the CIFAR-10 dataset if not exist.
DL.process(cifar10_dataset_folder_path, tar_gz_path)

preprocess_and_save_single_class_data(cifar10_dataset_folder_path, single_class_output_path, aug_enable=True, reshape_enable=False)

for i in range(0, 10):
    # Preprocess Training, Validation, and Testing Data
    preprocess_and_save_data(cifar10_dataset_folder_path, all_class_output_path, i, aug_enable=True, reshape_enable=False)        
