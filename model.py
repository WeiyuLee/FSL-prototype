# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('./utility')

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import scipy.misc
import random

from utils import log10, get_batch
import netfactory as nf

from keras.datasets import mnist
import cv2

import model_zoo

from tensorflow.examples.tutorials.mnist import input_data

class model(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, dropout, model_path, data_ob, log_dir, learnrate_init, anoCls,
                 ckpt_name, test_ckpt, train_ckpt=[], restore_model=False, restore_step=0, class_num=10, model_ticket="none", is_training=True):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.repeat_num = repeat
        self.dropout = dropout
        self.class_num = class_num
        self.saved_model_path = model_path
        self.data_ob = data_ob
        self.log_dir = log_dir
        self.learn_rate_init = learnrate_init
        self.anoCls = anoCls
        self.ckpt_name = ckpt_name
        self.test_ckpt = test_ckpt
        self.train_ckpt = train_ckpt
        self.restore_model = restore_model
        self.restore_step = restore_step
        self.log_vars = []
        self.log_imgs = []
        self.model_ticket = model_ticket
        self.is_training = is_training
                
        self.channel = 3
        self.output_size = data_ob.image_size
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel], name='input1')
        self.images2 = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel], name='input2')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.class_num], name='label')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout')
        self.delta = tf.placeholder(tf.float32, name='delta')
        self.mean_weight = tf.placeholder(tf.float32, [self.batch_size, 256])
        
        self.ATT_START_STEP = 1000
        
#        if self.is_training == True:
#            
#            ## Training set
#            self.dataset = tf.data.Dataset.from_tensor_slices(self.data_ob.train_data_list)       
#            #self.dataset = self.dataset.repeat(self.repeat_num)
#            self.dataset = self.dataset.shuffle(buffer_size=10000).repeat(self.repeat_num)
#            self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
#    
#            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
#            self.next_x = self.iterator.get_next()
#            self.training_init_op = self.iterator.make_initializer(self.dataset)
#    
#            ## Validation set
#            self.valid_dataset = tf.data.Dataset.from_tensor_slices(self.data_ob.valid_data_list)
#            self.valid_dataset = self.valid_dataset.repeat(None)
#            self.valid_dataset = self.valid_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
#    
#            self.valid_iterator = tf.data.Iterator.from_structure(self.valid_dataset.output_types, self.valid_dataset.output_shapes)
#            self.next_valid_x = self.valid_iterator.get_next()
#            self.valid_init_op = self.valid_iterator.make_initializer(self.valid_dataset)
#            
#        else:
#            
#            ## Testing set
#            self.test_dataset = tf.data.Dataset.from_tensor_slices(self.data_ob.test_data_list)
#            self.test_dataset = self.test_dataset.repeat(1)
#            self.test_dataset = self.test_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
#    
#            self.test_iterator = tf.data.Iterator.from_structure(self.test_dataset.output_types, self.test_dataset.output_shapes)
#            self.next_test_x = self.test_iterator.get_next()
#            self.testing_init_op = self.test_iterator.make_initializer(self.test_dataset)
        
        if self.is_training == True:
            
            ## Training set
            #self.dataset = self.data_ob.train_data_list          
            self.dataset = self.data_ob.train_data_dict         
    
            ## Validation set
            #self.valid_dataset = self.data_ob.valid_data_list
            self.valid_dataset = self.data_ob.valid_data_dict

            ## Anomaly set
            self.anomaly_dataset = self.data_ob.anomaly_data_list
            
        else:
            
            ## Testing set
            self.test_dataset = self.data_ob.test_data_list
            
        self.model_list = ["cifar10_alexnet_att", "AD_att_GAN", "AD_att_GAN_v2", "AD_att_VAE", "AD_att_VAE_WEAK", "AD_att_VAE_GAN", "AD_att_AE_GAN", 
                           "AD_att_AE_GAN_3DCode", "AD_att_AE_GAN_3DCode_32x32", 
                           "AD_att_AE_GAN_CLS",
                           "GANomaly", "RaGAN_MNIST"]

    def build_model(self):###              
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            print("Model name: {}".format(self.model_ticket))
            fn = getattr(self, "build_" + self.model_ticket)
            model = fn()
            return model    
        
    def build_eval_model(self):###              
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            print("Model name: {}".format(self.model_ticket))
            fn = getattr(self, "build_eval_" + self.model_ticket)
            model = fn()
            return model 
        
    def train(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "train_" + self.model_ticket)
            function = fn()
            return function 

    def test(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            print("Model name: {}".format(self.model_ticket))
            fn = getattr(self, "test_" + self.model_ticket)
            function = fn()
            return function 

    def build_cifar10_alexnet_att(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=True, model_ticket=self.model_ticket)
        
        logits, self.last_layer, conv1_att, conv2_att, conv3_att, conv4_att, conv5_att = mz.build_model({"reuse":False, "is_training":True})
        
        # Attention weighting
        self.att_weight = tf.concat([conv1_att, conv2_att, conv3_att, conv4_att, conv5_att], 3)
        self.att_weight = tf.squeeze(self.att_weight)
        #self.att_weight = tf.nn.softmax(self.att_weight)

        self.entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        
        # Total loss
        self.total_loss = self.entropy_loss       

        self.pred = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(self.pred, tf.argmax(self.labels, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('train_summary'):
            tf.summary.scalar("Total_loss", self.total_loss, collections=['train'])
            tf.summary.scalar("Acc.", self.acc, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("Total_loss", self.total_loss, collections=['test'])
            tf.summary.scalar("Acc.", self.acc, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')          

        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()
            
    def train_cifar10_alexnet_att(self):
       
        # Preprocess Training, Validation, and Testing Data
        #helper.preprocess_and_save_data(cifar10_dataset_folder_path, preprc.normalize, preprc.one_hot_encode)
        
        global_step = tf.Variable(self.restore_step, trainable=False)
        add_global = global_step.assign_add(1)               
        new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=10000, decay_rate=0.98)                

        optimizer = tf.train.AdamOptimizer(new_learning_rate, beta1=0.5, beta2=0.9).minimize(self.total_loss)
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_acc = 0

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            sess.run(self.training_init_op)
            sess.run(self.testing_init_op)

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0
                               
            while step <= self.max_iters:
                                                 
                next_x_images, next_y = sess.run(self.next_x)
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout: 0.75,
                      }
                
                next_test_x_images, next_test_y = sess.run(self.next_test_x)
                fd_test = {
                        self.images: next_test_x_images, 
                        self.labels: next_test_y, 
                        self.dropout: 1.,
                      }
                
                # optimization AE
                sess.run(optimizer, feed_dict=fd)

                #summary_writer.add_summary(summary_str, step)
                new_learn_rate = sess.run(new_learning_rate)

                if new_learn_rate > 0.00005:
                    sess.run(add_global)

                if step%200 == 0:

                    train_sum, train_loss, train_acc, new_learn_rate = sess.run([self.merged_summary_train, self.total_loss, self.acc, new_learning_rate], feed_dict=fd)
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.total_loss, self.acc], feed_dict=fd_test)
                    print("Step %d: Train = [%.7f, %.7f], Test = [%.7f, %.7f], LR=%.7f" % (step, train_loss, train_acc, test_loss, test_acc, new_learn_rate))

                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if best_acc < test_acc and test_acc > 0.5:
                        best_acc = test_acc
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_{}'.format(best_acc))
                        print("* Save ckpt: {}, Test Acc.: {}".format(ckpt_path, best_acc))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)

    def test_cifar10_alexnet_att(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
            
            next_x_images, next_test_y = sess.run(self.next_test_x)

            pred, att_weight, last_layer = sess.run([self.pred, self.att_weight, self.last_layer], feed_dict={self.images: next_x_images, self.labels: next_test_y})
            pred = np.expand_dims(pred, 1)
            
            next_test_y = np.argmax(next_test_y, axis=-1)
            next_test_y = np.expand_dims(next_test_y, 1)
            
            output = np.concatenate((next_test_y, pred, att_weight, last_layer), axis=1)
            print(np.shape(output))
            np.savetxt("./output.csv", output, delimiter=",")

    def build_AD_att_GAN(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        self.last_layer, self.att_weight = mz.build_model({"inputs":self.images, "reuse":False, "net":"Gen"})

        code_attention = tf.concat([self.last_layer, self.att_weight], -1)   

        self.cls_inputs = code_attention
        cls_logits = mz.build_model({"inputs":self.cls_inputs, "reuse":False, "net":"Cls"})

        self.dis_t_inputs = code_attention
        self.att_f_weight = tf.gather(self.att_weight, tf.random_shuffle(tf.range(tf.shape(self.att_weight)[0])))
        self.dis_f_inputs = tf.concat([self.last_layer, self.att_f_weight], -1) 
        dis_t = mz.build_model({"inputs":self.dis_t_inputs, "reuse":False, "net":"Dis"})
        dis_f = mz.build_model({"inputs":self.dis_f_inputs, "reuse":True, "net":"Dis"})         

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat = mz.build_model({"inputs":x_hat, "reuse":True, "net":"Dis"})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        self.disc_ture_loss = tf.reduce_mean(dis_t)
        self.disc_fake_loss = tf.reduce_mean(dis_f)
                
        self.entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_logits, labels=self.labels))
     
        print("======================")
        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        print("======================")        

        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)                
        
        # Total loss     
        self.d_loss = (self.disc_fake_loss - self.disc_ture_loss) + d_gp
        self.c_loss = self.entropy_loss# + self.reg_set_l2_loss
        self.g_loss = self.delta*(-1.0*self.disc_fake_loss) + self.c_loss 
        self.d_loss_keepdim = (dis_f - dis_t) 
        
        self.pred = tf.argmax(cls_logits, axis=1)
        correct_prediction = tf.equal(self.pred, tf.argmax(self.labels, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('train_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("c_loss", self.c_loss, collections=['train'])
            tf.summary.scalar("Acc.", self.acc, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("c_loss", self.c_loss, collections=['test'])
            tf.summary.scalar("Acc.", self.acc, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')          

        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def train_AD_att_GAN(self):
        
        new_learning_rate = self.learn_rate_init

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("AD_Gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("AD_Dis")]
        classifier_variables = [v for v in train_variables if v.name.startswith("AD_Cls")]
        
        self.train_g = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)        
        self.train_d = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_c = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.c_loss, var_list=classifier_variables)
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                     }
                               
                # Training Generator
                sess.run(self.train_g, feed_dict=fd)
                sess.run(self.train_c, feed_dict=fd)                
                
                if step > 1000:
                    
                    delta = 1e-1
                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch
                        random.shuffle(dataset_idx)
                        next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                        next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                                
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.delta: delta,
                             }
                        
                        sess.run(self.train_d, feed_dict=fd)                    

                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    train_sum, train_gloss, train_dloss, train_closs, train_acc, disc_ture_loss, disc_fake_loss = sess.run([self.merged_summary_train, self.g_loss, self.d_loss, self.c_loss, self.acc, self.disc_ture_loss, self.disc_fake_loss], feed_dict=fd)
                    
                    test_gloss = 0
                    test_dloss = 0
                    test_closs = 0
                    test_acc = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                  }
                                               
                        test_sum, temp_gloss, temp_dloss, temp_closs, temp_acc = sess.run([self.merged_summary_test, self.g_loss, self.d_loss, self.c_loss, self.acc], feed_dict=fd_test)
                        
                        test_gloss = test_gloss + temp_gloss
                        test_dloss = test_dloss + temp_dloss
                        test_closs = test_closs + temp_closs
                        test_acc = test_acc + temp_acc

                    test_gloss = test_gloss / test_count           
                    test_dloss = test_dloss / test_count
                    test_closs = test_closs / test_count
                    test_acc = test_acc / test_count

                    print("Step %d: LR = [%.7f] Delta = [%f]" % (step, new_learning_rate, delta*1e8))
                    print("Train[G,D,C] = [%.7f, %.7f, %.7f] Acc = [%.7f] dis_t [%.7f] dis_f [%.7f]" % 
                          (train_gloss, train_dloss, train_closs, train_acc, disc_ture_loss, disc_fake_loss))
                    print("Test[G,D,C]  = [%.7f, %.7f, %.7f] Acc = [%.7f]" % 
                          (test_gloss, test_dloss, test_closs, test_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if best_loss > test_gloss:
                        
                        best_loss = test_gloss
                        best_acc = test_acc
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f_%.4f' % (best_loss, best_acc))
                        print("* Save ckpt: {}, Test loss: {}, Test Acc.: {}".format(ckpt_path, best_loss, best_acc))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}, Best acc.: {}".format(best_loss, best_acc))

    def test_AD_att_GAN(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        output_name = os.path.basename(self.data_ob.test_images_path)
        output_name = os.path.splitext(output_name)[0] + ".csv"
        print("Output: {}".format(output_name))
        
        output = np.array([])

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                
                pred, d_loss, last_layer, att_weight = sess.run([self.pred, self.d_loss_keepdim, self.last_layer, self.att_weight], feed_dict={self.images: next_x_images, self.labels: next_test_y})
                pred = np.expand_dims(pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)

                curr_output = np.concatenate((next_test_y, pred, d_loss, last_layer, att_weight), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_GAN_v2(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        self.last_layer, self.att_weight = mz.build_model({"inputs":self.images, "reuse":False, "net":"Gen"})

        code_attention = tf.concat([self.last_layer, self.att_weight], -1)   

        self.cls_inputs = self.att_weight
        cls_logits = mz.build_model({"inputs":self.cls_inputs, "reuse":False, "net":"Cls"})

        half_size = tf.cast(tf.shape(self.att_weight)[0]/2, dtype=tf.int32)
        self.att_f = tf.gather(self.att_weight, tf.random_shuffle(tf.range(half_size)))
        self.code_f = tf.gather(self.last_layer, tf.range(half_size))
        
        self.dis_inputs_1 = tf.concat([self.code_f, self.att_f], -1)
        self.dis_inputs_2 = tf.gather(code_attention, tf.range(half_size,tf.shape(self.att_weight)[0]))
        self.dis_inputs = tf.concat([self.dis_inputs_1, self.dis_inputs_2], 0)

        dis_logits = mz.build_model({"inputs":self.dis_inputs, "reuse":False, "net":"Dis"})
        dis_labels_1 = tf.concat([tf.zeros([half_size, 1]), tf.ones([half_size, 1])], -1)
        dis_labels_2 = tf.concat([tf.ones([half_size, 1]), tf.zeros([half_size, 1])], -1)
        dis_labels = tf.concat([dis_labels_1, dis_labels_2], 0)
        
        self.dis_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_logits, labels=dis_labels))        
        self.cls_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_logits, labels=self.labels))
     
        print("======================")
        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        print("======================")        

        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)                
        
        # Total loss     
        self.d_loss = self.dis_entropy_loss
        self.c_loss = self.cls_entropy_loss
        self.g_loss = self.d_loss + self.c_loss
        
        self.cls_pred = tf.argmax(cls_logits, axis=1)
        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))

        self.dis_pred = tf.argmax(dis_logits, axis=1)
        self.dis_pred_raw = dis_logits
        self.dis_labels = dis_labels
        dis_correct_pred = tf.equal(self.dis_pred, tf.argmax(dis_labels, axis=1))
        self.dis_acc = tf.reduce_mean(tf.cast(dis_correct_pred, tf.float32))

        with tf.name_scope('train_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("c_loss", self.c_loss, collections=['train'])
            tf.summary.scalar("Cls_Acc.", self.cls_acc, collections=['train'])
            tf.summary.scalar("Dis_Acc.", self.dis_acc, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("c_loss", self.c_loss, collections=['test'])
            tf.summary.scalar("Cls_Acc.", self.cls_acc, collections=['test'])
            tf.summary.scalar("Dis_Acc.", self.dis_acc, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')          

        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def build_eval_AD_att_GAN_v2(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        self.last_layer, self.att_weight = mz.build_model({"inputs":self.images, "reuse":False, "net":"Gen"})

        code_attention = tf.concat([self.last_layer, self.att_weight], -1)   

        self.cls_inputs = self.att_weight
        cls_logits = mz.build_model({"inputs":self.cls_inputs, "reuse":False, "net":"Cls"})

        self.dis_inputs = code_attention
        dis_logits = mz.build_model({"inputs":self.dis_inputs, "reuse":False, "net":"Dis"})
                     
        self.cls_pred = tf.argmax(cls_logits, axis=1)
        self.dis_pred = tf.argmax(dis_logits, axis=1)
        
        self.saver = tf.train.Saver()
        
    def train_AD_att_GAN_v2(self):
        
        new_learning_rate = self.learn_rate_init
       
        self.train_g = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.g_loss)        
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                     }
                               
                # Training
                sess.run(self.train_g, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    train_sum, train_gloss, train_dloss, train_closs, train_cls_acc, train_dis_acc = sess.run([self.merged_summary_train, self.g_loss, self.d_loss, self.c_loss, self.cls_acc, self.dis_acc], feed_dict=fd)

                    test_gloss = 0
                    test_dloss = 0
                    test_closs = 0
                    test_cls_acc = 0
                    test_dis_acc = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                  }
                                               
                        test_sum, temp_gloss, temp_dloss, temp_closs, temp_cls_acc, temp_dis_acc = sess.run([self.merged_summary_test, self.g_loss, self.d_loss, self.c_loss, self.cls_acc, self.dis_acc], feed_dict=fd_test)
                        
                        test_gloss = test_gloss + temp_gloss
                        test_dloss = test_dloss + temp_dloss
                        test_closs = test_closs + temp_closs
                        test_cls_acc = test_cls_acc + temp_cls_acc
                        test_dis_acc = test_dis_acc + temp_dis_acc

                    test_gloss = test_gloss / test_count           
                    test_dloss = test_dloss / test_count
                    test_closs = test_closs / test_count
                    test_cls_acc = test_cls_acc / test_count
                    test_dis_acc = test_dis_acc / test_count

                    print("Step %d: LR = [%.7f]" % (step, new_learning_rate))
                    print("Train[G,D,C] = [%.7f, %.7f, %.7f] Cls_Acc = [%.7f] Dis_Acc = [%.7f]" % 
                          (train_gloss, train_dloss, train_closs, train_cls_acc, train_dis_acc))
                    print("Test[G,D,C]  = [%.7f, %.7f, %.7f] Cls_Acc = [%.7f] Dis_Acc = [%.7f]" % 
                          (test_gloss, test_dloss, test_closs, test_cls_acc, test_dis_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if best_loss > test_gloss:
                        
                        best_loss = test_gloss
                        best_acc = test_cls_acc
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f_%.4f' % (best_loss, best_acc))
                        print("* Save ckpt: {}, Test loss: {}, Test Acc.: {}".format(ckpt_path, best_loss, best_acc))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}, Best acc.: {}".format(best_loss, best_acc))

    def test_AD_att_GAN_v2(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        output_name = os.path.basename(self.data_ob.test_images_path)
        output_name = os.path.splitext(output_name)[0] + ".csv"
        print("Output: {}".format(output_name))
        
        output = np.array([])

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                
                cls_pred, dis_pred, last_layer, att_weight = sess.run([self.cls_pred, self.dis_pred, self.last_layer, self.att_weight], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
                cls_pred = np.expand_dims(cls_pred, 1)
                dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)

                curr_output = np.concatenate((next_test_y, cls_pred, dis_pred, last_layer, att_weight), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_VAE(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_en_mu, self.t_en_sigma, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code = self.t_en_mu + tf.exp(0.5*self.t_en_sigma) * tf.random_normal(tf.shape(self.t_en_mu), dtype=tf.float32)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
               
        # Discriminator
        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)
        
        self.dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
        self.dis_f_inputs_1 = tf.concat([self.f_code, self.f_att], -1)
        self.dis_f_inputs_2 = tf.concat([self.t_code, self.f_att], -1)
        self.dis_f_inputs_3 = tf.concat([self.f_code, self.t_att], -1)
        
        self.dis_f_inputs_1 = tf.gather(self.dis_f_inputs_1, tf.random_shuffle(tf.range(self.batch_size//3)))
        self.dis_f_inputs_2 = tf.gather(self.dis_f_inputs_2, tf.random_shuffle(tf.range(self.batch_size//3)))
        self.dis_f_inputs_3 = tf.gather(self.dis_f_inputs_3, tf.random_shuffle(tf.range(self.batch_size//3 + self.batch_size%3)))
        self.dis_f_inputs = tf.concat([self.dis_f_inputs_1, self.dis_f_inputs_2, self.dis_f_inputs_3], 0)

        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp       

        # Classifier
        self.cls_inputs = tf.concat([self.t_code, self.t_att], -1)##########################################
        self.cls_output = mz.build_model({"mode":"classifier", "cls_input":self.cls_inputs, "reuse":False})      

        marginal_likelihood = tf.reduce_sum(tf.squared_difference(self.decoder_output, self.images))
        KL_divergence = -0.5 * tf.reduce_sum(1 + self.t_en_sigma - tf.pow(self.t_en_mu, 2) - tf.exp(self.t_en_sigma))        

        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_output, labels=self.labels))
        self.cls_pred = tf.argmax(self.cls_output, axis=1)
        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))

        #self.loss = (marginal_likelihood + KL_divergence) / tf.cast(tf.shape(self.images)[0], tf.float32) + self.delta*(-1.0*disc_fake_loss) + 10*self.cls_loss
        self.loss = (marginal_likelihood + KL_divergence)# + self.delta*disc_fake_loss + 10*self.cls_loss
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['train'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['train'])
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['test'])
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['test'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['test'])
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['test'])
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_VAE(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_en_mu, self.t_en_sigma, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code = self.t_en_mu + tf.exp(0.5*self.t_en_sigma) * tf.random_normal(tf.shape(self.t_en_mu), dtype=tf.float32)

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Discriminator
        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)   
        
        self.dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
        self.dis_f_inputs = tf.concat([self.f_code, self.f_att], -1)
        self.dis_t_output = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        self.dis_f_output = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       

        self.saver = tf.train.Saver()

    def train_AD_att_VAE(self):
        
        new_learning_rate = self.learn_rate_init
       
        train_variables = tf.trainable_variables()
        discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        vae_variables = [v for v in train_variables if v.name.startswith(("encoder", "decoder", "classifier"))]
        #cls_variables = [v for v in train_variables if v.name.startswith(("classifier"))]
        
        self.train_g = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=vae_variables)
        self.train_d = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        #self.train_c = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.cls_loss)
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     

                if step > 0:
                    delta = 1.0
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                     }
                               
                # Training
                sess.run(self.train_g, feed_dict=fd)

                if step > 10000:
                    # Training Discriminator
                    for d_iter in range(0, 1):
                        
                        # Get the training batch
                        random.shuffle(dataset_idx)
                        next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                        next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                                
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.delta: delta,
                             }                      

                        sess.run(self.train_d, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    train_sum, train_loss, train_cls_acc = sess.run([self.merged_summary_train, self.loss, self.cls_acc], feed_dict=fd)

                    test_loss = 0
                    test_cls_acc = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                  }
                                               
                        test_sum, temp_loss, temp_cls_acc = sess.run([self.merged_summary_test, self.loss, self.cls_acc], feed_dict=fd_test)
                        
                        test_loss = test_loss + temp_loss
                        test_cls_acc = test_cls_acc + temp_cls_acc

                    test_loss = test_loss / test_count          
                    test_cls_acc = test_cls_acc / test_count

                    print("Step %d: LR = [%.7f], Train loss = [%.7f] Cls acc = [%.7f], Test loss = [%.7f] Cls acc = [%.7f]" % (step, new_learning_rate, train_loss, train_cls_acc, test_loss, test_cls_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_VAE(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            for i in range(2):      

                output_name = os.path.basename(self.data_ob.test_images_path)
                output_name = os.path.splitext(output_name)[0] + str(i) + ".csv"
                print("Output: {}".format(output_name))
                
                output = np.array([])
                
                curr_idx = 0
                while True:
                    
                    try:
                        #next_x_images, next_test_y = sess.run(self.next_test_x)
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        curr_idx = curr_idx + self.batch_size
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    if i == 0:
                        dis_pred, last_layer, att_weight = sess.run([self.dis_f_output, self.f_code, self.f_att], 
                                                                              feed_dict=
                                                                                          {self.images: next_x_images, 
                                                                                           self.labels: next_test_y, 
                                                                                           self.dropout_rate: 0})
                    elif i == 1:
                        dis_pred, last_layer, att_weight = sess.run([self.dis_t_output, self.t_code, self.t_att], 
                                                                              feed_dict=
                                                                                          {self.images: next_x_images, 
                                                                                           self.labels: next_test_y, 
                                                                                           self.dropout_rate: 0})
        
                    #dis_pred = np.expand_dims(dis_pred, 1)
                    
                    next_test_y = np.argmax(next_test_y, axis=-1)
                    next_test_y = np.expand_dims(next_test_y, 1)
                    
                    curr_output = np.concatenate((next_test_y, dis_pred, last_layer, att_weight), axis=1)
                    output = np.vstack([output, curr_output]) if output.size else curr_output
                    
                print(np.shape(output))
                np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_VAE_WEAK(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_en_mu, self.t_en_sigma, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code = self.t_en_mu + tf.exp(0.5*self.t_en_sigma) * tf.random_normal(tf.shape(self.t_en_mu), dtype=tf.float32)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
               
        # Discriminator
        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)
        
        self.dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
        self.dis_f_inputs_1 = tf.concat([self.f_code, self.f_att], -1)
        #self.dis_f_inputs_2 = tf.concat([self.t_code, self.f_att], -1)
        #self.dis_f_inputs_3 = tf.concat([self.f_code, self.t_att], -1)
        
        #self.dis_f_inputs_1 = tf.gather(self.dis_f_inputs_1, tf.random_shuffle(tf.range(self.batch_size//3)))
        #self.dis_f_inputs_2 = tf.gather(self.dis_f_inputs_2, tf.random_shuffle(tf.range(self.batch_size//3)))
        #self.dis_f_inputs_3 = tf.gather(self.dis_f_inputs_3, tf.random_shuffle(tf.range(self.batch_size//3 + self.batch_size%3)))
        #self.dis_f_inputs = tf.concat([self.dis_f_inputs_1, self.dis_f_inputs_2, self.dis_f_inputs_3], 0)

        self.dis_f_inputs = self.dis_f_inputs_1

        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp       

        # Classifier
        #self.cls_inputs = tf.concat([self.t_code, self.t_att], -1)
        #self.cls_output = mz.build_model({"mode":"classifier", "cls_input":self.cls_inputs, "reuse":False})      

        # Reconstruct Classifier
        self.cls_output = mz.build_model({"mode":"reconstruct_cls", "cls_input":self.decoder_output, "reuse":False})      
        
        #marginal_likelihood = tf.reduce_sum(tf.squared_difference(self.decoder_output, self.images))
        marginal_likelihood = tf.reduce_sum(tf.abs(self.decoder_output - self.images))
        KL_divergence = -0.5 * tf.reduce_sum(1 + self.t_en_sigma - tf.pow(self.t_en_mu, 2) - tf.exp(self.t_en_sigma))        

        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_output, labels=self.labels))
        self.cls_pred = tf.argmax(self.cls_output, axis=1)
        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))

        #self.loss = (marginal_likelihood + KL_divergence) / tf.cast(tf.shape(self.images)[0], tf.float32) + self.delta*(-1.0*disc_fake_loss) + 10*self.cls_loss
        self.loss = (marginal_likelihood + KL_divergence) + self.delta*disc_fake_loss + 1*self.cls_loss
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['train'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['train'])
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['test'])
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['test'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['test'])
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['test'])
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_VAE_WEAK(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_en_mu, self.t_en_sigma, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code = self.t_en_mu + tf.exp(0.5*self.t_en_sigma) * tf.random_normal(tf.shape(self.t_en_mu), dtype=tf.float32)

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Discriminator
        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)   
        
        self.dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
        self.dis_f_inputs = tf.concat([self.f_code, self.f_att], -1)
        self.dis_t_output = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        self.dis_f_output = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       

        self.saver = tf.train.Saver()

    def train_AD_att_VAE_WEAK(self):
        
        new_learning_rate = self.learn_rate_init
       
        train_variables = tf.trainable_variables()
        discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        vae_variables = [v for v in train_variables if v.name.startswith(("encoder", "decoder", "reconstruct_cls"))]
        #cls_variables = [v for v in train_variables if v.name.startswith(("classifier"))]
        
        self.train_g = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=vae_variables)
        self.train_d = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        #self.train_c = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.cls_loss)
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     

                if step > 0:
                    delta = 1.0
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                     }
                               
                # Training
                sess.run(self.train_g, feed_dict=fd)
                
                if step >0:
                    sess.run(self.train_d, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    train_sum, train_loss, train_cls_acc = sess.run([self.merged_summary_train, self.loss, self.cls_acc], feed_dict=fd)

                    test_loss = 0
                    test_cls_acc = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                  }
                                               
                        test_sum, temp_loss, temp_cls_acc = sess.run([self.merged_summary_test, self.loss, self.cls_acc], feed_dict=fd_test)
                        
                        test_loss = test_loss + temp_loss
                        test_cls_acc = test_cls_acc + temp_cls_acc

                    test_loss = test_loss / test_count          
                    test_cls_acc = test_cls_acc / test_count

                    print("Step %d: LR = [%.7f], Train loss = [%.7f] Cls acc = [%.7f], Test loss = [%.7f] Cls acc = [%.7f]" % (step, new_learning_rate, train_loss, train_cls_acc, test_loss, test_cls_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_VAE_WEAK(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            for i in range(2):      

                output_name = os.path.basename(self.data_ob.test_images_path)
                output_name = os.path.splitext(output_name)[0] + str(i) + ".csv"
                print("Output: {}".format(output_name))
                
                output = np.array([])
                
                curr_idx = 0
                while True:
                    
                    try:
                        #next_x_images, next_test_y = sess.run(self.next_test_x)
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        curr_idx = curr_idx + self.batch_size
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    if i == 0:
                        dis_pred, last_layer, att_weight = sess.run([self.dis_f_output, self.f_code, self.f_att], 
                                                                              feed_dict=
                                                                                          {self.images: next_x_images, 
                                                                                           self.labels: next_test_y, 
                                                                                           self.dropout_rate: 0})
                    elif i == 1:
                        dis_pred, last_layer, att_weight = sess.run([self.dis_t_output, self.t_code, self.t_att], 
                                                                              feed_dict=
                                                                                          {self.images: next_x_images, 
                                                                                           self.labels: next_test_y, 
                                                                                           self.dropout_rate: 0})
        
                    #dis_pred = np.expand_dims(dis_pred, 1)
                    
                    next_test_y = np.argmax(next_test_y, axis=-1)
                    next_test_y = np.expand_dims(next_test_y, 1)
                    
                    curr_output = np.concatenate((next_test_y, dis_pred, last_layer, att_weight), axis=1)
                    output = np.vstack([output, curr_output]) if output.size else curr_output
                    
                print(np.shape(output))
                np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_VAE_GAN(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_en_mu, self.t_en_sigma, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code = self.t_en_mu + tf.exp(0.5*self.t_en_sigma) * tf.random_normal(tf.shape(self.t_en_mu), dtype=tf.float32)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # Discriminator ===================================================================================================================       
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp       

        # Code Discriminator ==============================================================================================================
#        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
#        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)
#        
#        self.code_dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
#        self.code_dis_f_inputs = tf.concat([self.f_code, self.f_att], -1)
#        code_dis_t = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_t_inputs, "reuse":False})       
#        code_dis_f = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_f_inputs, "reuse":True})       
#
#        #### WGAN-GP ####
#        # Calculate gradient penalty
#        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
#        code_x_hat = epsilon * self.code_dis_t_inputs + (1. - epsilon) * (self.code_dis_f_inputs)
#        code_d_hat = mz.build_model({"mode":"code_dis", "code_dis_input":code_x_hat, "reuse":True})
#        
#        code_d_gp = tf.gradients(code_d_hat, [code_x_hat])[0]
#        code_d_gp = tf.sqrt(tf.reduce_sum(tf.square(code_d_gp), axis=1))
#        code_d_gp = tf.reduce_mean((code_d_gp - 1.0)**2) * 10
#
#        code_disc_ture_loss = tf.reduce_mean(code_dis_t)
#        code_disc_fake_loss = tf.reduce_mean(code_dis_f)
#        
#        self.code_d_loss = -(code_disc_fake_loss - code_disc_ture_loss) + code_d_gp       
        
        # Code Discriminator RaGAN + GP ====================================================================================================
        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})     
        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)
        
        #self.code_dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
        #self.code_dis_f_inputs = tf.concat([self.f_code, self.f_att], -1)
        self.code_dis_t_inputs = self.t_code
        self.code_dis_f_inputs = self.f_code
        code_dis_t = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_t_inputs, "reuse":False})       
        code_dis_f = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        code_x_hat = epsilon * self.code_dis_t_inputs + (1. - epsilon) * (self.code_dis_f_inputs)
        code_d_hat = mz.build_model({"mode":"code_dis", "code_dis_input":code_x_hat, "reuse":True})
        
        code_d_gp = tf.gradients(code_d_hat, [code_x_hat])[0]
        code_d_gp = tf.sqrt(tf.reduce_sum(tf.square(code_d_gp), axis=1))
        code_d_gp = tf.reduce_mean((code_d_gp - 1.0)**2) * 10

        real_logit = (code_dis_t - tf.reduce_mean(code_dis_f))
        fake_logit = (code_dis_f - tf.reduce_mean(code_dis_t))

        code_disc_ture_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(code_dis_t), logits=real_logit))
        code_disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(code_dis_f), logits=fake_logit))
        
        self.code_d_loss = code_disc_fake_loss + code_disc_ture_loss + code_d_gp           

        code_gen_ture_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(code_dis_t), logits=real_logit))
        code_gen_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(code_dis_f), logits=fake_logit))
        
        self.code_g_loss = code_gen_fake_loss + code_gen_ture_loss 
        
        # ==================================================================================================================================
        
        marginal_likelihood = tf.reduce_sum(tf.abs(self.decoder_output - self.images))
        KL_divergence = -0.5 * tf.reduce_sum(1 + self.t_en_sigma - tf.pow(self.t_en_mu, 2) - tf.exp(self.t_en_sigma))   
        
        self.e2_loss = tf.reduce_mean(tf.pow(self.f_code - self.t_code, 2))# + self.code_g_loss
        self.vae_loss = (marginal_likelihood + KL_divergence) / tf.cast(tf.shape(self.images)[0], tf.float32) + (self.delta*disc_fake_loss) + self.e2_loss
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("vae_loss", self.vae_loss, collections=['train'])
            tf.summary.scalar("e2_loss", self.e2_loss, collections=['train'])
            tf.summary.scalar("code_dist", self.e2_loss-self.code_g_loss, collections=['train'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['train'])
            
            #tf.summary.scalar("code_d_loss", self.code_d_loss, collections=['train'])
            #tf.summary.scalar("code_g_loss", self.code_g_loss, collections=['train'])
            #tf.summary.scalar("code_disc_fake_loss", code_disc_fake_loss, collections=['train'])
            #tf.summary.scalar("code_disc_ture_loss", code_disc_ture_loss, collections=['train'])            
            #tf.summary.scalar("code_gen_fake_loss", code_disc_fake_loss, collections=['train'])
            #tf.summary.scalar("code_gen_ture_loss", code_disc_ture_loss, collections=['train'])     
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['train'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("vae_loss", self.vae_loss, collections=['test'])
            tf.summary.scalar("e2_loss", self.e2_loss, collections=['test'])
            tf.summary.scalar("code_dist", self.e2_loss-self.code_g_loss, collections=['test'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['test'])

            #tf.summary.scalar("code_d_loss", self.code_d_loss, collections=['test'])
            #tf.summary.scalar("code_g_loss", self.code_g_loss, collections=['test'])
            #tf.summary.scalar("code_disc_fake_loss", code_disc_fake_loss, collections=['test'])
            #tf.summary.scalar("code_disc_ture_loss", code_disc_ture_loss, collections=['test'])  
            #tf.summary.scalar("code_gen_fake_loss", code_disc_fake_loss, collections=['test'])
            #tf.summary.scalar("code_gen_ture_loss", code_disc_ture_loss, collections=['test'])                 
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['test'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):
            tf.summary.scalar("vae_loss", self.vae_loss, collections=['anomaly'])
            tf.summary.scalar("e2_loss", self.e2_loss, collections=['anomaly'])
            tf.summary.scalar("code_dist", self.e2_loss-self.code_g_loss, collections=['anomaly'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['anomaly'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['anomaly'])

            #tf.summary.scalar("code_d_loss", self.code_d_loss, collections=['anomaly'])
            #tf.summary.scalar("code_g_loss", self.code_g_loss, collections=['anomaly'])
            #tf.summary.scalar("code_disc_fake_loss", code_disc_fake_loss, collections=['anomaly'])
            #tf.summary.scalar("code_disc_ture_loss", code_disc_ture_loss, collections=['anomaly'])  
            #tf.summary.scalar("code_gen_fake_loss", code_disc_fake_loss, collections=['anomaly'])
            #tf.summary.scalar("code_gen_ture_loss", code_disc_ture_loss, collections=['anomaly'])                 
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['anomaly'])
            tf.summary.scalar("KL_divergence", tf.reduce_mean(KL_divergence), collections=['anomaly'])
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_VAE_GAN(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_en_mu, self.t_en_sigma, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code = self.t_en_mu + tf.exp(0.5*self.t_en_sigma) * tf.random_normal(tf.shape(self.t_en_mu), dtype=tf.float32)

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Discriminator
        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)   
        
        self.code_dis_inputs = tf.concat([self.f_code, self.f_att], -1)       
        self.code_dis_output = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_inputs, "reuse":False})       
        self.code_dis_output = tf.sigmoid(self.code_dis_output)

        self.saver = tf.train.Saver()

    def train_AD_att_VAE_GAN(self):
        
        new_learning_rate = self.learn_rate_init
       
        train_variables = tf.trainable_variables()
        discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        #code_dis_variables = [v for v in train_variables if v.name.startswith("code_dis")]
        vae_variables = [v for v in train_variables if v.name.startswith(("encoder", "decoder", "encoder2"))]
        #e2_variables = [v for v in train_variables if v.name.startswith(("encoder2"))]
        #cls_variables = [v for v in train_variables if v.name.startswith(("classifier"))]
        
        self.train_vae = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.vae_loss, var_list=vae_variables)
        self.train_d = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        #self.train_e2 = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.e2_loss, var_list=e2_variables)        
        #self.train_code_dis = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.code_d_loss, var_list=code_dis_variables)
        #self.train_c = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.cls_loss)
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))
            anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     

                if step > 0:
                    delta = 50.0
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                     }
                               
                # Training
                sess.run(self.train_vae, feed_dict=fd)
                #sess.run(self.train_e2, feed_dict=fd)
                
                if step > 0:
                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch
                        random.shuffle(dataset_idx)
                        next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                        next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                                
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.delta: delta,
                             }                      

                        sess.run(self.train_d, feed_dict=fd)
                        #sess.run(self.train_code_dis, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.vae_loss], feed_dict=fd)

                    # Validation set
                    test_loss = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                  }
                                               
                        test_sum, temp_loss = sess.run([self.merged_summary_test, self.vae_loss], feed_dict=fd_test)                       
                        test_loss = test_loss + temp_loss

                    test_loss = test_loss / test_count          

                    # Anomaly set
                    ano_loss = 0
                    ano_count = 0
                    curr_idx = 0
                    random.shuffle(anomaly_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.anomaly_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        ano_count = ano_count + 1

                        fd_test = {
                                    self.images: next_ano_x_images, 
                                    self.labels: next_ano_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                  }
                                               
                        ano_sum, temp_ano_loss = sess.run([self.merged_summary_anomaly, self.vae_loss], feed_dict=fd_test)                       
                        ano_loss = ano_loss + temp_ano_loss

                    ano_loss = ano_loss / ano_count    

                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_VAE_GAN(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            output_name = os.path.basename(self.data_ob.test_images_path)
            output_name = os.path.splitext(output_name)[0] + ".csv"
            print("Output: {}".format(output_name))
            
            output = np.array([])
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                

                dis_pred, last_layer, att_weight, decoder_output = sess.run([self.code_dis_output, self.f_code, self.f_att, self.decoder_output], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
    
                
                for idx in range(len(decoder_output)):
                    scipy.misc.imsave("./output_image/" + "decode_" + str(idx) + '.png', decoder_output[idx])
                    scipy.misc.imsave("./output_image/" + "encode_" + str(idx) + '.png', next_x_images[idx])
    
                #dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)
                
                curr_output = np.concatenate((next_test_y, dis_pred, last_layer, att_weight), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_AE_GAN(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_code, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # Discriminator ===================================================================================================================       
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp   
        #self.g_loss = disc_fake_loss + tf.reduce_mean(tf.pow((disc_fake_loss-disc_ture_loss),2))

        # Code Discriminator ==============================================================================================================
#        self.f_en_mu, self.f_en_sigma, self.f_att = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})     
#        self.f_code = self.f_en_mu + tf.exp(0.5*self.f_en_sigma) * tf.random_normal(tf.shape(self.f_en_mu), dtype=tf.float32)
#        
#        self.code_dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
#        self.code_dis_f_inputs = tf.concat([self.f_code, self.f_att], -1)
#        code_dis_t = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_t_inputs, "reuse":False})       
#        code_dis_f = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_f_inputs, "reuse":True})       
#
#        #### WGAN-GP ####
#        # Calculate gradient penalty
#        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
#        code_x_hat = epsilon * self.code_dis_t_inputs + (1. - epsilon) * (self.code_dis_f_inputs)
#        code_d_hat = mz.build_model({"mode":"code_dis", "code_dis_input":code_x_hat, "reuse":True})
#        
#        code_d_gp = tf.gradients(code_d_hat, [code_x_hat])[0]
#        code_d_gp = tf.sqrt(tf.reduce_sum(tf.square(code_d_gp), axis=1))
#        code_d_gp = tf.reduce_mean((code_d_gp - 1.0)**2) * 10
#
#        code_disc_ture_loss = tf.reduce_mean(code_dis_t)
#        code_disc_fake_loss = tf.reduce_mean(code_dis_f)
#        
#        self.code_d_loss = -(code_disc_fake_loss - code_disc_ture_loss) + code_d_gp       
        
        # Code Discriminator RaGAN + GP ====================================================================================================
        self.f_code, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})        
        
        #self.code_dis_t_inputs = tf.concat([self.t_code, self.t_att], -1)
        #self.code_dis_f_inputs = tf.concat([self.f_code, self.f_att], -1)
        
        self.code_dis_t_inputs = self.t_code       
        self.code_dis_f_inputs = self.f_code
        code_dis_t = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_t_inputs, "reuse":False})       
        code_dis_f = mz.build_model({"mode":"code_dis", "code_dis_input":self.code_dis_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        code_x_hat = epsilon * self.code_dis_t_inputs + (1. - epsilon) * (self.code_dis_f_inputs)
        code_d_hat = mz.build_model({"mode":"code_dis", "code_dis_input":code_x_hat, "reuse":True})
        
        code_d_gp = tf.gradients(code_d_hat, [code_x_hat])[0]
        code_d_gp = tf.sqrt(tf.reduce_sum(tf.square(code_d_gp), axis=1))
        code_d_gp = tf.reduce_mean((code_d_gp - 1.0)**2) * 10

        real_logit = (code_dis_t - tf.reduce_mean(code_dis_f))
        fake_logit = (code_dis_f - tf.reduce_mean(code_dis_t))

        code_disc_ture_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(code_dis_t), logits=real_logit))
        code_disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(code_dis_f), logits=fake_logit))
        
        self.code_d_loss = code_disc_fake_loss + code_disc_ture_loss + code_d_gp           

        code_gen_ture_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(code_dis_t), logits=real_logit))
        code_gen_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(code_dis_f), logits=fake_logit))
        
        self.code_g_loss = code_gen_fake_loss + code_gen_ture_loss 
        
        # Code Classifier ==================================================================================================================
               
        cls_logits = mz.build_model({"mode":"code_cls", "code_cls_input":self.t_code, "reuse":False})             
        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_logits, labels=self.labels))
        
        self.cls_pred = tf.argmax(cls_logits, axis=1)
        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))
        
        # ==================================================================================================================================
        marginal_likelihood = tf.reduce_mean(tf.abs(self.decoder_output - self.images))
       
        self.e2_loss = tf.reduce_mean(tf.pow(self.f_code - self.t_code, 2))# + self.code_g_loss
        self.vae_loss = 50*marginal_likelihood + self.e2_loss + (self.delta*disc_fake_loss) + (50*self.cls_loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        an_score = tf.reduce_mean(tf.abs(self.f_code-self.t_code))

        train_variables = tf.trainable_variables()
        discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        #code_dis_variables = [v for v in train_variables if v.name.startswith("code_dis")]
        vae_variables = [v for v in train_variables if v.name.startswith(("encoder", "decoder", "encoder2", "code_cls"))]
        #e2_variables = [v for v in train_variables if v.name.startswith(("encoder2"))]
        #cls_variables = [v for v in train_variables if v.name.startswith(("classifier"))]
        
        self.train_vae = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.vae_loss, var_list=vae_variables)
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        #self.train_e2 = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.e2_loss, var_list=e2_variables)        
        #self.train_code_dis = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.code_d_loss, var_list=code_dis_variables)
        #self.train_c = tf.train.AdamOptimizer(self.learn_rate_init, beta1=0.5, beta2=0.9).minimize(self.cls_loss)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("vae_loss", self.vae_loss, collections=['train'])
            tf.summary.scalar("e2_loss", self.e2_loss, collections=['train'])
            tf.summary.scalar("anomaly_score", an_score, collections=['train'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['train'])

            tf.summary.scalar("cls_acc", self.cls_acc, collections=['train'])
            
            #tf.summary.scalar("code_d_loss", self.code_d_loss, collections=['train'])
            #tf.summary.scalar("code_g_loss", self.code_g_loss, collections=['train'])
            #tf.summary.scalar("code_disc_fake_loss", code_disc_fake_loss, collections=['train'])
            #tf.summary.scalar("code_disc_ture_loss", code_disc_ture_loss, collections=['train'])            
            #tf.summary.scalar("code_gen_fake_loss", code_disc_fake_loss, collections=['train'])
            #tf.summary.scalar("code_gen_ture_loss", code_disc_ture_loss, collections=['train'])     
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("vae_loss", self.vae_loss, collections=['test'])
            tf.summary.scalar("e2_loss", self.e2_loss, collections=['test'])
            tf.summary.scalar("anomaly_score", an_score, collections=['test'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['test'])

            tf.summary.scalar("cls_acc", self.cls_acc, collections=['test'])

            #tf.summary.scalar("code_d_loss", self.code_d_loss, collections=['test'])
            #tf.summary.scalar("code_g_loss", self.code_g_loss, collections=['test'])
            #tf.summary.scalar("code_disc_fake_loss", code_disc_fake_loss, collections=['test'])
            #tf.summary.scalar("code_disc_ture_loss", code_disc_ture_loss, collections=['test'])  
            #tf.summary.scalar("code_gen_fake_loss", code_disc_fake_loss, collections=['test'])
            #tf.summary.scalar("code_gen_ture_loss", code_disc_ture_loss, collections=['test'])                 
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):
            tf.summary.scalar("vae_loss", self.vae_loss, collections=['anomaly'])
            tf.summary.scalar("e2_loss", self.e2_loss, collections=['anomaly'])            
            tf.summary.scalar("anomaly_score", an_score, collections=['anomaly'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
            tf.summary.scalar("disc_fake_loss", disc_fake_loss, collections=['anomaly'])
            tf.summary.scalar("disc_ture_loss", disc_ture_loss, collections=['anomaly'])

            #tf.summary.scalar("code_d_loss", self.code_d_loss, collections=['anomaly'])
            #tf.summary.scalar("code_g_loss", self.code_g_loss, collections=['anomaly'])
            #tf.summary.scalar("code_disc_fake_loss", code_disc_fake_loss, collections=['anomaly'])
            #tf.summary.scalar("code_disc_ture_loss", code_disc_ture_loss, collections=['anomaly'])  
            #tf.summary.scalar("code_gen_fake_loss", code_disc_fake_loss, collections=['anomaly'])
            #tf.summary.scalar("code_gen_ture_loss", code_disc_ture_loss, collections=['anomaly'])                 
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("marginal_likelihood", tf.reduce_mean(marginal_likelihood), collections=['anomaly'])
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_AE_GAN(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_code, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Encoder2
        self.f_code, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})     

        self.saver = tf.train.Saver()

    def train_AD_att_AE_GAN(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))
            anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     

                if step > 0:
                    delta = 1.0
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                        self.lr: new_learning_rate,
                     }
                               
                # Training
                sess.run(self.train_vae, feed_dict=fd)
                #sess.run(self.train_e2, feed_dict=fd)
                
                if step > 0:
                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch
                        random.shuffle(dataset_idx)
                        next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                        next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                                
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.delta: delta,
                                self.lr: new_learning_rate,
                             }                      

                        sess.run(self.train_d, feed_dict=fd)
                        #sess.run(self.train_code_dis, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.vae_loss], feed_dict=fd)

                    # Validation set
                    test_loss = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                    self.lr: new_learning_rate,
                                  }
                                               
                        test_sum, temp_loss = sess.run([self.merged_summary_test, self.vae_loss], feed_dict=fd_test)                       
                        test_loss = test_loss + temp_loss

                    test_loss = test_loss / test_count          

                    # Anomaly set
                    ano_loss = 0
                    ano_count = 0
                    curr_idx = 0
                    random.shuffle(anomaly_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.anomaly_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        ano_count = ano_count + 1

                        fd_test = {
                                    self.images: next_ano_x_images, 
                                    self.labels: next_ano_y, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                    self.lr: new_learning_rate,
                                  }
                                               
                        ano_sum, temp_ano_loss = sess.run([self.merged_summary_anomaly, self.vae_loss], feed_dict=fd_test)                       
                        ano_loss = ano_loss + temp_ano_loss

                    ano_loss = ano_loss / ano_count    

                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_AE_GAN(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            output_name = os.path.basename(self.data_ob.test_images_path)
            output_name = os.path.splitext(output_name)[0] + ".csv"
            print("Output: {}".format(output_name))
            
            output = np.array([])
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                

                t_code, f_code, decoder_output = sess.run([self.t_code, self.f_code, self.decoder_output], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
    
                
                for idx in range(len(decoder_output)):
                    scipy.misc.imsave("./output_image/" + "decode_" + str(idx) + '.png', decoder_output[idx])
                    scipy.misc.imsave("./output_image/" + "encode_" + str(idx) + '.png', next_x_images[idx])
    
                #dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)
                
                curr_output = np.concatenate((next_test_y, t_code, f_code), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_AE_GAN_3DCode_32x32(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_code, self.t_conv_1, self.t_conv_2, self.t_conv_3 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # AE + WGAN =============================================================================================================   
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t, _ = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f, _ = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat, _ = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))

        self.g_loss_1 = 50*self.content_loss + disc_fake_loss
        self.d_loss_1 = -(disc_fake_loss - disc_ture_loss) + d_gp           

        # Classifier =============================================================================================================
        cls_logits = mz.build_model({"mode":"classifier", "cls_input":self.images, "reuse":False})  
          
        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_logits, labels=self.labels))
        self.cls_pred = tf.argmax(cls_logits, axis=1)
        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))     
        
        # WGAN ==================================================================================================================       
        self.f_images = mz.build_model({"mode":"generator", "code":self.t_code, "reuse":False})      
        self.dis2_t_inputs = self.images2
        self.dis2_f_inputs = self.f_images
        dis2_t, _ = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_t_inputs, "reuse":False})       
        dis2_f, _ = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x2_hat = epsilon * self.dis2_t_inputs + (1. - epsilon) * (self.dis2_f_inputs)
        d2_hat, _ = mz.build_model({"mode":"discriminator2", "dis_input":x2_hat, "reuse":True})
        
        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10

        disc2_ture_loss = tf.reduce_mean(dis2_t)
        disc2_fake_loss = tf.reduce_mean(dis2_f)

        cls_f_logits = mz.build_model({"mode":"classifier", "cls_input":self.f_images, "reuse":True})  
        self.cls_f_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_f_logits, labels=self.labels))

        self.g_loss_2 = disc2_fake_loss + self.cls_f_loss
        self.d_loss_2 = -(disc2_fake_loss - disc2_ture_loss) + d2_gp  

        self.g_loss = self.g_loss_1# + self.g_loss_2    
        self.guass_loss = tf.reduce_mean(tf.abs(self.f_images - self.images))

        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()
        d1_var = [v for v in train_variables if v.name.startswith("discriminator")]
        g1_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d2_var = [v for v in train_variables if v.name.startswith("discriminator2")]
        g2_var = [v for v in train_variables if v.name.startswith(("generator"))]
        c_var = [v for v in train_variables if v.name.startswith(("classifier"))]
        
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g1_var)
        #self.train_g1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_1, var_list=g1_var)
        self.train_d1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_1, var_list=d1_var)
        self.train_g2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_2, var_list=g2_var)
        self.train_d2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_2, var_list=d2_var)
        
        self.train_c = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.cls_loss, var_list=c_var)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['train'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['train'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['train'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['train'])
            
            tf.summary.scalar("cls_loss", self.cls_loss, collections=['train'])
            tf.summary.scalar("cls_f_loss", self.cls_f_loss, collections=['train'])
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("guass_loss", self.guass_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("input_image2", self.images2, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("f_image", self.f_images, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['test'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['test'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['test'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['test'])
            
            tf.summary.scalar("cls_loss", self.cls_loss, collections=['test'])
            tf.summary.scalar("cls_f_loss", self.cls_f_loss, collections=['test'])
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['test'])            
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("guass_loss", self.guass_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("input_image2", self.images2, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("f_image", self.f_images, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['anomaly'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['anomaly'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['anomaly'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['anomaly'])
                            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("guass_loss", self.guass_loss, collections=['anomaly'])
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            tf.summary.image("f_image", self.f_images, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_AE_GAN_3DCode_32x32(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_code, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Encoder2
        self.f_code, self.cls_logits, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})     
        self.t_code_2, _, _ = mz.build_model({"mode":"encoder2", "en_input":self.images, "reuse":True})     

        self.saver = tf.train.Saver()

    def train_AD_att_AE_GAN_3DCode_32x32(self):
        
        new_learning_rate = self.learn_rate_init
        WGAN2_lr = self.learn_rate_init
        
        WGAN2_start_step = 10000
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            #dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            #valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))
            anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                #random.shuffle(dataset_idx)
                #next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                #next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                
                next_x_images, next_y = get_batch(self.dataset, self.batch_size, self.anoCls)
                next_x_images2, _ = get_batch(self.dataset, self.batch_size, self.anoCls)  
                  
                fd = {
                        self.images: next_x_images, 
                        self.images2: next_x_images2, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                     }
                               
                # Training                 
                sess.run(self.train_g, feed_dict=fd)
                sess.run(self.train_c, feed_dict=fd)    
                
                if step > WGAN2_start_step:
                    fd = {
                            self.images: next_x_images, 
                            self.images2: next_x_images2,
                            self.labels: next_y, 
                            self.dropout_rate: self.dropout,
                            self.lr: WGAN2_lr,
                         }
                    sess.run(self.train_g2, feed_dict=fd)
                                
                # Training Discriminator
                for d_iter in range(0, 5):
                    
                    # Get the training batch
                    #random.shuffle(dataset_idx)
                    #next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                    #next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                    next_x_images, next_y = get_batch(self.dataset, self.batch_size, self.anoCls)
                    next_x_images2, _ = get_batch(self.dataset, self.batch_size, self.anoCls)    
                    
                    fd = {
                            self.images: next_x_images, 
                            self.images2: next_x_images2, 
                            self.labels: next_y, 
                            self.dropout_rate: self.dropout,
                            self.lr: new_learning_rate,
                         }                      

                    sess.run(self.train_d1, feed_dict=fd)
                    
                    if step > WGAN2_start_step:

                        fd = {
                                self.images: next_x_images, 
                                self.images2: next_x_images2, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.lr: WGAN2_lr,
                             }
                        sess.run(self.train_d2, feed_dict=fd)
    
                # Update Learning rate                
                if step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                    if step > WGAN2_start_step: 
                        WGAN2_lr = WGAN2_lr * 0.1
                        print("STEP {}, WGAN2 Learning rate: {}".format(step, WGAN2_lr))
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.g_loss_1], feed_dict=fd)

                    # Validation set
                    #test_loss = 0
                    #test_count = 0
                    #curr_idx = 0
                    #random.shuffle(valid_dataset_idx)
                    
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size, self.anoCls)
                    next_valid_x_images2, _ = get_batch(self.valid_dataset, self.batch_size, self.anoCls)
                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.images2: next_valid_x_images2, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.lr: new_learning_rate,
                              }
                                           
                    test_sum, test_loss = sess.run([self.merged_summary_test, self.g_loss], feed_dict=fd_test)  
                    
#                    while True:
#                                                                        
#                        try:                    
#                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
#                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
#                            curr_idx = curr_idx + self.batch_size
#                            
#                            if curr_idx > len(self.valid_dataset[0]):
#                                break
#                                
#                        except tf.errors.OutOfRangeError:
#                            break
#
#                        test_count = test_count + 1
#
#                        fd_test = {
#                                    self.images: next_valid_x_images, 
#                                    self.labels: next_valid_y, 
#                                    self.dropout_rate: 0,
#                                    self.lr: new_learning_rate,
#                                  }
#                                               
#                        test_sum, temp_loss = sess.run([self.merged_summary_test, self.g_loss], feed_dict=fd_test)                       
#                        test_loss = test_loss + temp_loss
#
#                    test_loss = test_loss / test_count          

                    # Anomaly set
                    #ano_loss = 0
                    #ano_count = 0
                    curr_idx = 0
                    random.shuffle(anomaly_dataset_idx)                   

                    next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                    next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                    curr_idx = curr_idx + self.batch_size
                    next_ano_x_images2 = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                    
                    fd_test = {
                                self.images: next_ano_x_images, 
                                self.images2: next_ano_x_images2, 
                                self.labels: next_ano_y, 
                                self.dropout_rate: 0,
                                self.lr: new_learning_rate,
                              }
                                               
                    ano_sum, ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss], feed_dict=fd_test) 
                    
                    
                    
#                    while True:
#                                                                        
#                        try:                    
#                            next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
#                            next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
#                            curr_idx = curr_idx + self.batch_size
#                            if curr_idx > len(self.anomaly_dataset[0]):
#                                break
#                                
#                        except tf.errors.OutOfRangeError:
#                            break
#
#                        ano_count = ano_count + 1
#
#                        fd_test = {
#                                    self.images: next_ano_x_images, 
#                                    self.labels: next_ano_y, 
#                                    self.dropout_rate: 0,
#                                    self.lr: new_learning_rate,
#                                  }
#                                               
#                        ano_sum, temp_ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss], feed_dict=fd_test)                       
#                        ano_loss = ano_loss + temp_ano_loss
#
#                    ano_loss = ano_loss / ano_count    

                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_AE_GAN_3DCode_32x32(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            output_name = os.path.basename(self.data_ob.test_images_path)
            output_name = os.path.splitext(output_name)[0] + ".csv"
            print("Output: {}".format(output_name))
            
            output = np.array([])
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                

                f_code, t_code_2, decoder_output = sess.run([self.f_code, self.t_code_2, self.decoder_output], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
    
                f_code = np.reshape(f_code, (np.shape(next_x_images)[0], -1))
                t_code_2 = np.reshape(t_code_2, (np.shape(next_x_images)[0], -1))   
                an_score = np.mean(abs(f_code-t_code_2), axis=-1, keepdims=True)
                
                for idx in range(len(decoder_output)):
                    scipy.misc.imsave("./output_image/" + "decode_" + str(idx) + '.png', decoder_output[idx])
                    scipy.misc.imsave("./output_image/" + "encode_" + str(idx) + '.png', next_x_images[idx])
    
                #dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)
                
                curr_output = np.concatenate((next_test_y, an_score, t_code_2, f_code), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_AE_GAN_3DCode(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_code, cls_logits, self.t_conv_1, self.t_conv_2, self.t_conv_3 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # AE + WGAN =============================================================================================================   
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t, _ = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f, _ = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat, _ = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))

        self.g_loss_1 = 50*self.content_loss + disc_fake_loss
        self.d_loss_1 = -(disc_fake_loss - disc_ture_loss) + d_gp           
        
        # WGAN ==================================================================================================================       
        self.f_images = mz.build_model({"mode":"generator", "code":self.t_code, "reuse":False})      
        self.dis2_t_inputs = self.images
        self.dis2_f_inputs = self.f_images
        dis2_t, _ = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_t_inputs, "reuse":False})       
        dis2_f, _ = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x2_hat = epsilon * self.dis2_t_inputs + (1. - epsilon) * (self.dis2_f_inputs)
        d2_hat, _ = mz.build_model({"mode":"discriminator2", "dis_input":x2_hat, "reuse":True})
        
        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10

        disc2_ture_loss = tf.reduce_mean(dis2_t)
        disc2_fake_loss = tf.reduce_mean(dis2_f)

        self.g_loss_2 = disc2_fake_loss
        self.d_loss_2 = -(disc2_fake_loss - disc2_ture_loss) + d2_gp  

        # ========================================================================================================================
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

#        t_att_1 = tf.reduce_mean(self.t_conv_1, axis=3, keepdims=True)
#        f_att_1 = tf.reduce_mean(self.f_conv_1, axis=3, keepdims=True)
#        
#        t_att_2 = tf.reduce_mean(self.t_conv_2, axis=3, keepdims=True)
#        f_att_2 = tf.reduce_mean(self.f_conv_2, axis=3, keepdims=True)
#        
#        t_att_3 = tf.reduce_mean(self.t_conv_3, axis=3, keepdims=True)
#        f_att_3 = tf.reduce_mean(self.f_conv_3, axis=3, keepdims=True)
#
#        t_att_image = tf.image.resize_images(t_att_1, (64, 64)) + tf.image.resize_images(t_att_2, (64, 64)) + tf.image.resize_images(t_att_3, (64, 64))
#        f_att_image = tf.image.resize_images(f_att_1, (64, 64)) + tf.image.resize_images(f_att_2, (64, 64)) + tf.image.resize_images(f_att_3, (64, 64))
        
        train_variables = tf.trainable_variables()
        d1_var = [v for v in train_variables if v.name.startswith("discriminator")]
        g1_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d2_var = [v for v in train_variables if v.name.startswith("discriminator2")]
        g2_var = [v for v in train_variables if v.name.startswith(("generator"))]
        
        self.train_g1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_1, var_list=g1_var)
        self.train_d1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_1, var_list=d1_var)
        self.train_g2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_2, var_list=g2_var)
        self.train_d2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_2, var_list=d2_var)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['train'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['train'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['train'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", tf.reduce_mean(self.content_loss), collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("f_image", self.f_images, collections=['train']) 

#            tf.summary.image("t_att_1", t_att_1, collections=['train'])
#            tf.summary.image("f_att_1", f_att_1, collections=['train'])
#
#            tf.summary.image("t_att_2", t_att_2, collections=['train'])
#            tf.summary.image("f_att_2", f_att_2, collections=['train'])
#            
#            tf.summary.image("t_att_3", t_att_3, collections=['train'])
#            tf.summary.image("f_att_3", f_att_3, collections=['train'])            

#            tf.summary.image("t_att", t_att_image, collections=['train'])
#            tf.summary.image("f_att", f_att_image, collections=['train'])            
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['test'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['test'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['test'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            
            tf.summary.scalar("content_loss", tf.reduce_mean(self.content_loss), collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("f_image", self.f_images, collections=['test']) 
            
#            tf.summary.image("t_att_1", t_att_1, collections=['test'])
#            tf.summary.image("f_att_1", f_att_1, collections=['test'])
#
#            tf.summary.image("t_att_2", t_att_2, collections=['test'])
#            tf.summary.image("f_att_2", f_att_2, collections=['test'])
#            
#            tf.summary.image("t_att_3", t_att_3, collections=['test'])
#            tf.summary.image("f_att_3", f_att_3, collections=['test'])            

#            tf.summary.image("t_att", t_att_image, collections=['test'])
#            tf.summary.image("f_att", f_att_image, collections=['test'])            
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['anomaly'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['anomaly'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['anomaly'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['anomaly'])
                            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", tf.reduce_mean(self.content_loss), collections=['anomaly'])
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            tf.summary.image("f_image", self.f_images, collections=['anomaly']) 
            
#            tf.summary.image("t_att_1", t_att_1, collections=['anomaly'])
#            tf.summary.image("f_att_1", f_att_1, collections=['anomaly'])
#
#            tf.summary.image("t_att_2", t_att_2, collections=['anomaly'])
#            tf.summary.image("f_att_2", f_att_2, collections=['anomaly'])
#            
#            tf.summary.image("t_att_3", t_att_3, collections=['anomaly'])
#            tf.summary.image("f_att_3", f_att_3, collections=['anomaly'])            

#            tf.summary.image("t_att", t_att_image, collections=['anomaly'])
#            tf.summary.image("f_att", f_att_image, collections=['anomaly'])            
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_AE_GAN_3DCode(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_code, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Encoder2
        self.f_code, self.cls_logits, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})     
        self.t_code_2, _, _ = mz.build_model({"mode":"encoder2", "en_input":self.images, "reuse":True})     

        self.saver = tf.train.Saver()

    def train_AD_att_AE_GAN_3DCode(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))
            anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                     }
                               
                # Training
                sess.run(self.train_g1, feed_dict=fd)
                sess.run(self.train_g2, feed_dict=fd)
                                
                # Training Discriminator
                for d_iter in range(0, 5):
                    
                    # Get the training batch
                    random.shuffle(dataset_idx)
                    next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                    next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                                            
                    fd = {
                            self.images: next_x_images, 
                            self.labels: next_y, 
                            self.dropout_rate: self.dropout,
                            self.lr: new_learning_rate,
                         }                      

                    sess.run(self.train_d1, feed_dict=fd)
                    sess.run(self.train_d2, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.g_loss_1], feed_dict=fd)

                    # Validation set
                    test_loss = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    self.labels: next_valid_y, 
                                    self.dropout_rate: 0,
                                    self.lr: new_learning_rate,
                                  }
                                               
                        test_sum, temp_loss = sess.run([self.merged_summary_test, self.g_loss_1], feed_dict=fd_test)                       
                        test_loss = test_loss + temp_loss

                    test_loss = test_loss / test_count          

                    # Anomaly set
                    ano_loss = 0
                    ano_count = 0
                    curr_idx = 0
                    random.shuffle(anomaly_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            if curr_idx > len(self.anomaly_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        ano_count = ano_count + 1

                        fd_test = {
                                    self.images: next_ano_x_images, 
                                    self.labels: next_ano_y, 
                                    self.dropout_rate: 0,
                                    self.lr: new_learning_rate,
                                  }
                                               
                        ano_sum, temp_ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss_1], feed_dict=fd_test)                       
                        ano_loss = ano_loss + temp_ano_loss

                    ano_loss = ano_loss / ano_count    

                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_AE_GAN_3DCode(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            output_name = os.path.basename(self.data_ob.test_images_path)
            output_name = os.path.splitext(output_name)[0] + ".csv"
            print("Output: {}".format(output_name))
            
            output = np.array([])
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                

                f_code, t_code_2, decoder_output = sess.run([self.f_code, self.t_code_2, self.decoder_output], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
    
                f_code = np.reshape(f_code, (np.shape(next_x_images)[0], -1))
                t_code_2 = np.reshape(t_code_2, (np.shape(next_x_images)[0], -1))   
                an_score = np.mean(abs(f_code-t_code_2), axis=-1, keepdims=True)
                
                for idx in range(len(decoder_output)):
                    scipy.misc.imsave("./output_image/" + "decode_" + str(idx) + '.png', decoder_output[idx])
                    scipy.misc.imsave("./output_image/" + "encode_" + str(idx) + '.png', next_x_images[idx])
    
                #dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)
                
                curr_output = np.concatenate((next_test_y, an_score, t_code_2, f_code), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_AD_att_AE_GAN_CLS(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_code, self.t_conv_1, self.t_conv_2, self.t_conv_3 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # AE + WGAN =============================================================================================================   
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t, _ = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f, _ = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat, _ = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)
        
        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))

        self.g_loss_1 = 50*self.content_loss + disc_fake_loss
        self.d_loss_1 = -(disc_fake_loss - disc_ture_loss) + d_gp           

        # Classifier =============================================================================================================
        cls_logits = mz.build_model({"mode":"classifier", "cls_input":self.images, "reuse":False})  
          
        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_logits, labels=self.labels))
        self.cls_pred = tf.argmax(tf.nn.softmax(cls_logits), axis=1)
        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))     

        # Code Disentangle =======================================================================================================
#        self.dis_t_code = tf.slice(self.t_code, [0,0,0,0], [-1, -1, -1, 16])
#        code_dis_logits = mz.build_model({"mode":"code_cls", "code_cls_input":self.dis_t_code, "reuse":False})  
#          
#        self.code_dis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_dis_logits, labels=self.labels))
#        self.code_dis_pred = tf.argmax(tf.nn.softmax(code_dis_logits), axis=1)
#        code_dis_correct_pred = tf.equal(self.code_dis_pred, tf.argmax(self.labels, axis=1))
#        self.code_dis_acc = tf.reduce_mean(tf.cast(code_dis_correct_pred, tf.float32))     
        
        # WGAN ==================================================================================================================       
        self.f_images = mz.build_model({"mode":"generator", "code":self.t_code, "reuse":False})      
        self.dis2_t_inputs = self.images2
        self.dis2_f_inputs = self.f_images
        dis2_t, _ = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_t_inputs, "reuse":False})       
        dis2_f, _ = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x2_hat = epsilon * self.dis2_t_inputs + (1. - epsilon) * (self.dis2_f_inputs)
        d2_hat, _ = mz.build_model({"mode":"discriminator2", "dis_input":x2_hat, "reuse":True})
        
        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10

        disc2_ture_loss = tf.reduce_mean(dis2_t)
        disc2_fake_loss = tf.reduce_mean(dis2_f)

        cls_f_logits = mz.build_model({"mode":"classifier", "cls_input":self.f_images, "reuse":True})  
        self.cls_f_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_f_logits, labels=self.labels))
        self.cls_f_pred = tf.argmax(tf.nn.softmax(cls_f_logits), axis=1)
        cls_correct_f_pred = tf.equal(self.cls_f_pred, tf.argmax(self.labels, axis=1))
        self.cls_f_acc = tf.reduce_mean(tf.cast(cls_correct_f_pred, tf.float32))     
        
        self.g_loss_2 = disc2_fake_loss + 50*self.cls_f_loss
        self.d_loss_2 = -(disc2_fake_loss - disc2_ture_loss) + d2_gp  

        self.g_loss = self.g_loss_1  
        self.guass_loss = tf.reduce_mean(tf.abs(self.f_images - self.images))

        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()
        d1_var = [v for v in train_variables if v.name.startswith("discriminator")]
        g1_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d2_var = [v for v in train_variables if v.name.startswith("discriminator2")]
        g2_var = [v for v in train_variables if v.name.startswith(("generator"))]
        c_var = [v for v in train_variables if v.name.startswith(("classifier"))]
        #c_d_var = [v for v in train_variables if v.name.startswith(("code_cls"))]
        
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g1_var)
        #self.train_g1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_1, var_list=g1_var)
        self.train_d1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_1, var_list=d1_var)
        self.train_g2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_2, var_list=g2_var)
        self.train_d2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_2, var_list=d2_var)
        
        self.train_c = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.cls_loss, var_list=c_var)
        #self.train_c_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.code_dis_loss, var_list=c_d_var)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['train'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['train'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['train'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['train'])
            
            tf.summary.scalar("cls_loss", self.cls_loss, collections=['train'])
            tf.summary.scalar("cls_f_loss", self.cls_f_loss, collections=['train'])
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['train'])
            tf.summary.scalar("cls_f_acc", self.cls_f_acc, collections=['train'])

            #tf.summary.scalar("code_dis_loss", self.code_dis_loss, collections=['train'])
            #tf.summary.scalar("code_dis_acc", self.code_dis_acc, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("guass_loss", self.guass_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("f_image", self.f_images, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['test'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['test'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['test'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['test'])
            
            tf.summary.scalar("cls_loss", self.cls_loss, collections=['test'])
            tf.summary.scalar("cls_f_loss", self.cls_f_loss, collections=['test'])
            tf.summary.scalar("cls_acc", self.cls_acc, collections=['test'])            
            tf.summary.scalar("cls_f_acc", self.cls_f_acc, collections=['test'])

            #tf.summary.scalar("code_dis_loss", self.code_dis_loss, collections=['test'])
            #tf.summary.scalar("code_dis_acc", self.code_dis_acc, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("guass_loss", self.guass_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("f_image", self.f_images, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])
            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['anomaly'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['anomaly'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['anomaly'])
            tf.summary.scalar("d_loss_2", self.d_loss_2, collections=['anomaly'])
                            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("guass_loss", self.guass_loss, collections=['anomaly'])
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            tf.summary.image("f_image", self.f_images, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_att_AE_GAN_CLS(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_code, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Encoder2
        self.f_code, self.cls_logits, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})     
        self.t_code_2, _, _ = mz.build_model({"mode":"encoder2", "en_input":self.images, "reuse":True})     

        self.saver = tf.train.Saver()

    def train_AD_att_AE_GAN_CLS(self):
        
        new_learning_rate = self.learn_rate_init
        WGAN2_lr = self.learn_rate_init
        
        WGAN2_start_step = 10000
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                  
                # Get the training batch
                next_x_images, next_y = get_batch(self.dataset, self.batch_size, self.anoCls)
                next_x_images2, _ = get_batch(self.dataset, self.batch_size, self.anoCls)  
                  
                fd = {
                        self.images: next_x_images, 
                        self.images2: next_x_images2, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                     }
                               
                # Training                 
                sess.run(self.train_g, feed_dict=fd)
                
                sess.run(self.train_c, feed_dict=fd)    
                
                if step > WGAN2_start_step:
                    fd = {
                            self.images: next_x_images, 
                            self.images2: next_x_images2,
                            self.labels: next_y, 
                            self.dropout_rate: self.dropout,
                            self.lr: WGAN2_lr,
                         }
                    sess.run(self.train_g2, feed_dict=fd)
                                
                # Training Discriminator
                for d_iter in range(0, 5):
                    
                    # Get the training batch 
                    next_x_images, next_y = get_batch(self.dataset, self.batch_size, self.anoCls)
                    next_x_images2, _ = get_batch(self.dataset, self.batch_size, self.anoCls)    
                    
                    fd = {
                            self.images: next_x_images, 
                            self.images2: next_x_images2, 
                            self.labels: next_y, 
                            self.dropout_rate: self.dropout,
                            self.lr: new_learning_rate,
                         }                      

                    sess.run(self.train_d1, feed_dict=fd)
                    
                    if step > WGAN2_start_step:

                        fd = {
                                self.images: next_x_images, 
                                self.images2: next_x_images2, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.lr: WGAN2_lr,
                             }
                        sess.run(self.train_d2, feed_dict=fd)
    
                # Update Learning rate                
                if step == 20000 or step == 40000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                    if step > WGAN2_start_step + 20000: 
                        WGAN2_lr = WGAN2_lr * 0.1
                        print("STEP {}, WGAN2 Learning rate: {}".format(step, WGAN2_lr))
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.g_loss_1], feed_dict=fd)
                    
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size, self.anoCls)
                    next_valid_x_images2, _ = get_batch(self.valid_dataset, self.batch_size, self.anoCls)
                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.images2: next_valid_x_images2, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.lr: new_learning_rate,
                              }
                                           
                    test_sum, test_loss = sess.run([self.merged_summary_test, self.g_loss], feed_dict=fd_test)  
                    
                    curr_idx = 0
                    random.shuffle(anomaly_dataset_idx)                   

                    next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                    next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                    curr_idx = curr_idx + self.batch_size
                    next_ano_x_images2 = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                    
                    fd_test = {
                                self.images: next_ano_x_images, 
                                self.images2: next_ano_x_images2, 
                                self.labels: next_ano_y, 
                                self.dropout_rate: 0,
                                self.lr: new_learning_rate,
                              }
                                               
                    ano_sum, ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss], feed_dict=fd_test) 
                      
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_att_AE_GAN_CLS(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            output_name = os.path.basename(self.data_ob.test_images_path)
            output_name = os.path.splitext(output_name)[0] + ".csv"
            print("Output: {}".format(output_name))
            
            output = np.array([])
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                

                f_code, t_code_2, decoder_output = sess.run([self.f_code, self.t_code_2, self.decoder_output], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
    
                f_code = np.reshape(f_code, (np.shape(next_x_images)[0], -1))
                t_code_2 = np.reshape(t_code_2, (np.shape(next_x_images)[0], -1))   
                an_score = np.mean(abs(f_code-t_code_2), axis=-1, keepdims=True)
                
                for idx in range(len(decoder_output)):
                    scipy.misc.imsave("./output_image/" + "decode_" + str(idx) + '.png', decoder_output[idx])
                    scipy.misc.imsave("./output_image/" + "encode_" + str(idx) + '.png', next_x_images[idx])
    
                #dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)
                
                curr_output = np.concatenate((next_test_y, an_score, t_code_2, f_code), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_GANomaly(self):
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_code = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # Discriminator ===================================================================================================================       
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t, dis_t_feature = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f, dis_f_feature = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        dis_f_feature = tf.reduce_mean(dis_f_feature, axis=0)

        self.d_loss = - tf.reduce_mean(tf.log(dis_t) + tf.log(1. -dis_f))

        self.f_code = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})        

        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))      
        self.enc_loss = tf.reduce_mean(tf.pow(self.f_code - self.t_code, 2))# + self.code_g_loss
        self.adv_loss = tf.reduce_mean(tf.pow(dis_t_feature - dis_f_feature, 2))
        self.g_loss = 50*self.content_loss + self.enc_loss + self.adv_loss
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        an_score = tf.reduce_mean(tf.abs(self.f_code-self.t_code))

        train_variables = tf.trainable_variables()
        d_var = [v for v in train_variables if v.name.startswith("discriminator")]
        g_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_var)
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_var)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("enc_loss", self.enc_loss, collections=['train'])
            tf.summary.scalar("adv_loss", self.adv_loss, collections=['train'])
            tf.summary.scalar("anomaly_score", an_score, collections=['train'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
                        
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("enc_loss", self.enc_loss, collections=['test'])
            tf.summary.scalar("adv_loss", self.adv_loss, collections=['test'])
            tf.summary.scalar("anomaly_score", an_score, collections=['test'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
                       
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):
            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])  
            tf.summary.scalar("enc_loss", self.enc_loss, collections=['anomaly'])
            tf.summary.scalar("adv_loss", self.adv_loss, collections=['anomaly'])
            tf.summary.scalar("anomaly_score", an_score, collections=['anomaly'])
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
                        
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_GANomaly(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.t_code, self.t_att = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})   
        
        # Encoder2
        self.f_code, self.cls_logits, self.f_att = mz.build_model({"mode":"encoder2", "en_input":self.decoder_output, "reuse":False})     
        self.t_code_2, _, _ = mz.build_model({"mode":"encoder2", "en_input":self.images, "reuse":True})     

        self.saver = tf.train.Saver()

    def train_GANomaly(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            dataset_idx = np.array(list(range(0, len(self.dataset[0]))))
            valid_dataset_idx = np.array(list(range(0, len(self.valid_dataset[0]))))
            anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            delta = 0
            
            # =================================================================
                       
            #(x_train, y_train), (x_test, y_test) = mnist.load_data()
            #x_train_normal = x_train[y_train == 1] 
            #x_test_normal = x_test[y_test == 1] 
            #x_test_abnormal = x_test[y_test == 7] 

            def reshape_x(x):
                new_x = np.empty((len(x), 64, 64, 3))
                for i, e in enumerate(x):
                    new_x[i] = cv2.resize(e, (64, 64))
                #return np.expand_dims(new_x, axis=-1) / 127 - 1
                return new_x
              
            #x_train_normal = reshape_x(x_train_normal)
            #x_test_normal = reshape_x(x_test_normal)
            #x_test_abnormal = reshape_x(x_test_abnormal)

            def get_data_generator(data, batch_size=32):
                datalen = len(data)
                cnt = 0
                while True:
                    idxes = np.arange(datalen)
                    np.random.shuffle(idxes)
                    cnt += 1
                    for i in range(datalen//batch_size):
                        train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
                        yield train_x
            
            #train_data_generator = get_data_generator(x_train_normal, 32)
            #test_data_generator = get_data_generator(x_test_normal, 32)
            #anomaly_data_generator = get_data_generator(x_test_abnormal, 32)
            
            # =================================================================
            
            while step <= self.max_iters:
                  
                # Get the training batch
                random.shuffle(dataset_idx)
                next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                #next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                next_x_images = reshape_x(next_x_images)

                #next_x_images = train_data_generator.__next__()

                if step > 0:
                    delta = 1.0
                                    
                fd = {
                        self.images: next_x_images, 
                        #self.labels: None, 
                        self.dropout_rate: self.dropout,
                        self.delta: delta,
                        self.lr: new_learning_rate,
                     }
                               
                # Training
                sess.run(self.train_g, feed_dict=fd)
                                   
                # Training Discriminator
                for d_iter in range(0, 1):
                    
                    # Get the training batch
                    random.shuffle(dataset_idx)
                    next_x_images = self.dataset[0][dataset_idx[0:self.batch_size]]
                    #next_y = self.dataset[1][dataset_idx[0:self.batch_size]]     
                    next_x_images = reshape_x(next_x_images)
                                            
                    fd = {
                            self.images: next_x_images, 
                            #self.labels: None, 
                            self.dropout_rate: self.dropout,
                            self.delta: delta,
                            self.lr: new_learning_rate,
                         }                      

                    sess.run(self.train_d, feed_dict=fd)
    
                # Update Learning rate                
                if step == 10000 or step == 15000 or step == 20000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.g_loss], feed_dict=fd)
                    
                    # Validation set
                    test_loss = 0
                    test_count = 0
                    curr_idx = 0
                    random.shuffle(valid_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_valid_x_images = self.valid_dataset[0][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            #next_valid_y = self.valid_dataset[1][valid_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            next_valid_x_images = reshape_x(next_valid_x_images)
                            if curr_idx > len(self.valid_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        test_count = test_count + 1

                        fd_test = {
                                    self.images: next_valid_x_images, 
                                    #self.labels: None, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                    self.lr: new_learning_rate,
                                  }
                                               
                        test_sum, temp_loss = sess.run([self.merged_summary_test, self.g_loss], feed_dict=fd_test)                       
                        test_loss = test_loss + temp_loss

                    test_loss = test_loss / test_count          

#                    next_valid_x_images = test_data_generator.__next__()
#
#                    fd_test = {
#                                self.images: next_valid_x_images, 
#                                #self.labels: None, 
#                                self.dropout_rate: 0,
#                                self.delta: delta,
#                                self.lr: new_learning_rate,
#                              }
#                    
#                    test_sum, test_loss = sess.run([self.merged_summary_test, self.g_loss], feed_dict=fd_test)    
                    
                    # Anomaly set
                    ano_loss = 0
                    ano_count = 0
                    curr_idx = 0
                    random.shuffle(anomaly_dataset_idx)
                    while True:
                                                                        
                        try:                    
                            next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                            #next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                            curr_idx = curr_idx + self.batch_size
                            next_ano_x_images = reshape_x(next_ano_x_images)
                            if curr_idx > len(self.anomaly_dataset[0]):
                                break
                                
                        except tf.errors.OutOfRangeError:
                            break

                        ano_count = ano_count + 1

                        fd_test = {
                                    self.images: next_ano_x_images, 
                                    #self.labels: None, 
                                    self.dropout_rate: 0,
                                    self.delta: delta,
                                    self.lr: new_learning_rate,
                                  }
                                               
                        ano_sum, temp_ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss], feed_dict=fd_test)                       
                        ano_loss = ano_loss + temp_ano_loss

                    ano_loss = ano_loss / ano_count    

#                    next_ano_x_images = anomaly_data_generator.__next__()
#
#                    fd_test = {
#                                self.images: next_ano_x_images, 
#                                #self.labels: None, 
#                                self.dropout_rate: 0,
#                                self.delta: delta,
#                                self.lr: new_learning_rate,
#                              }
#                    
#                    ano_sum, ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss], feed_dict=fd_test)                       
                    
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) > abs(test_loss) and step > 10000:
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_GANomaly(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            output_name = os.path.basename(self.data_ob.test_images_path)
            output_name = os.path.splitext(output_name)[0] + ".csv"
            print("Output: {}".format(output_name))
            
            output = np.array([])
            
            curr_idx = 0
            while True:
                
                try:
                    #next_x_images, next_test_y = sess.run(self.next_test_x)
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                except tf.errors.OutOfRangeError:
                    break
                

                f_code, t_code_2, decoder_output = sess.run([self.f_code, self.t_code_2, self.decoder_output], 
                                                                      feed_dict=
                                                                                  {self.images: next_x_images, 
                                                                                   self.labels: next_test_y, 
                                                                                   self.dropout_rate: 0})
    
                f_code = np.reshape(f_code, (np.shape(next_x_images)[0], -1))
                t_code_2 = np.reshape(t_code_2, (np.shape(next_x_images)[0], -1))   
                an_score = np.mean(abs(f_code-t_code_2), axis=-1, keepdims=True)
                
                for idx in range(len(decoder_output)):
                    scipy.misc.imsave("./output_image/" + "decode_" + str(idx) + '.png', decoder_output[idx])
                    scipy.misc.imsave("./output_image/" + "encode_" + str(idx) + '.png', next_x_images[idx])
    
                #dis_pred = np.expand_dims(dis_pred, 1)
                
                next_test_y = np.argmax(next_test_y, axis=-1)
                next_test_y = np.expand_dims(next_test_y, 1)
                
                curr_output = np.concatenate((next_test_y, an_score, t_code_2, f_code), axis=1)
                output = np.vstack([output, curr_output]) if output.size else curr_output
                
            print(np.shape(output))
            np.savetxt(output_name, output, delimiter=",")

    def build_RaGAN_MNIST(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.image_input = tf.random_normal([self.batch_size, 128], name='input')  
        self.image_target = tf.placeholder(tf.float32, [self.batch_size, 28*28*1], name='labels')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        gen_f = mz.build_model({"net":"Gen", "reuse":False})
        dis_t = mz.build_model({"net":"Dis", "d_inputs":self.image_target, "reuse":False})
        dis_f = mz.build_model({"net":"Dis", "d_inputs":gen_f, "reuse":True})

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.image_target + (1. - epsilon) * (gen_f)
        d_hat = mz.build_model({"net":"Dis", "d_inputs":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        real_logit = (dis_t - tf.reduce_mean(dis_f))
        fake_logit = (dis_f - tf.reduce_mean(dis_t))

        disc_ture_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_t), logits=real_logit))
        disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dis_f), logits=fake_logit))

        self.d_loss = disc_fake_loss + disc_ture_loss + d_gp           

        gen_ture_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dis_t), logits=real_logit))
        gen_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_f), logits=fake_logit))
        
        self.g_loss = gen_fake_loss + gen_ture_loss 

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)
        
        self.g_output = gen_f
        
        with tf.name_scope('train_summary'):
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("dis_f_mean", tf.reduce_mean(dis_f), collections=['train'])
            tf.summary.scalar("dis_t_mean", tf.reduce_mean(dis_t), collections=['train'])
            tf.summary.image("target_image", tf.reshape(self.image_target, [self.batch_size, 28, 28, 1]), collections=['train'])
            tf.summary.image("output_image", tf.reshape(gen_f, [self.batch_size, 28, 28, 1]), collections=['train'])
    
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.image("target_image", tf.reshape(self.image_target, [self.batch_size, 28, 28, 1]), collections=['test'])
            tf.summary.image("output_image", tf.reshape(gen_f, [self.batch_size, 28, 28, 1]), collections=['test'])
        
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_RaGAN_MNIST(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        mnist = input_data.read_data_sets("/data/wei/dataset/MNIST_data/", one_hot=True)
        print("...")
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        with tf.Session() as sess:                    
            
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)    
        
            sess.run(tf.global_variables_initializer())
                
            # Define iteration counter, learning rate...
            itera_counter = 0
            learning_rate = self.learn_rate_init
    
            epoch_pbar = tqdm(range(0, self.max_iters))
            for ep in epoch_pbar:            
                # Run by batch images
                idxs = 1000
    
                epoch_pbar.set_description("Step: [%2d], lr:%f" % ((ep+1), learning_rate))
                epoch_pbar.refresh()
            
                batch_pbar = tqdm(range(0, idxs), desc="Batch: [0]")
    
                for idx in batch_pbar:                
                    batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                    itera_counter += 1
                    
                    mnist_img, _ = mnist.train.next_batch(self.batch_size)
                                      
                    for d_iter in range(0, 5):
                        _, d_loss, g_loss \
                        = sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                                 feed_dict={   
                                                                                             self.image_target: mnist_img,
                                                                                             self.dropout_rate: 1.,
                                                                                             self.lr:learning_rate
                                                                                           })
                    
                    _ = sess.run([self.train_g], 
                                                       feed_dict={
                                                                   self.image_target: mnist_img,
                                                                   self.dropout_rate: 1.,
                                                                   self.lr:learning_rate 
                                                                 })
                
                print("EP:[{}], d_loss = [{}], g_loss = [{}]\n".format(ep, d_loss, g_loss))
        
                if ep % 5 == 0 and ep != 0:
                    
                    train_sum = sess.run(self.merged_summary_train, 
                                                                        feed_dict={
                                                                                    self.image_target: mnist_img,
                                                                                    self.dropout_rate: 1.
                                                                                  })
                    
                    test_sum, g_output = sess.run([self.merged_summary_test, self.g_output],
                                                                         feed_dict={
                                                                                     self.image_target: mnist_img,
                                                                                     self.dropout_rate: 1.,
                                                                                   })
                                                                                                                                      
                    summary_writer.add_summary(train_sum, ep)
                    summary_writer.add_summary(test_sum, ep)                


    def _read_by_function(self, filename):

        array = get_image(filename, 0, is_crop=False, resize_w=self.output_size,
                           is_grayscale=False)        
        
        real_images = np.array(array)
        return real_images











