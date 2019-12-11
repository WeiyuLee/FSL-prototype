# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('./utility')

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import scipy.misc
import imageio
import random

from utils import log10, get_batch, get_batch_by_class, img_merge, img_output, mkdir_p
import netfactory as nf

from keras.datasets import mnist
import cv2

import model_zoo

from tensorflow.examples.tutorials.mnist import input_data

class model2(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, dropout, model_path, data_ob, log_dir, output_dir, learnrate_init, anoCls,
                 ckpt_name, test_ckpt, train_ckpt=[], restore_model=False, restore_step=0, class_num=10, model_ticket="none", lat_dim=128, is_training=True):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.repeat_num = repeat
        self.dropout = dropout
        self.saved_model_path = model_path
        self.data_ob = data_ob
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.learn_rate_init = learnrate_init
        self.anoCls = anoCls
        self.ckpt_name = ckpt_name
        self.test_ckpt = test_ckpt
        self.train_ckpt = train_ckpt
        self.restore_model = restore_model
        self.restore_step = restore_step
        #self.class_num = 10 - len(anoCls)
        self.log_vars = []
        self.log_imgs = []
        self.model_ticket = model_ticket
        self.lat_dim = lat_dim
        self.is_training = is_training
                
        self.channel = 3
        self.output_size = data_ob.image_size
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel], name='input1')
        self.images2 = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel], name='input2')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10], name='label')
        self.adv_weight = tf.placeholder(tf.float32, name='adv_weight')
        self.z1_src = tf.placeholder(tf.float32, [self.batch_size, 10], name='z1_src')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout')
        self.delta = tf.placeholder(tf.float32, name='delta')
        self.mean_weight = tf.placeholder(tf.float32, [self.batch_size, 256])
        
        self.ATT_START_STEP = 1000
        
        if self.is_training == True:
            
            ## Training set
            self.dataset = self.data_ob.train_data_list          
            #self.dataset = self.data_ob.train_data_dict         
    
            ## Validation set
            self.valid_dataset = self.data_ob.valid_data_list
            #self.valid_dataset = self.data_ob.valid_data_dict

            ## Anomaly set
            self.anomaly_dataset = self.data_ob.anomaly_data_list
            
        else:
            
            ## Testing set
            self.test_dataset = self.data_ob.test_data_list
            
        self.model_list = ["AD_DISE", "AD_CLS_DISE", "AD_CLS_DISE2", "AD_CLS_DISE3", "AD_CLS_DISE4", "AD_CLS_BASELINE"]

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

    def get_lr(self, step):
                            
        if step < 25000:
            lr = self.learn_rate_init

        if step >= 25000 and step < 50000:
            lr = self.learn_rate_init * 0.5            

        if step >= 50000 and step < 65000:
            lr = self.learn_rate_init * 0.5 * 0.5

        if step >= 65000 and step < 80000:
            lr = self.learn_rate_init * 0.5 * 0.5 * 0.5

        if step >= 80000 and step < 95000:
            lr = self.learn_rate_init * 0.5 * 0.5 * 0.5

        if step >= 95000 and step < 110000:
            lr = self.learn_rate_init * 0.5 * 0.5 * 0.5 * 0.5

        if step >= 110000 and step < 125000:
            lr = self.learn_rate_init * 0.5 * 0.5 * 0.5 * 0.5 * 0.5

        if step >= 125000:
            lr = self.learn_rate_init * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5
        
#        if step < 50000:
#            lr = self.learn_rate_init
#
#        if step >= 50000 and step < 200000:
#            lr = self.learn_rate_init * 0.5 
#            
#        if step >= 200000:
#            lr = self.learn_rate_init * 0.5 * 0.5
            
        return lr

    def build_AD_DISE(self):
       
        self.cdis_coef = 50        
        print("[cdis_coef]: {}".format(self.cdis_coef))
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Encoder
        self.t_code_1 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})       
        self.t_code_2 = mz.build_model({"mode":"encoder", "en_input":self.images2, "reuse":True})       

        # Code
        self.z1_1 = tf.gather(self.t_code_1, tf.range(0, 16), axis=-1)
        self.z2_1 = tf.gather(self.t_code_1, tf.range(16, 32), axis=-1)

        self.z1_2 = tf.gather(self.t_code_2, tf.range(0, 16), axis=-1)
        self.z2_2 = tf.gather(self.t_code_2, tf.range(16, 32), axis=-1)

        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code_1, "reuse":False})      
        
        # Code Disentangle =======================================================================================================
        code_dis_logits = mz.build_model({"mode":"code_cls", "code_cls_input":self.z1_1, "reuse":False})  
          
        self.code_dis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_dis_logits, labels=self.labels))
        self.code_dis_pred = tf.argmax(tf.nn.softmax(code_dis_logits), axis=1)
        code_dis_correct_pred = tf.equal(self.code_dis_pred, tf.argmax(self.labels, axis=1))
        self.code_dis_acc = tf.reduce_mean(tf.cast(code_dis_correct_pred, tf.float32))  
        
        # AE + WGAN =============================================================================================================   
        # Dis1
        self.dis_t_inputs = self.images
        self.dis_f_inputs = self.decoder_output
        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.dis_t_inputs + (1. - epsilon) * (self.dis_f_inputs)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=1))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        # Disentangle: z1 =======================================================================================================
#        sample_num = self.batch_size // self.class_num 
#        shuffled_z1 = tf.gather(self.z1, tf.random_shuffle(tf.range(0, sample_num)))
#        for i in range(sample_num, self.batch_size, sample_num):
#            curr_z1 = tf.gather(self.z1, tf.random_shuffle(tf.range(i, i+sample_num)))
#            shuffled_z1 = tf.concat([shuffled_z1, curr_z1], 0)

#        shuffled_z2 = tf.gather(self.z2, tf.random_shuffle(tf.range(sample_num, sample_num+sample_num)))
#        for i in range(sample_num+sample_num, self.batch_size, sample_num):            
#            curr_z2 = tf.gather(self.z2, tf.random_shuffle(tf.range(i, i+sample_num)))
#            shuffled_z2 = tf.concat([shuffled_z2, curr_z2], 0)
#        curr_z2 = tf.gather(self.z2, tf.random_shuffle(tf.range(0, sample_num)))
#        shuffled_z2 = tf.concat([shuffled_z2, curr_z2], 0)            
        
        self.dise1_t_inputs = tf.concat([self.z2_1, self.z2_1], 3)
        self.dise1_f_inputs = tf.concat([self.z2_1, self.z2_2], 3)
        dise1_t = mz.build_model({"mode":"dise1", "dis_input":self.dise1_t_inputs, "reuse":False})       
        dise1_f = mz.build_model({"mode":"dise1", "dis_input":self.dise1_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x1_hat = epsilon * self.dise1_t_inputs + (1. - epsilon) * (self.dise1_f_inputs)
        d1_hat = mz.build_model({"mode":"dise1", "dis_input":x1_hat, "reuse":True})
        
        d1_gp = tf.gradients(d1_hat, [x1_hat])[0]
        d1_gp = tf.sqrt(tf.reduce_sum(tf.square(d1_gp), axis=1))
        d1_gp = tf.reduce_mean((d1_gp - 1.0)**2) * 10

        dise1_ture_loss = tf.reduce_mean(dise1_t)
        dise1_fake_loss = tf.reduce_mean(dise1_f)
        
#        # Disentangle: z2 =======================================================================================================
#
#        self.dise2_t_inputs = self.z2_1
#        self.dise2_f_inputs = self.z2_2
#        dise2_t = mz.build_model({"mode":"dise2", "dis_input":self.dise2_t_inputs, "reuse":False})       
#        dise2_f = mz.build_model({"mode":"dise2", "dis_input":self.dise2_f_inputs, "reuse":True})       
#
#        #### WGAN-GP ####
#        # Calculate gradient penalty
#        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
#        x2_hat = epsilon * self.dise2_t_inputs + (1. - epsilon) * (self.dise2_f_inputs)
#        d2_hat = mz.build_model({"mode":"dise2", "dis_input":x2_hat, "reuse":True})
#        
#        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
#        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
#        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10
#
#        dise2_ture_loss = tf.reduce_mean(dise2_t)
#        dise2_fake_loss = tf.reduce_mean(dise2_f)

        # Generator =============================================================================================================
        self.f_code = self.z1_1
        self.f_images = mz.build_model({"mode":"generator", "code":self.f_code, "reuse":False})      
        self.f_code = mz.build_model({"mode":"encoder", "en_input":self.f_images, "reuse":True})       
        self.f_z1 = tf.gather(self.f_code, tf.range(0, 16), axis=-1)
        
        f_code_dis_logits = mz.build_model({"mode":"code_cls", "code_cls_input":self.f_z1, "reuse":True})  
                
        self.f_code_content_loss = tf.reduce_mean(tf.abs(self.f_z1 - self.z1_1))
        self.f_code_dis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f_code_dis_logits, labels=self.labels))        
        
        # Loss ==================================================================================================================
        
        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))

        self.g_loss_1 = 50*self.content_loss + disc_fake_loss + 2*dise1_fake_loss + self.cdis_coef*self.code_dis_loss
        self.d_loss_1 = -(disc_fake_loss - disc_ture_loss) + d_gp             
        self.dise_loss_1 = -(dise1_fake_loss - dise1_ture_loss) + d1_gp             
        #self.dise_loss_2 = -(dise2_fake_loss - dise2_ture_loss) + d2_gp 
        
        self.g_loss_2 = self.f_code_content_loss + self.f_code_dis_loss
                        
        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()       
        g1_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder", "code_cls"))]
        d1_var = [v for v in train_variables if v.name.startswith("discriminator")]
        dise1_var = [v for v in train_variables if v.name.startswith("dise1")]
        g2_var = [v for v in train_variables if v.name.startswith(("generator"))]
        
        self.train_g1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_1, var_list=g1_var)
        self.train_d1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss_1, var_list=d1_var)
        self.train_dise1 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.dise_loss_1, var_list=dise1_var)      
        
        self.train_g2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss_2, var_list=g2_var)
        
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['train'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['train'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['train'])
            tf.summary.scalar("dise_loss_1", self.dise_loss_1, collections=['train'])

            tf.summary.scalar("code_dis_loss", self.code_dis_loss, collections=['train'])
            tf.summary.scalar("code_dis_acc", self.code_dis_acc, collections=['train'])
            tf.summary.scalar("f_code_dis_loss", self.f_code_dis_loss, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("f_code_content_loss", self.f_code_content_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("f_image", self.f_images, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['test'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['test'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['test'])
            tf.summary.scalar("dise_loss_1", self.dise_loss_1, collections=['test'])

            tf.summary.scalar("code_dis_loss", self.code_dis_loss, collections=['test'])
            tf.summary.scalar("code_dis_acc", self.code_dis_acc, collections=['test'])
            tf.summary.scalar("f_code_dis_loss", self.f_code_dis_loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("f_code_content_loss", self.f_code_content_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("f_image", self.f_images, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):

            tf.summary.scalar("g_loss_1", self.g_loss_1, collections=['anomaly'])
            tf.summary.scalar("g_loss_2", self.g_loss_2, collections=['anomaly'])
            
            tf.summary.scalar("d_loss_1", self.d_loss_1, collections=['anomaly'])
            tf.summary.scalar("dise_loss_1", self.dise_loss_1, collections=['anomaly'])
                            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("f_code_content_loss", self.f_code_content_loss, collections=['anomaly'])
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            tf.summary.image("f_image", self.f_images, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_DISE(self):

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

    def train_AD_DISE(self):
        
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
                next_x_images, next_y = get_batch_by_class(self.dataset, self.batch_size, self.anoCls)
                next_x_images2, _ = get_batch_by_class(self.dataset, self.batch_size, self.anoCls) 
                  
                fd = {
                        self.images: next_x_images, 
                        self.images2: next_x_images2, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                     }
                               
                # Training                 
                sess.run(self.train_g1, feed_dict=fd) 
                
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
                    next_x_images, next_y = get_batch_by_class(self.dataset, self.batch_size, self.anoCls)
                    next_x_images2, _ = get_batch_by_class(self.dataset, self.batch_size, self.anoCls) 
                    
                    fd = {
                            self.images: next_x_images, 
                            self.images2: next_x_images2, 
                            self.labels: next_y, 
                            self.dropout_rate: self.dropout,
                            self.lr: new_learning_rate,
                         }                      

                    sess.run(self.train_d1, feed_dict=fd)
                    sess.run(self.train_dise1, feed_dict=fd)
    
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
                    
                    next_valid_x_images, next_valid_y = get_batch_by_class(self.valid_dataset, self.batch_size, self.anoCls)
                    next_valid_x_images2, _ = get_batch_by_class(self.valid_dataset, self.batch_size, self.anoCls) 
                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.images2: next_valid_x_images2, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.lr: new_learning_rate,
                              }
                                           
                    test_sum, test_loss = sess.run([self.merged_summary_test, self.g_loss_1], feed_dict=fd_test)  
                    
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
                                               
                    ano_sum, ano_loss = sess.run([self.merged_summary_anomaly, self.g_loss_1], feed_dict=fd_test) 
                      
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

    def test_AD_DISE(self):

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

    def build_AD_CLS_DISE(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
#        # Classifier =============================================================================================================
#        cls_logits, cls_feature = mz.build_model({"mode":"classifier", "cls_input":self.images, "reuse":False})  
#        
#        if self.z1_src == True:
#            self.z1 = tf.nn.softmax(cls_logits)
#        else:
#            self.z1 = self.labels
#                                   
#        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_logits, labels=self.labels))
#        self.cls_pred = tf.argmax(tf.nn.softmax(cls_logits), axis=1)
#        cls_correct_pred = tf.equal(self.cls_pred, tf.argmax(self.labels, axis=1))
#        self.cls_acc = tf.reduce_mean(tf.cast(cls_correct_pred, tf.float32))             
        
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z1_con, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z1_con, self.z2], -1)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      

        # 2nd Encode 
        self.z1_2nd, self.z1_con_2nd, self.z2_2nd = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})         
        
        # Backward ================================================================================================================
        # Decoder
        self.rdm_z1 = self.labels
        self.rdm_z1_con = tf.random_uniform([self.batch_size, 20], -1.0, 1.0)
        self.rdm_z2 = tf.random_uniform([self.batch_size, 128], -1.0, 1.0)
        self.rdm_code = tf.concat([self.rdm_z1, self.rdm_z1_con, self.rdm_z2], -1)
        self.rdm_decoder_output = mz.build_model({"mode":"decoder", "code":self.rdm_code, "reuse":True})      
        
        # Encoder
        self.f_z1, self.f_z1_con, self.f_z2 = mz.build_model({"mode":"encoder", "en_input":self.rdm_decoder_output, "reuse":True})         
        self.f_z1_softmax = tf.nn.softmax(self.f_z1)    
        
        # WGAN =============================================================================================================   
        self.dis_t_inputs = (self.images, self.t_code)
        self.dis_f_inputs = (self.rdm_decoder_output, self.rdm_code)
        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat_image = epsilon * self.dis_t_inputs[0] + (1. - epsilon) * (self.dis_f_inputs[0])
        epsilon = tf.squeeze(epsilon, axis=[2,3])
        x_hat_latent = epsilon * self.dis_t_inputs[1] + (1. - epsilon) * (self.dis_f_inputs[1])
        x_hat = (x_hat_image, x_hat_latent)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp_image = tf.gradients(d_hat, [x_hat[0]])[0]
        d_gp_latent = tf.gradients(d_hat, [x_hat[1]])[0]
        d_gp_image = tf.sqrt(tf.reduce_sum(tf.square(d_gp_image), axis=1)) 
        d_gp_latent = tf.sqrt(tf.reduce_sum(tf.square(d_gp_latent), axis=1))
        d_gp = tf.reduce_mean((d_gp_image - 1.0)**2) * 10 + tf.reduce_mean((d_gp_latent - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        # Discriminator 2 =======================================================================================================
        self.dis2_t_inputs = self.images
        self.dis2_f_inputs = self.decoder_output
        dis2_t = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_t_inputs, "reuse":False})       
        dis2_f = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x2_hat = epsilon * self.dis2_t_inputs + (1. - epsilon) * (self.dis2_f_inputs)
        d2_hat = mz.build_model({"mode":"discriminator2", "dis_input":x2_hat, "reuse":True})
        
        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10

        disc2_ture_loss = tf.reduce_mean(dis2_t)
        disc2_fake_loss = tf.reduce_mean(dis2_f)        
           
        # Loss ==================================================================================================================

        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))
        #self.content_loss = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        self.dise_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.f_z1, labels=self.labels)) + tf.reduce_mean(tf.squared_difference(self.rdm_z1_con, self.f_z1_con))# + tf.reduce_mean(tf.squared_difference(self.rdm_z2, self.f_z2))    
        self.code_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1, labels=self.labels))
        
        self.g_loss = 50*self.content_loss + 25*self.code_loss + disc2_fake_loss + (self.adv_weight * disc_fake_loss) + (1 * self.adv_weight * self.dise_loss)
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp                               
        self.d2_loss = -(disc2_fake_loss - disc2_ture_loss) + d2_gp          
        
        #_, f_cls_feature = mz.build_model({"mode":"classifier", "cls_input":self.decoder_output, "reuse":True}) 
        #self.feature_loss = tf.reduce_mean(tf.squared_difference(f_cls_feature, cls_feature))    
                    
        self.encode_loss_2nd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax))
        
        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()       
        g_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d_var = [v for v in train_variables if v.name.startswith("discriminator")]
        d2_var = [v for v in train_variables if v.name.startswith("discriminator2")]

        self.train_g = tf.train.AdamOptimizer(self.lr*3, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_var)
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_var)         
        self.train_d2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d2_loss, var_list=d2_var)         
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['train'])
            
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['train'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['test'])

            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])            
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['test'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['anomaly'])
                
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['anomaly'])
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['anomaly'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['anomaly'])            
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_AD_CLS_DISE(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)
        
        # Encoder
        self.z1, self.z1_con, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)
        
        # Code
        self.dise_code = tf.concat([self.z1_src, self.z1_con, self.z2], -1)
        self.t_code = tf.concat([self.z1_softmax, self.z1_con, self.z2], -1)
        
        # Decoder        
        self.dise_decoder_output = mz.build_model({"mode":"decoder", "code":self.dise_code, "reuse":False})      
        self.t_decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":True})      
        
        # fixed Encoder
        self.z1_2nd, _, _ = mz.build_model({"mode":"encoder", "en_input":self.t_decoder_output, "reuse":True})         
        
        self.anomaly_score = tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax)

        self.saver = tf.train.Saver()

    def train_AD_CLS_DISE(self):
        
        new_learning_rate = self.learn_rate_init

        init_adv_w = 0
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            #anomaly_dataset_idx = np.array(list(range(0, len(self.anomaly_dataset[0]))))

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:

                new_learning_rate = self.get_lr(step)
                
                # Get the training batch
                #next_x_images, next_y = get_batch_by_class(self.dataset, self.batch_size, self.anoCls)
                next_x_images, next_y = get_batch(self.dataset, self.batch_size)

                # adv weighting 
                if step <= 10000:
                    adv_weight = init_adv_w + step * (1.0/10000)
                else:
                    adv_weight = 1.0
                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.adv_weight: adv_weight,
                     }
                
                if step >= 0:
                    
                    # Training AE                
                    sess.run(self.train_g, feed_dict=fd) 
                                                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch 
                        #next_x_images, next_y = get_batch_by_class(self.dataset, self.batch_size, self.anoCls)
                        next_x_images, next_y = get_batch(self.dataset, self.batch_size)
                        
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.lr: new_learning_rate,
                                self.adv_weight: adv_weight,
                             }                      
    
                        sess.run(self.train_d, feed_dict=fd)
                        sess.run(self.train_d2, feed_dict=fd)
    
                # Update Learning rate                
                #if step == 25000 or step == 50000 or step == 65000 or step == 80000 or step == 95000 or step == 110000 or step == 125000:
                #    new_learning_rate = new_learning_rate * 0.5
                #    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                                
                #if step > 50000:
                #    new_learning_rate = 0.0001 * 0.25
                
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss = sess.run([self.merged_summary_train, self.g_loss], feed_dict=fd)
                    
                    #next_valid_x_images, next_valid_y = get_batch_by_class(self.valid_dataset, self.batch_size, self.anoCls)
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                           
                    test_sum, test_loss, test_z1 = sess.run([self.merged_summary_test, self.g_loss, self.z1_softmax], feed_dict=fd_test)  
                    
                    #curr_idx = 0
                    #random.shuffle(anomaly_dataset_idx)                   

                    next_ano_x_images, next_ano_y = get_batch(self.anomaly_dataset, self.batch_size)
                    #next_ano_x_images = self.anomaly_dataset[0][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]
                    #next_ano_y = self.anomaly_dataset[1][anomaly_dataset_idx[curr_idx:curr_idx+self.batch_size]]   
                    #curr_idx = curr_idx + self.batch_size
                    
                    fd_test = {
                                self.images: next_ano_x_images, 
                                self.labels: next_ano_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                               
                    ano_sum, ano_loss, ano_z1 = sess.run([self.merged_summary_anomaly, self.g_loss, self.z1_softmax], feed_dict=fd_test) 
                      
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (step, new_learning_rate, train_loss, test_loss, ano_loss))
                    
                    #print("Label z1: ", next_valid_y[0])
                    #print("Test z1: ", test_z1[0])
                    #print("Anomaly z1: ", ano_z1[0])
                    
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

    def test_AD_CLS_DISE(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            output_basename = os.path.basename(self.data_ob.test_images_path)
            output_path = os.path.join(self.output_dir, os.path.splitext(output_basename)[0])
            output_dise_path = os.path.join(output_path, 'Disentanglement')
            
            input_class = int(os.path.splitext(output_basename)[0].split(sep='_')[-1])
            print("Current input class: {}".format(input_class))
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if not os.path.exists(output_dise_path):
                os.makedirs(output_dise_path)
        
            encode_output_name = os.path.splitext(output_basename)[0] + "_encode.png"
            t_decode_output_name = os.path.splitext(output_basename)[0] + "_t_decode.png"
            ano_score_output_name = os.path.splitext(output_basename)[0] + "_ano_score.csv"
            
            for i in range(10):
                
                decode_output_name = os.path.splitext(output_basename)[0] + "_decode_{}.png".format(i)               
                print("Output: {}".format(decode_output_name))

                output_t_decode_image = []                
                output_decode_image = []
                output_encode_image = []
                output_ano_score = []
                
                z1_src = np.zeros(10)
                z1_src[i] = 1
                z1_src = np.expand_dims(z1_src, axis=0)
                z1_src = np.repeat(z1_src, self.batch_size, axis=0)
                
                #z1_con = np.random.uniform(-1, 1, (self.batch_size, 20))

                #z1_src = np.append(z1_src, z1_con, axis=-1)
                
                curr_idx = 0
                while True:
                    
                    try:
                        #next_x_images, next_test_y = sess.run(self.next_test_x)
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        curr_idx = curr_idx + self.batch_size
                        
                        if len(next_x_images) < self.batch_size:
                            break

                        if curr_idx > 256:
                            break
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    if i == input_class:
                        fd_ano_score = {
                                            self.images: next_x_images, 
                                            self.labels: next_test_y, 
                                            self.dropout_rate: 0,
                                            self.z1_src: z1_src,
                                       }                        
                        ano_score, t_decoder_output = sess.run([self.anomaly_score, self.t_decoder_output], feed_dict=fd_ano_score)                   
                        t_decoder_output = np.squeeze(np.array(t_decoder_output))

                        if output_t_decode_image == []:
                            output_t_decode_image = t_decoder_output
                            output_ano_score = ano_score
                        else:
                            output_t_decode_image = np.append(output_t_decode_image, t_decoder_output, axis=0)
                            output_ano_score = np.append(output_ano_score, ano_score, axis=0)
                        
                    fd_test = {
                                self.images: next_x_images, 
                                self.labels: next_test_y, 
                                self.dropout_rate: 0,
                                self.z1_src: z1_src,
                              }
    
                    dise_decoder_output = sess.run([self.dise_decoder_output], feed_dict=fd_test)               
                    dise_decoder_output = np.squeeze(np.array(dise_decoder_output))

                    if i == 0: # Only the first iteration needs to output the encode images
                        if output_encode_image == []:
                            output_encode_image = next_x_images     
                        else:
                            output_encode_image = np.append(output_encode_image, next_x_images, axis=0)
    
                    if output_decode_image == []:
                        output_decode_image = dise_decoder_output                       
                    else:
                        output_decode_image = np.append(output_decode_image, dise_decoder_output, axis=0)
                                        
                if i == 0:  # Only the first iteration needs to output the encode images
                    img_output(output_encode_image, os.path.join(output_dise_path, encode_output_name))                         
                    
                if i == input_class:
                    img_output(output_t_decode_image, os.path.join(output_path, t_decode_output_name))                          
                    np.savetxt(os.path.join(output_path, ano_score_output_name), output_ano_score, delimiter=",")
                
                img_output(output_decode_image, os.path.join(output_dise_path, decode_output_name))                                             
                    
#                print(output_decode_image.shape)
#                output_decode_image = img_merge(output_decode_image[0:256], [16, 16])
#                output_encode_image = img_merge(output_encode_image[0:256], [16, 16])
#
#                output_encode_image = output_encode_image * 255
#                output_decode_image = output_decode_image * 255
#                output_encode_image = output_encode_image.astype(np.uint8)
#                output_decode_image = output_decode_image.astype(np.uint8)
#                
#                imageio.imwrite(os.path.join(output_path, 'Disentanglement', decode_output_name), output_decode_image)
#                imageio.imwrite(os.path.join(output_path, 'Disentanglement', encode_output_name), output_encode_image)
                   

    def build_AD_CLS_DISE2(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model              
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z2], -1)
        self.zt_code = tf.concat([self.labels, self.z2], -1)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # 2nd Encode 
        self.z1_2nd, _ = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})          
        
        # Backward ================================================================================================================
        # Decoder
        self.rdm_z1 = self.labels
        self.rdm_z2 = tf.random_uniform([self.batch_size, self.lat_dim], -1.0, 1.0)
        self.rdm_code = tf.concat([self.rdm_z1, self.rdm_z2], -1)
        self.rdm_decoder_output = mz.build_model({"mode":"decoder", "code":self.rdm_code, "reuse":True})      

        # Encoder
        self.f_z1, self.f_z2 = mz.build_model({"mode":"encoder", "en_input":self.rdm_decoder_output, "reuse":True})         
        self.f_z1_softmax = tf.nn.softmax(self.f_z1)    
        
        # WGAN =============================================================================================================   
        #self.dis_t_inputs = (self.images, self.t_code)
        self.dis_t_inputs = (self.images, self.zt_code)
        self.dis_f_inputs = (self.rdm_decoder_output, self.rdm_code)
        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat_image = epsilon * self.dis_t_inputs[0] + (1. - epsilon) * (self.dis_f_inputs[0])
        epsilon = tf.squeeze(epsilon, axis=[2,3])
        x_hat_latent = epsilon * self.dis_t_inputs[1] + (1. - epsilon) * (self.dis_f_inputs[1])
        x_hat = (x_hat_image, x_hat_latent)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp_image = tf.gradients(d_hat, [x_hat[0]])[0]
        d_gp_latent = tf.gradients(d_hat, [x_hat[1]])[0]
        d_gp_image = tf.sqrt(tf.reduce_sum(tf.square(d_gp_image), axis=1)) 
        d_gp_latent = tf.sqrt(tf.reduce_sum(tf.square(d_gp_latent), axis=1))
        d_gp = tf.reduce_mean((d_gp_image - 1.0)**2) * 10 + tf.reduce_mean((d_gp_latent - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        # Discriminator 2 =======================================================================================================
        self.dis2_t_inputs = self.images
        self.dis2_f_inputs = self.decoder_output
        dis2_t = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_t_inputs, "reuse":False})       
        dis2_f = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x2_hat = epsilon * self.dis2_t_inputs + (1. - epsilon) * (self.dis2_f_inputs)
        d2_hat = mz.build_model({"mode":"discriminator2", "dis_input":x2_hat, "reuse":True})
        
        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10

        disc2_ture_loss = tf.reduce_mean(dis2_t)
        disc2_fake_loss = tf.reduce_mean(dis2_f)   
               
        # Loss ==================================================================================================================

        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))        
        self.code_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1, labels=self.labels))
        #self.dise_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.f_z1, labels=self.labels)) + tf.reduce_mean(tf.squared_difference(self.rdm_z2, self.f_z2))
        self.dise_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.f_z1, labels=self.labels)) + tf.reduce_mean(tf.abs(self.rdm_z2 - self.f_z2))

        self.encode_loss_2nd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax))
        
        self.g_loss = 75*self.content_loss + 25*self.code_loss + disc2_fake_loss + (1 * self.adv_weight * self.dise_loss) + (1 * self.adv_weight * disc_fake_loss)
        #self.g_loss = 50*self.content_loss + 25*self.code_loss + disc2_fake_loss + (self.dise_loss) + (disc_fake_loss)
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp                               
        self.d2_loss = -(disc2_fake_loss - disc2_ture_loss) + d2_gp                      
                
        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()       
        g_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d_var = [v for v in train_variables if v.name.startswith("discriminator")]
        d2_var = [v for v in train_variables if v.name.startswith("discriminator2")]
        
        self.train_g = tf.train.AdamOptimizer(self.lr*3, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_var)
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_var)         
        self.train_d2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d2_loss, var_list=d2_var)          
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['train'])
            
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['train'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['test'])

            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])            
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['test'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['anomaly'])
                
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['anomaly'])
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['anomaly'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['anomaly'])            
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()                    

    def build_eval_AD_CLS_DISE2(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z2], -1)
        self.dise_code = tf.concat([self.z1_src, self.z2], -1)
        
        # Decoder        
        self.dise_decoder_output = mz.build_model({"mode":"decoder", "code":self.dise_code, "reuse":False})      
        self.t_decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":True})      
        
        # 2nd Encode 
        self.z1_2nd, _ = mz.build_model({"mode":"encoder", "en_input":self.t_decoder_output, "reuse":True})          
        self.z1_2nd_softmax = tf.nn.softmax(self.z1_2nd)    
        
        self.anomaly_score = tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax)

        self.content_loss = tf.reduce_mean(tf.abs(self.t_decoder_output - self.images), axis=[1,2,3])

#        # WGAN =============================================================================================================   
#        self.dis_f_inputs = (self.t_decoder_output, self.t_code)
#        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":False})       
#
#        disc_fake_loss = tf.reduce_mean(dis_f, axis=1)
#        self.d2_loss = disc_fake_loss
        
#        # Discriminator 2 =======================================================================================================
#        self.dis2_f_inputs = self.t_decoder_output     
#        dis2_f = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":False})       
#
#        disc2_fake_loss = tf.reduce_mean(dis2_f, axis=[1,2,3])  
#
#        self.d2_loss = disc2_fake_loss

        self.saver = tf.train.Saver()
                
    def train_AD_CLS_DISE2(self):
        
        new_learning_rate = self.learn_rate_init

        init_adv_w = 0
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                start_step = self.restore_step                
                
            else:
                start_step = 0
            
            epoch_pbar = tqdm(range(start_step, self.max_iters+1))
            for step in epoch_pbar:   
                
                new_learning_rate = self.get_lr(step)
                
                # Get the training batch
                next_x_images, next_y = get_batch(self.dataset, self.batch_size)

                # adv weighting 
                if step <= 10000:
                    adv_weight = init_adv_w + step * (1.0/10000)
                else:
                    adv_weight = 1.0
                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.adv_weight: adv_weight,
                     }
                
                if step >= 0:
                    
                    # Training AE                
                    sess.run(self.train_g, feed_dict=fd) 
                                                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch 
                        next_x_images, next_y = get_batch(self.dataset, self.batch_size)
                        
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.lr: new_learning_rate,
                                self.adv_weight: adv_weight,
                             }                      
    
                        sess.run([self.train_d, self.train_d2], feed_dict=fd)
                        #sess.run(self.train_d2, feed_dict=fd)
                    
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_encode_loss_2nd = sess.run([self.merged_summary_train, self.encode_loss_2nd], feed_dict=fd)
                    
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                           
                    test_sum, test_encode_loss_2nd = sess.run([self.merged_summary_test, self.encode_loss_2nd], feed_dict=fd_test)  

                    next_ano_x_images, next_ano_y = get_batch(self.anomaly_dataset, self.batch_size)
                    
                    fd_test = {
                                self.images: next_ano_x_images, 
                                self.labels: next_ano_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                               
                    ano_sum, ano_encode_loss_2nd = sess.run([self.merged_summary_anomaly, self.encode_loss_2nd], feed_dict=fd_test) 
                      
                    print("[%s] Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (self.ckpt_name, step, new_learning_rate, train_encode_loss_2nd, test_encode_loss_2nd, ano_encode_loss_2nd))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) < abs(test_encode_loss_2nd) and step > 10000:
                        
                        best_loss = test_encode_loss_2nd
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_CLS_DISE2(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            output_basename = os.path.basename(self.data_ob.test_images_path)
            output_path = os.path.join(self.output_dir, os.path.splitext(output_basename)[0])
            output_dise_path = os.path.join(output_path, 'Disentanglement')
            
            input_class = int(os.path.splitext(output_basename)[0].split(sep='_')[-1])
            print("Current input class: {}".format(input_class))
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if not os.path.exists(output_dise_path):
                os.makedirs(output_dise_path)
        
            encode_output_name = os.path.splitext(output_basename)[0] + "_encode.png"
            t_decode_output_name = os.path.splitext(output_basename)[0] + "_t_decode.png"
            ano_score_output_name = os.path.splitext(output_basename)[0] + "_ano_score.csv"
            
            for i in range(10):
                
                decode_output_name = os.path.splitext(output_basename)[0] + "_decode_{}.png".format(i)               
                print("Output: {}".format(decode_output_name))

                output_t_decode_image = []                
                output_decode_image = []
                output_encode_image = []
                output_ano_score = np.empty((0, 22))
                
                z1_src = np.zeros(10)
                z1_src[i] = 1
                z1_src = np.expand_dims(z1_src, axis=0)
                z1_src = np.repeat(z1_src, self.batch_size, axis=0)
                
                curr_idx = 0
                while True:
                    
                    try:
                        #next_x_images, next_test_y = sess.run(self.next_test_x)
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        curr_idx = curr_idx + self.batch_size
                        
                        if len(next_x_images) < self.batch_size:
                            break

                        #if curr_idx > 256:
                        #    break
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    if i == input_class:
                        fd_ano_score = {
                                            self.images: next_x_images, 
                                            self.labels: next_test_y, 
                                            self.dropout_rate: 0,
                                            self.z1_src: z1_src,
                                       }                        
                        ano_score, z1_softmax, z1_2nd_softmax, t_decoder_output, content_loss = sess.run([self.anomaly_score, self.z1_softmax, self.z1_2nd_softmax, self.t_decoder_output, self.content_loss], feed_dict=fd_ano_score)                   
                        t_decoder_output = np.squeeze(np.array(t_decoder_output))
                        ano_score = np.expand_dims(ano_score, axis=-1) 
                        content_loss = np.expand_dims(content_loss, axis=-1) 

                        output_ano_score = np.append(output_ano_score, np.hstack((ano_score, z1_softmax, z1_2nd_softmax, content_loss)), axis=0)
                        
                        if curr_idx <= 256:
                            if output_t_decode_image == []:
                                output_t_decode_image = t_decoder_output                           
                            else:
                                output_t_decode_image = np.append(output_t_decode_image, t_decoder_output, axis=0)
                    
                    if curr_idx <= 256:
                        
                        fd_test = {
                                    self.images: next_x_images, 
                                    self.labels: next_test_y, 
                                    self.dropout_rate: 0,
                                    self.z1_src: z1_src,
                                  }
        
                        dise_decoder_output = sess.run([self.dise_decoder_output], feed_dict=fd_test)               
                        dise_decoder_output = np.squeeze(np.array(dise_decoder_output))

                        if i == 0: # Only the first iteration needs to output the encode images
                            if output_encode_image == []:
                                output_encode_image = next_x_images     
                            else:
                                output_encode_image = np.append(output_encode_image, next_x_images, axis=0)
        
                        if output_decode_image == []:
                            output_decode_image = dise_decoder_output                       
                        else:
                            output_decode_image = np.append(output_decode_image, dise_decoder_output, axis=0)
                                        
                if i == 0:  # Only the first iteration needs to output the encode images
                    img_output(output_encode_image, os.path.join(output_dise_path, encode_output_name))                         
                    
                if i == input_class:
                    img_output(output_t_decode_image, os.path.join(output_path, t_decode_output_name))                          
                    np.savetxt(os.path.join(output_path, ano_score_output_name), output_ano_score, delimiter=",")
                
                img_output(output_decode_image, os.path.join(output_dise_path, decode_output_name))                                             

    def build_AD_CLS_DISE3(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model              
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z2], -1)
        self.zt_code = tf.concat([self.labels, self.z2], -1)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # 2nd Encode 
        self.z1_2nd, _ = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})          
        
        # Backward ================================================================================================================
        # Decoder
        self.rdm_z1 = self.labels
        self.rdm_z2 = tf.random_uniform([self.batch_size, self.lat_dim], -1.0, 1.0)
        self.rdm_code = tf.concat([self.rdm_z1, self.rdm_z2], -1)
        self.rdm_decoder_output = mz.build_model({"mode":"decoder", "code":self.rdm_code, "reuse":True})      

        # Encoder
        self.f_z1, self.f_z2 = mz.build_model({"mode":"encoder", "en_input":self.rdm_decoder_output, "reuse":True})         
        self.f_z1_softmax = tf.nn.softmax(self.f_z1)    
        
#        # WGAN =============================================================================================================   
#        self.dis_t_inputs = (self.images, self.zt_code)
#        
#        half_size = tf.cast(tf.shape(self.rdm_decoder_output)[0]/2, dtype=tf.int32)
#        self.dis_f_decode_outputs_1 = tf.gather(self.rdm_decoder_output, tf.random_shuffle(tf.range(half_size)))
#        self.dis_f_decode_outputs_2 = tf.gather(self.decoder_output, tf.random_shuffle(tf.range(half_size)))
#        self.dis_f_decode_outputs = tf.concat([self.dis_f_decode_outputs_1, self.dis_f_decode_outputs_2], 0)
#        
#        half_size = tf.cast(tf.shape(self.rdm_code)[0]/2, dtype=tf.int32)
#        self.dis_f_code_1 = tf.gather(self.rdm_code, tf.random_shuffle(tf.range(half_size)))
#        self.dis_f_code_2 = tf.gather(self.t_code, tf.random_shuffle(tf.range(half_size)))
#        self.dis_f_code = tf.concat([self.dis_f_code_1, self.dis_f_code_2], 0)
#        
#        self.dis_f_inputs = (self.dis_f_decode_outputs, self.dis_f_code)
#
#        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
#        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
#        
#        #### WGAN-GP ####
#        # Calculate gradient penalty
#        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
#        x_hat_image = epsilon * self.dis_t_inputs[0] + (1. - epsilon) * (self.dis_f_inputs[0])
#        epsilon = tf.squeeze(epsilon, axis=[2,3])
#        x_hat_latent = epsilon * self.dis_t_inputs[1] + (1. - epsilon) * (self.dis_f_inputs[1])
#        x_hat = (x_hat_image, x_hat_latent)
#        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
#        
#        d_gp_image = tf.gradients(d_hat, [x_hat[0]])[0]
#        d_gp_latent = tf.gradients(d_hat, [x_hat[1]])[0]
#        d_gp_image = tf.sqrt(tf.reduce_sum(tf.square(d_gp_image), axis=1)) 
#        d_gp_latent = tf.sqrt(tf.reduce_sum(tf.square(d_gp_latent), axis=1))
#        d_gp = tf.reduce_mean((d_gp_image - 1.0)**2) * 10 + tf.reduce_mean((d_gp_latent - 1.0)**2) * 10
#
#        disc_ture_loss = tf.reduce_mean(dis_t)
#        disc_fake_loss = tf.reduce_mean(dis_f)
        
        # WGAN =============================================================================================================   
        self.dis_t_inputs = (self.images, self.zt_code)
                
        self.dis_f_inputs_1 = (self.rdm_decoder_output, self.rdm_code)
        self.dis_f_inputs_2 = (self.decoder_output, self.t_code)

        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f_1 = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs_1, "reuse":True})       
        dis_f_2 = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs_2, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat_image_1 = epsilon * self.dis_t_inputs[0] + (1. - epsilon) * (self.dis_f_inputs_1[0])
        epsilon = tf.squeeze(epsilon, axis=[2,3])
        x_hat_latent_1 = epsilon * self.dis_t_inputs[1] + (1. - epsilon) * (self.dis_f_inputs_1[1])
        x_hat_1 = (x_hat_image_1, x_hat_latent_1)
        d_hat_1 = mz.build_model({"mode":"discriminator", "dis_input":x_hat_1, "reuse":True})

        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat_image_2 = epsilon * self.dis_t_inputs[0] + (1. - epsilon) * (self.dis_f_inputs_2[0])
        epsilon = tf.squeeze(epsilon, axis=[2,3])
        x_hat_latent_2 = epsilon * self.dis_t_inputs[1] + (1. - epsilon) * (self.dis_f_inputs_2[1])
        x_hat_2 = (x_hat_image_2, x_hat_latent_2)
        d_hat_2 = mz.build_model({"mode":"discriminator", "dis_input":x_hat_2, "reuse":True})
        
        d_gp_image_1 = tf.gradients(d_hat_1, [x_hat_1[0]])[0]
        d_gp_latent_1 = tf.gradients(d_hat_1, [x_hat_1[1]])[0]
        d_gp_image_1 = tf.sqrt(tf.reduce_sum(tf.square(d_gp_image_1), axis=1)) 
        d_gp_latent_1 = tf.sqrt(tf.reduce_sum(tf.square(d_gp_latent_1), axis=1))
        d_gp_1 = tf.reduce_mean((d_gp_image_1 - 1.0)**2) * 10 + tf.reduce_mean((d_gp_latent_1 - 1.0)**2) * 10

        d_gp_image_2 = tf.gradients(d_hat_2, [x_hat_2[0]])[0]
        d_gp_latent_2 = tf.gradients(d_hat_2, [x_hat_2[1]])[0]
        d_gp_image_2 = tf.sqrt(tf.reduce_sum(tf.square(d_gp_image_2), axis=1)) 
        d_gp_latent_2 = tf.sqrt(tf.reduce_sum(tf.square(d_gp_latent_2), axis=1))
        d_gp_2 = tf.reduce_mean((d_gp_image_2 - 1.0)**2) * 10 + tf.reduce_mean((d_gp_latent_2 - 1.0)**2) * 10

        d_gp = d_gp_1 + d_gp_2

        disc_ture_loss_1 = tf.reduce_mean(dis_t)
        disc_fake_loss_1 = tf.reduce_mean(dis_f_1)        

        disc_ture_loss_2 = tf.reduce_mean(dis_t)
        disc_fake_loss_2 = tf.reduce_mean(dis_f_2)        

        disc_ture_loss = disc_ture_loss_1 + disc_ture_loss_2
        disc_fake_loss = disc_fake_loss_1 + disc_fake_loss_2        

        # Loss ==================================================================================================================

        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))        
        self.code_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1, labels=self.labels))
        self.dise_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.f_z1, labels=self.labels)) + tf.reduce_mean(tf.abs(self.rdm_z2 - self.f_z2))

        self.encode_loss_2nd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax))
        
        self.g_loss = 50*self.content_loss + 25*self.code_loss + (1 * self.adv_weight * self.dise_loss) + (1 * self.adv_weight * disc_fake_loss)
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp                               
                
        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()       
        g_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d_var = [v for v in train_variables if v.name.startswith("discriminator")]
        
        self.train_g = tf.train.AdamOptimizer(self.lr*3, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_var)
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_var)         
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['train'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])

            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])            
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['test'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
                
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['anomaly'])
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['anomaly'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['anomaly'])            
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()                    

    def build_eval_AD_CLS_DISE3(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z2], -1)
        self.dise_code = tf.concat([self.z1_src, self.z2], -1)
        
        # Decoder        
        self.dise_decoder_output = mz.build_model({"mode":"decoder", "code":self.dise_code, "reuse":False})      
        self.t_decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":True})      
        
        # 2nd Encode 
        self.z1_2nd, _ = mz.build_model({"mode":"encoder", "en_input":self.t_decoder_output, "reuse":True})          
        self.z1_2nd_softmax = tf.nn.softmax(self.z1_2nd)    
        
        self.anomaly_score = tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax)

        self.content_loss = tf.reduce_mean(tf.abs(self.t_decoder_output - self.images), axis=[1,2,3])

        self.saver = tf.train.Saver()
                
    def train_AD_CLS_DISE3(self):
        
        new_learning_rate = self.learn_rate_init

        init_adv_w = 0
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                start_step = self.restore_step                
                
            else:
                start_step = 0
            
            epoch_pbar = tqdm(range(start_step, self.max_iters+1))
            for step in epoch_pbar:   
                
                new_learning_rate = self.get_lr(step)
                
                # Get the training batch
                next_x_images, next_y = get_batch(self.dataset, self.batch_size)

                # adv weighting 
                if step <= 10000:
                    adv_weight = init_adv_w + step * (1.0/10000)
                else:
                    adv_weight = 1.0
                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.adv_weight: adv_weight,
                     }
                
                if step >= 0:
                    
                    # Training AE                
                    sess.run(self.train_g, feed_dict=fd) 
                                                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch 
                        next_x_images, next_y = get_batch(self.dataset, self.batch_size)
                        
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.lr: new_learning_rate,
                                self.adv_weight: adv_weight,
                             }                      
    
                        sess.run([self.train_d], feed_dict=fd)
                    
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_encode_loss_2nd = sess.run([self.merged_summary_train, self.encode_loss_2nd], feed_dict=fd)
                    
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                           
                    test_sum, test_encode_loss_2nd = sess.run([self.merged_summary_test, self.encode_loss_2nd], feed_dict=fd_test)  

                    next_ano_x_images, next_ano_y = get_batch(self.anomaly_dataset, self.batch_size)
                    
                    fd_test = {
                                self.images: next_ano_x_images, 
                                self.labels: next_ano_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                               
                    ano_sum, ano_encode_loss_2nd = sess.run([self.merged_summary_anomaly, self.encode_loss_2nd], feed_dict=fd_test) 
                      
                    print("[%s] Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (self.ckpt_name, step, new_learning_rate, train_encode_loss_2nd, test_encode_loss_2nd, ano_encode_loss_2nd))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) < abs(test_encode_loss_2nd) and step > 10000:
                        
                        best_loss = test_encode_loss_2nd
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_CLS_DISE3(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            output_basename = os.path.basename(self.data_ob.test_images_path)
            output_path = os.path.join(self.output_dir, os.path.splitext(output_basename)[0])
            output_dise_path = os.path.join(output_path, 'Disentanglement')
            
            input_class = int(os.path.splitext(output_basename)[0].split(sep='_')[-1])
            print("Current input class: {}".format(input_class))
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if not os.path.exists(output_dise_path):
                os.makedirs(output_dise_path)
        
            encode_output_name = os.path.splitext(output_basename)[0] + "_encode.png"
            t_decode_output_name = os.path.splitext(output_basename)[0] + "_t_decode.png"
            ano_score_output_name = os.path.splitext(output_basename)[0] + "_ano_score.csv"
            
            for i in range(10):
                
                decode_output_name = os.path.splitext(output_basename)[0] + "_decode_{}.png".format(i)               
                print("Output: {}".format(decode_output_name))

                output_t_decode_image = []                
                output_decode_image = []
                output_encode_image = []
                output_ano_score = np.empty((0, 22))
                
                z1_src = np.zeros(10)
                z1_src[i] = 1
                z1_src = np.expand_dims(z1_src, axis=0)
                z1_src = np.repeat(z1_src, self.batch_size, axis=0)
                
                curr_idx = 0
                while True:
                    
                    try:
                        #next_x_images, next_test_y = sess.run(self.next_test_x)
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        curr_idx = curr_idx + self.batch_size
                        
                        if len(next_x_images) < self.batch_size:
                            break

                        #if curr_idx > 256:
                        #    break
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    if i == input_class:
                        fd_ano_score = {
                                            self.images: next_x_images, 
                                            self.labels: next_test_y, 
                                            self.dropout_rate: 0,
                                            self.z1_src: z1_src,
                                       }                        
                        ano_score, z1_softmax, z1_2nd_softmax, t_decoder_output, content_loss = sess.run([self.anomaly_score, self.z1_softmax, self.z1_2nd_softmax, self.t_decoder_output, self.content_loss], feed_dict=fd_ano_score)                   
                        t_decoder_output = np.squeeze(np.array(t_decoder_output))
                        ano_score = np.expand_dims(ano_score, axis=-1) 
                        content_loss = np.expand_dims(content_loss, axis=-1) 

                        output_ano_score = np.append(output_ano_score, np.hstack((ano_score, z1_softmax, z1_2nd_softmax, content_loss)), axis=0)
                        
                        if curr_idx <= 256:
                            if output_t_decode_image == []:
                                output_t_decode_image = t_decoder_output                           
                            else:
                                output_t_decode_image = np.append(output_t_decode_image, t_decoder_output, axis=0)
                    
                    if curr_idx <= 256:
                        
                        fd_test = {
                                    self.images: next_x_images, 
                                    self.labels: next_test_y, 
                                    self.dropout_rate: 0,
                                    self.z1_src: z1_src,
                                  }
        
                        dise_decoder_output = sess.run([self.dise_decoder_output], feed_dict=fd_test)               
                        dise_decoder_output = np.squeeze(np.array(dise_decoder_output))

                        if i == 0: # Only the first iteration needs to output the encode images
                            if output_encode_image == []:
                                output_encode_image = next_x_images     
                            else:
                                output_encode_image = np.append(output_encode_image, next_x_images, axis=0)
        
                        if output_decode_image == []:
                            output_decode_image = dise_decoder_output                       
                        else:
                            output_decode_image = np.append(output_decode_image, dise_decoder_output, axis=0)
                                        
                if i == 0:  # Only the first iteration needs to output the encode images
                    img_output(output_encode_image, os.path.join(output_dise_path, encode_output_name))                         
                    
                if i == input_class:
                    img_output(output_t_decode_image, os.path.join(output_path, t_decode_output_name))                          
                    np.savetxt(os.path.join(output_path, ano_score_output_name), output_ano_score, delimiter=",")
                
                img_output(output_decode_image, os.path.join(output_dise_path, decode_output_name))                            

    def build_AD_CLS_DISE4(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model              
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z2], -1)
        self.zt_code = tf.concat([self.labels, self.z2], -1)
        
        # Decoder        
        self.decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":False})      
        
        # 2nd Encode 
        self.z1_2nd, _ = mz.build_model({"mode":"encoder", "en_input":self.decoder_output, "reuse":True})          
        
        # Backward ================================================================================================================
        # Decoder
        self.rdm_z1 = self.labels
        self.rdm_z2 = tf.random_uniform([self.batch_size, self.lat_dim], -1.0, 1.0)
        self.rdm_code = tf.concat([self.rdm_z1, self.rdm_z2], -1)
        self.rdm_decoder_output = mz.build_model({"mode":"decoder", "code":self.rdm_code, "reuse":True})      

        # Encoder
        self.f_z1, self.f_z2 = mz.build_model({"mode":"encoder", "en_input":self.rdm_decoder_output, "reuse":True})         
        self.f_z1_softmax = tf.nn.softmax(self.f_z1)    
        
        # WGAN =============================================================================================================   
        self.dis_t_inputs = (self.images, self.zt_code)
        self.dis_f_inputs = (self.rdm_decoder_output, self.rdm_code)
        dis_t = mz.build_model({"mode":"discriminator", "dis_input":self.dis_t_inputs, "reuse":False})       
        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":True})       
        
        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat_image = epsilon * self.dis_t_inputs[0] + (1. - epsilon) * (self.dis_f_inputs[0])
        epsilon = tf.squeeze(epsilon, axis=[2,3])
        x_hat_latent = epsilon * self.dis_t_inputs[1] + (1. - epsilon) * (self.dis_f_inputs[1])
        x_hat = (x_hat_image, x_hat_latent)
        d_hat = mz.build_model({"mode":"discriminator", "dis_input":x_hat, "reuse":True})
        
        d_gp_image = tf.gradients(d_hat, [x_hat[0]])[0]
        d_gp_latent = tf.gradients(d_hat, [x_hat[1]])[0]
        d_gp_image = tf.sqrt(tf.reduce_sum(tf.square(d_gp_image), axis=1)) 
        d_gp_latent = tf.sqrt(tf.reduce_sum(tf.square(d_gp_latent), axis=1))
        d_gp = tf.reduce_mean((d_gp_image - 1.0)**2) * 10 + tf.reduce_mean((d_gp_latent - 1.0)**2) * 10

        disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        # Discriminator 2 =======================================================================================================
        self.dis2_t_inputs = self.images
        self.dis2_f_inputs = self.decoder_output
        dis2_t = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_t_inputs, "reuse":False})       
        dis2_f = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":True})       

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        x2_hat = epsilon * self.dis2_t_inputs + (1. - epsilon) * (self.dis2_f_inputs)
        d2_hat = mz.build_model({"mode":"discriminator2", "dis_input":x2_hat, "reuse":True})
        
        d2_gp = tf.gradients(d2_hat, [x2_hat])[0]
        d2_gp = tf.sqrt(tf.reduce_sum(tf.square(d2_gp), axis=1))
        d2_gp = tf.reduce_mean((d2_gp - 1.0)**2) * 10

        disc2_ture_loss = tf.reduce_mean(dis2_t)
        disc2_fake_loss = tf.reduce_mean(dis2_f)   
               
        # Loss ==================================================================================================================

        self.content_loss = tf.reduce_mean(tf.abs(self.decoder_output - self.images))        
        self.code_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1, labels=self.labels))
        self.dise_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.f_z1, labels=self.labels)) + tf.reduce_mean(tf.abs(self.rdm_z2 - self.f_z2))

        self.encode_loss_2nd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax))
        
        self.g_loss = 50*self.content_loss + 1*self.code_loss + disc2_fake_loss + (1 * self.adv_weight * self.dise_loss) + (1 * self.adv_weight * disc_fake_loss)
        self.d_loss = -(disc_fake_loss - disc_ture_loss) + d_gp                               
        self.d2_loss = -(disc2_fake_loss - disc2_ture_loss) + d2_gp                      
                
        # ========================================================================================================================        
        
        MSE = tf.reduce_mean(tf.squared_difference(self.decoder_output, self.images))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        train_variables = tf.trainable_variables()       
        g_var = [v for v in train_variables if v.name.startswith(("encoder", "decoder"))]
        d_var = [v for v in train_variables if v.name.startswith("discriminator")]
        d2_var = [v for v in train_variables if v.name.startswith("discriminator2")]
        
        self.train_g = tf.train.AdamOptimizer(self.lr*3, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_var)
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_var)         
        self.train_d2 = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d2_loss, var_list=d2_var)          
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['train'])
            
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['train'])
            
            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR", PSNR, collections=['train'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['train'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['train'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['train'])
            
            tf.summary.image("input_image", self.images, collections=['train'])
            tf.summary.image("output_image", self.decoder_output, collections=['train']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['train']) 
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['test'])

            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR", PSNR, collections=['test'])            
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['test'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['test'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['test'])
            
            tf.summary.image("input_image", self.images, collections=['test'])
            tf.summary.image("output_image", self.decoder_output, collections=['test']) 
            tf.summary.image("rdm_decode", self.rdm_decoder_output, collections=['test']) 
            
            self.merged_summary_test = tf.summary.merge_all('test')          

        with tf.name_scope('anomaly_summary'):

            tf.summary.scalar("g_loss", self.g_loss, collections=['anomaly'])           
            tf.summary.scalar("d_loss", self.d_loss, collections=['anomaly'])
            tf.summary.scalar("d2_loss", self.d2_loss, collections=['anomaly'])
                
            tf.summary.scalar("encode_loss_2nd", self.encode_loss_2nd, collections=['anomaly'])
            
            tf.summary.scalar("MSE", MSE, collections=['anomaly'])
            tf.summary.scalar("PSNR", PSNR, collections=['anomaly'])
            
            tf.summary.scalar("content_loss", self.content_loss, collections=['anomaly'])
            tf.summary.scalar("dise_loss", self.dise_loss, collections=['anomaly'])
            tf.summary.scalar("code_loss", self.code_loss, collections=['anomaly'])            
            
            tf.summary.image("input_image", self.images, collections=['anomaly'])
            tf.summary.image("output_image", self.decoder_output, collections=['anomaly']) 
            
            self.merged_summary_anomaly = tf.summary.merge_all('anomaly')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()                    

    def build_eval_AD_CLS_DISE4(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        # Autoencoder ============================================================================================================
        # Forward ================================================================================================================
        # Encoder
        self.z1, self.z2 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
        
        # Code
        self.t_code = tf.concat([self.z1_softmax, self.z2], -1)
        self.dise_code = tf.concat([self.z1_src, self.z2], -1)
        
        # Decoder        
        self.dise_decoder_output = mz.build_model({"mode":"decoder", "code":self.dise_code, "reuse":False})      
        self.t_decoder_output = mz.build_model({"mode":"decoder", "code":self.t_code, "reuse":True})      
        
        # 2nd Encode 
        self.z1_2nd, self.z2_2nd = mz.build_model({"mode":"encoder", "en_input":self.t_decoder_output, "reuse":True})          
        self.z1_2nd_softmax = tf.nn.softmax(self.z1_2nd)    
        
        # 2nd Code
        self.t_code_2nd = tf.concat([self.z1_2nd_softmax, self.z2_2nd], -1)

        # Decoder        
        self.t_decoder_output_2nd = mz.build_model({"mode":"decoder", "code":self.t_code_2nd, "reuse":True})      
        
        self.anomaly_score = tf.nn.softmax_cross_entropy_with_logits(logits=self.z1_2nd, labels=self.z1_softmax)

        self.content_loss = tf.reduce_mean(tf.abs(self.t_decoder_output - self.images), axis=[1,2,3])
        
        self.content_loss_2nd = tf.reduce_mean(tf.abs(self.t_decoder_output_2nd - self.images), axis=[1,2,3])

#        # WGAN =============================================================================================================   
#        self.dis_f_inputs = (self.t_decoder_output, self.t_code)
#        dis_f = mz.build_model({"mode":"discriminator", "dis_input":self.dis_f_inputs, "reuse":False})       
#
#        disc_fake_loss = tf.reduce_mean(dis_f, axis=1)
#        self.d2_loss = disc_fake_loss
        
#        # Discriminator 2 =======================================================================================================
#        self.dis2_f_inputs = self.t_decoder_output     
#        dis2_f = mz.build_model({"mode":"discriminator2", "dis_input":self.dis2_f_inputs, "reuse":False})       
#
#        disc2_fake_loss = tf.reduce_mean(dis2_f, axis=[1,2,3])  
#
#        self.d2_loss = disc2_fake_loss

        self.saver = tf.train.Saver()
                
    def train_AD_CLS_DISE4(self):
        
        new_learning_rate = self.learn_rate_init

        init_adv_w = 0
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                start_step = self.restore_step                
                
            else:
                start_step = 0
            
            epoch_pbar = tqdm(range(start_step, self.max_iters+1))
            for step in epoch_pbar:   
                
                new_learning_rate = self.get_lr(step)
                
                # Get the training batch
                next_x_images, next_y = get_batch(self.dataset, self.batch_size)

                # adv weighting 
                if step <= 10000:
                    adv_weight = init_adv_w + step * (1.0/10000)
                else:
                    adv_weight = 1.0
                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.adv_weight: adv_weight,
                     }
                
                if step >= 0:
                    
                    # Training AE                
                    sess.run(self.train_g, feed_dict=fd) 
                                                    
                    # Training Discriminator
                    for d_iter in range(0, 5):
                        
                        # Get the training batch 
                        next_x_images, next_y = get_batch(self.dataset, self.batch_size)
                        
                        fd = {
                                self.images: next_x_images, 
                                self.labels: next_y, 
                                self.dropout_rate: self.dropout,
                                self.lr: new_learning_rate,
                                self.adv_weight: adv_weight,
                             }                      
    
                        sess.run([self.train_d, self.train_d2], feed_dict=fd)
                    
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_encode_loss_2nd = sess.run([self.merged_summary_train, self.encode_loss_2nd], feed_dict=fd)
                    
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                           
                    test_sum, test_encode_loss_2nd = sess.run([self.merged_summary_test, self.encode_loss_2nd], feed_dict=fd_test)  

                    next_ano_x_images, next_ano_y = get_batch(self.anomaly_dataset, self.batch_size)
                    
                    fd_test = {
                                self.images: next_ano_x_images, 
                                self.labels: next_ano_y, 
                                self.dropout_rate: 0,
                                self.adv_weight: adv_weight,
                              }
                                               
                    ano_sum, ano_encode_loss_2nd = sess.run([self.merged_summary_anomaly, self.encode_loss_2nd], feed_dict=fd_test) 
                      
                    print("[%s] Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Anomaly loss = [%.7f]" % (self.ckpt_name, step, new_learning_rate, train_encode_loss_2nd, test_encode_loss_2nd, ano_encode_loss_2nd))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   
                    summary_writer.add_summary(ano_sum, step)         

                    if abs(best_loss) < abs(test_encode_loss_2nd) and step > 10000:
                        
                        best_loss = test_encode_loss_2nd
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step , 2000) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_AD_CLS_DISE4(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            output_basename = os.path.basename(self.data_ob.test_images_path)
            output_path = os.path.join(self.output_dir, os.path.splitext(output_basename)[0])
            output_dise_path = os.path.join(output_path, 'Disentanglement')
            
            input_class = int(os.path.splitext(output_basename)[0].split(sep='_')[-1])
            print("Current input class: {}".format(input_class))
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if not os.path.exists(output_dise_path):
                os.makedirs(output_dise_path)
        
            encode_output_name = os.path.splitext(output_basename)[0] + "_encode.png"
            t_decode_output_name = os.path.splitext(output_basename)[0] + "_t_decode.png"
            ano_score_output_name = os.path.splitext(output_basename)[0] + "_ano_score.csv"
            
            for i in range(10):
                
                decode_output_name = os.path.splitext(output_basename)[0] + "_decode_{}.png".format(i)               
                print("Output: {}".format(decode_output_name))

                output_t_decode_image = []                
                output_decode_image = []
                output_encode_image = []
                output_ano_score = np.empty((0, 22))
                
                z1_src = np.zeros(10)
                z1_src[i] = 1
                z1_src = np.expand_dims(z1_src, axis=0)
                z1_src = np.repeat(z1_src, self.batch_size, axis=0)
                
                curr_idx = 0
                while True:
                    
                    try:
                        #next_x_images, next_test_y = sess.run(self.next_test_x)
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        curr_idx = curr_idx + self.batch_size
                        
                        if len(next_x_images) < self.batch_size:
                            break

                        #if curr_idx > 256:
                        #    break
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    if i == input_class:
                        fd_ano_score = {
                                            self.images: next_x_images, 
                                            self.labels: next_test_y, 
                                            self.dropout_rate: 0,
                                            self.z1_src: z1_src,
                                       }                        
                        ano_score, z1_softmax, z1_2nd_softmax, t_decoder_output, content_loss = sess.run([self.anomaly_score, self.z1_softmax, self.z1_2nd_softmax, self.t_decoder_output, self.content_loss_2nd], feed_dict=fd_ano_score)                   
                        t_decoder_output = np.squeeze(np.array(t_decoder_output))
                        ano_score = np.expand_dims(ano_score, axis=-1) 
                        content_loss = np.expand_dims(content_loss, axis=-1) 

                        output_ano_score = np.append(output_ano_score, np.hstack((ano_score, z1_softmax, z1_2nd_softmax, content_loss)), axis=0)
                        
                        if curr_idx <= 256:
                            if output_t_decode_image == []:
                                output_t_decode_image = t_decoder_output                           
                            else:
                                output_t_decode_image = np.append(output_t_decode_image, t_decoder_output, axis=0)
                    
                    if curr_idx <= 256:
                        
                        fd_test = {
                                    self.images: next_x_images, 
                                    self.labels: next_test_y, 
                                    self.dropout_rate: 0,
                                    self.z1_src: z1_src,
                                  }
        
                        dise_decoder_output = sess.run([self.dise_decoder_output], feed_dict=fd_test)               
                        dise_decoder_output = np.squeeze(np.array(dise_decoder_output))

                        if i == 0: # Only the first iteration needs to output the encode images
                            if output_encode_image == []:
                                output_encode_image = next_x_images     
                            else:
                                output_encode_image = np.append(output_encode_image, next_x_images, axis=0)
        
                        if output_decode_image == []:
                            output_decode_image = dise_decoder_output                       
                        else:
                            output_decode_image = np.append(output_decode_image, dise_decoder_output, axis=0)
                                        
                if i == 0:  # Only the first iteration needs to output the encode images
                    img_output(output_encode_image, os.path.join(output_dise_path, encode_output_name))                         
                    
                if i == input_class:
                    img_output(output_t_decode_image, os.path.join(output_path, t_decode_output_name))                          
                    np.savetxt(os.path.join(output_path, ano_score_output_name), output_ano_score, delimiter=",")
                
                img_output(output_decode_image, os.path.join(output_dise_path, decode_output_name))  
                    
    def build_AD_CLS_BASELINE(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model              
        self.z1 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    
               
        # Loss ==================================================================================================================       
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.z1, labels=self.labels))

        self.pred = tf.argmax(self.z1_softmax, axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.labels, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
                
        # ========================================================================================================================               
        train_variables = tf.trainable_variables()       
        var = [v for v in train_variables if v.name.startswith("encoder")]
        
        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=var)
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("loss", self.loss, collections=['train'])           
            tf.summary.scalar("acc", self.acc, collections=['train'])           
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.loss, collections=['test'])           
            tf.summary.scalar("acc", self.acc, collections=['test'])           
            
            self.merged_summary_test = tf.summary.merge_all('test')                     
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()                    

    def build_eval_AD_CLS_BASELINE(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, dropout=self.dropout, lat_dim=self.lat_dim, is_training=self.is_training, model_ticket=self.model_ticket)        

        self.z1 = mz.build_model({"mode":"encoder", "en_input":self.images, "reuse":False})         
        self.z1_softmax = tf.nn.softmax(self.z1)    

        self.saver = tf.train.Saver()
                
    def train_AD_CLS_BASELINE(self):
        
        new_learning_rate = self.learn_rate_init
             
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                start_step = self.restore_step                
                
            else:
                start_step = 0
            
            epoch_pbar = tqdm(range(start_step, self.max_iters+1))
            for step in epoch_pbar:   
                
                new_learning_rate = self.get_lr(step)
                
                # Get the training batch
                next_x_images, next_y = get_batch(self.dataset, self.batch_size)
                    
                fd = {
                        self.images: next_x_images, 
                        self.labels: next_y, 
                        self.lr: new_learning_rate,
                        self.dropout_rate: self.dropout,
                     }
                
                sess.run(self.train_op, feed_dict=fd) 
                    
                # Record
                if step%200 == 0:

                    # Training set
                    train_sum, train_loss, train_acc = sess.run([self.merged_summary_train, self.loss, self.acc], feed_dict=fd)
                    
                    next_valid_x_images, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.images: next_valid_x_images, 
                                self.labels: next_valid_y, 
                                self.dropout_rate: 0,
                              }
                                           
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.loss, self.acc], feed_dict=fd_test)  
                                                                     
                    print("[%s] Step %d: LR = [%.7f], Train loss = [%.7f], Test loss = [%.7f], Train acc = [%.7f], Test acc = [%.7f]" % (self.ckpt_name, step, new_learning_rate, train_loss, test_loss, train_acc, test_acc))
                    
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

    def test_AD_CLS_BASELINE(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
           
            # Initialzie the iterator
            #sess.run(self.testing_init_op)

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            

            output_basename = os.path.basename(self.data_ob.test_images_path)
            output_path = os.path.join(self.output_dir, os.path.splitext(output_basename)[0])
            
            input_class = int(os.path.splitext(output_basename)[0].split(sep='_')[-1])
            print("Current input class: {}".format(input_class))
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
      
            ano_score_output_name = os.path.splitext(output_basename)[0] + "_ano_score.csv"
                           
            print("Output: {}".format(ano_score_output_name))

            output_ano_score = np.empty((0, 10))
            
            curr_idx = 0
            while True:
                
                try:
                    
                    if curr_idx >= len(self.test_dataset[0]):
                        break
                    
                    next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                    next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                    curr_idx = curr_idx + self.batch_size
                    
                    if len(next_x_images) < self.batch_size:
                        break
                    
                except tf.errors.OutOfRangeError:
                    break
                
                    
                fd_ano_score = {
                                    self.images: next_x_images, 
                                    self.labels: next_test_y, 
                                    self.dropout_rate: 0,
                               }                        
                
                z1_softmax = sess.run([self.z1_softmax], feed_dict=fd_ano_score)                   
                z1_softmax = np.squeeze(z1_softmax)
                output_ano_score = np.append(output_ano_score, z1_softmax, axis=0)
                
            np.savetxt(os.path.join(output_path, ano_score_output_name), output_ano_score, delimiter=",")








