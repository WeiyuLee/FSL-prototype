import tensorflow as tf
import netfactory as nf
import numpy as np
import math

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training

    def build_model(self, kwargs = {}):

        model_list = ["cifar10_alexnet_att", "BCL", "BCL_att", "BCL_att_v2", "BCL_att_v3", "BCL_att_v4", "BCL_att_GAN", 
                      "AD_att_GAN", "AD_att_GAN_v2", "AD_att_VAE", "AD_att_VAE_WEAK", "AD_att_VAE_GAN", "AD_att_AE_GAN", "AD_att_AE_GAN_3DCode", "RaGAN_MNIST"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
           
            fn = getattr(self,self.model_ticket)
            
            if kwargs == {}:
                netowrk = fn()
            else:
                netowrk = fn(kwargs)
            return netowrk
        
    def conv2d(self, name, tensor,ksize, out_dim, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, tensor.get_shape()[-1],out_dim], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d(tensor,w,[1,stride, stride,1],padding=padding)
            b = tf.get_variable('b', [out_dim], 'float32',initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)
    
    def deconv2d(self, name, tensor, ksize, outshape, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
            b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)        

    def fully_connected(self, name,value, output_shape):
        with tf.variable_scope(name, reuse=None) as scope:
            shape = value.get_shape().as_list()
            w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    
            return tf.matmul(value, w) + b

    def cifar10_alexnet_att(self, kwargs):
        
        reuse = kwargs["reuse"]
        is_training = kwargs["is_training"]
        
        #init = tf.random_normal_initializer(stddev=0.01)
        dropout = 0.9            
        model_params = {

                        'conv1': [3,3,24],
                        'conv2': [3,3,96],
                        'conv3': [3,3,192],
                        'conv4': [3,3,192],
                        'conv5': [3,3,96],
                        'fc1': 1024,
                        'fc2': 1024,
                        'fc3': 10,

                        }        
       
        with tf.variable_scope("ALEXNET_att", reuse=reuse):  
            
            conv1 = nf.convolution_layer(self.inputs, model_params['conv1'], [1,1,1,1], name="conv1")
            conv1, conv1_att = nf.channel_attention(conv1, name='conv1_att')
            conv1_mp = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')

            conv2 = nf.convolution_layer(conv1_mp, model_params['conv2'], [1,1,1,1], name="conv2")
            conv2, conv2_att = nf.channel_attention(conv2, name='conv2_att')
            conv2_mp = tf.nn.max_pool(conv2, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')
            
            conv3 = nf.convolution_layer(conv2_mp, model_params['conv3'], [1,1,1,1], name="conv3")
            conv3, conv3_att = nf.channel_attention(conv3, name='conv3_att')
            
            conv4 = nf.convolution_layer(conv3, model_params['conv4'], [1,1,1,1], name="conv4")
            conv4, conv4_att = nf.channel_attention(conv4, name='conv4_att')
            
            conv5 = nf.convolution_layer(conv4, model_params['conv5'], [1,1,1,1], name="conv5")
            conv5, conv5_att = nf.channel_attention(conv5, name='conv5_att')
            conv5_mp = tf.nn.max_pool(conv5, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')
            
            bsize, a, b, c = conv5_mp.get_shape().as_list()
            conv5_mp_flatten = tf.reshape(conv5_mp, [bsize, int(np.prod(conv5_mp.get_shape()[1:]))])                    
                
            fc1 = nf.fc_layer(conv5_mp_flatten, model_params['fc1'], name="fc1")
            
            fc2 = nf.fc_layer(fc1, model_params['fc2'], name="fc2")
            
            fc2 = tf.layers.dropout(fc2, rate=dropout, training=self.is_training, name='dropout')
            
            logits = nf.fc_layer(fc2, model_params['fc3'], name="logits", activat_fn=None)
            
            return logits, fc2, conv1_att, conv2_att, conv3_att, conv4_att, conv5_att          

    def BCL(self, kwargs):
        
        reuse = kwargs["reuse"]
        is_training = kwargs["is_training"]
        
        #init = tf.random_normal_initializer(stddev=0.01)
        dropout = 0.9            
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 10,

                        }        
       
        with tf.variable_scope("BCL", reuse=reuse):  
            
            conv11 = nf.convolution_layer(self.inputs, model_params['conv11'], [1,1,1,1], name="conv11")
            conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12")
            conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21")
            conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22")
            conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
            
            conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31")
            conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32")
            conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33")
            conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34")
            conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
            
            bsize, a, b, c = conv34_mp.get_shape().as_list()
            conv34_mp_flatten = tf.reshape(conv34_mp, [bsize, int(np.prod(conv34_mp.get_shape()[1:]))])                    
                
            fc4 = nf.fc_layer(conv34_mp_flatten, model_params['fc4'], name="fc4")
            
            fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout')
            
            fc5 = nf.fc_layer(fc4, model_params['fc5'], name="fc5")
            
            fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout')
            
            logits = nf.fc_layer(fc5, model_params['fc6'], name="logits", activat_fn=None)
            
            return logits, fc5

    def BCL_att(self, kwargs):
        
        reuse = kwargs["reuse"]
        
        #init = tf.random_normal_initializer(stddev=0.01)
        l2_reg = None#tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 10,

                        }        
       
        with tf.variable_scope("BCL", reuse=reuse):  
            
            conv11 = nf.convolution_layer(self.inputs, model_params['conv11'], [1,1,1,1], name="conv11", reg=l2_reg)
            conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", reg=l2_reg)
            conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", reg=l2_reg)
            conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", reg=l2_reg)
            conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
            
            conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", reg=l2_reg)
            conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", reg=l2_reg)
            conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", reg=l2_reg)
            conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", reg=l2_reg)           
            conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv34_mp, att = nf.channel_attention(conv34_mp, name='att')
            
            bsize, a, b, c = conv34_mp.get_shape().as_list()
            conv34_mp_flatten = tf.reshape(conv34_mp, [bsize, int(np.prod(conv34_mp.get_shape()[1:]))])                    
                
            fc4 = nf.fc_layer(conv34_mp_flatten, model_params['fc4'], name="fc4", reg=l2_reg)
            
            #fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout')
            
            fc5 = nf.fc_layer(fc4, model_params['fc5'], name="fc5", reg=l2_reg)
            
            fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout')
            
            logits = nf.fc_layer(fc5, model_params['fc6'], name="logits", activat_fn=None)
            
            return logits, fc5, att        

    def BCL_att_v2(self, kwargs):
        
        reuse = kwargs["reuse"]
        
        #init = tf.random_normal_initializer(stddev=0.01)
        l2_reg = None#tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 10,

                        }        
       
        with tf.variable_scope("BCL", reuse=reuse):  
            
            conv11 = nf.convolution_layer(self.inputs, model_params['conv11'], [1,1,1,1], name="conv11", reg=l2_reg)
            conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", reg=l2_reg)
            conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", reg=l2_reg)
            conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", reg=l2_reg)
            conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
            
            conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", reg=l2_reg)
            conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", reg=l2_reg)
            conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", reg=l2_reg)            
            conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", reg=l2_reg)           
            conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv34_att, att = nf.channel_attention(conv34_mp, name='att')

            conv_output = conv34_att
            
            bsize, a, b, c = conv_output.get_shape().as_list()
            conv_output_flatten = tf.reshape(conv_output, [bsize, int(np.prod(conv_output.get_shape()[1:]))])                    
                
            fc4 = nf.fc_layer(conv_output_flatten, model_params['fc4'], name="fc4", reg=l2_reg)            
            
            fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout')
            
            fc5 = nf.fc_layer(fc4, model_params['fc5'], name="fc5", reg=l2_reg)            
            
            fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout')
            
            logits = nf.fc_layer(fc5, model_params['fc6'], name="logits", activat_fn=None)
            
            return logits, fc5, att             

    def BCL_att_v3(self, kwargs):
        
        reuse = kwargs["reuse"]
        
        #init = tf.random_normal_initializer(stddev=0.01)
        l2_reg = None#tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 10,

                        }        
       
        with tf.variable_scope("BCL", reuse=reuse):  
            
            conv11 = nf.convolution_layer(self.inputs, model_params['conv11'], [1,1,1,1], name="conv11", reg=l2_reg)
            conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", reg=l2_reg)
            conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", reg=l2_reg)
            conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", reg=l2_reg)
            conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
            
            conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", reg=l2_reg)
            conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", reg=l2_reg)
            conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", reg=l2_reg)            
            conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", reg=l2_reg)           
            #conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv34_att, att = nf.channel_attention(conv34, name='att')

            conv_output = conv34_att + conv33 + conv32 + conv31
            
            bsize, a, b, c = conv_output.get_shape().as_list()
            conv_output_flatten = tf.reshape(conv_output, [bsize, int(np.prod(conv_output.get_shape()[1:]))])                    
                
            fc4 = nf.fc_layer(conv_output_flatten, model_params['fc4'], name="fc4", reg=l2_reg)            
            
            fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout')
            
            fc5 = nf.fc_layer(fc4, model_params['fc5'], name="fc5", reg=l2_reg)            
            
            fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout')
            
            logits = nf.fc_layer(fc5, model_params['fc6'], name="logits", activat_fn=None)
            
            return logits, fc5, att    
        
    def BCL_att_v4(self, kwargs):
        
        reuse = kwargs["reuse"]

        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0)
        l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 10,

                        }        
       
        with tf.variable_scope("BCL", reuse=reuse):  
            
            conv11 = nf.convolution_layer(self.inputs, model_params['conv11'], [1,1,1,1], name="conv11", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
            
            conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)
            conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", initializer=init, reg=l2_reg, is_bn=True, is_training=self.is_training)           
            conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

            conv34_mp, att = nf.channel_attention(conv34_mp, name='att')

            conv34_att = tf.reduce_mean(conv34_mp, axis=[1,2])
            
            bsize, a, b, c = conv34_mp.get_shape().as_list()
            conv34_mp_flatten = tf.reshape(conv34_mp, [bsize, int(np.prod(conv34_mp.get_shape()[1:]))])                    
                
            fc4 = nf.fc_layer(conv34_mp_flatten, model_params['fc4'], initializer=tf.random_uniform_initializer(-1./math.sqrt(256 * 4 * 4), 1./math.sqrt(256 * 4 * 4)), name="fc4", reg=l2_reg)
            
            fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout1')
            
            fc5 = nf.fc_layer(fc4, model_params['fc5'], initializer=tf.random_uniform_initializer(-1./math.sqrt(1024), 1./math.sqrt(1024)), name="fc5", reg=l2_reg)
            
            fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout2')
            
            logits = nf.fc_layer(fc5, model_params['fc6'], initializer=tf.random_uniform_initializer(-1./math.sqrt(1024), 1./math.sqrt(1024)), name="logits", activat_fn=None)
            
            return logits, fc5, att, conv34_att           

    def BCL_att_GAN(self, kwargs):
        
        net = kwargs["net"]

        reuse = kwargs["reuse"]

        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0)
        Gen_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        Dis_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 10,

                        'fc11': 1280,
                        'fc12': 1024,
                        'fc13': 512,
                        'fc14': 512,
                        'fc15': 256,
                        'fc16': 128,
                        'fc17': 64,
                        'fc18': 1,

                        }    
        
        if net is "Gen":
                                   
            with tf.variable_scope("BCL_Gen", reuse=reuse):  

                inputs = kwargs["inputs"]
                
                conv11 = nf.convolution_layer(inputs, model_params['conv11'], [1,1,1,1], name="conv11", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    
                conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
                
                conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)           
                conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    
                conv34_mp, att = nf.channel_attention(conv34_mp, name='att')
                    
                bsize, a, b, c = conv34_mp.get_shape().as_list()
                conv34_mp_flatten = tf.reshape(conv34_mp, [bsize, int(np.prod(conv34_mp.get_shape()[1:]))])                    
                    
                fc4 = nf.fc_layer(conv34_mp_flatten, model_params['fc4'], initializer=tf.random_uniform_initializer(-1./math.sqrt(256 * 4 * 4), 1./math.sqrt(256 * 4 * 4)), name="fc4", reg=Gen_l2_reg)
                
                fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout1')
                
                fc5 = nf.fc_layer(fc4, model_params['fc5'], initializer=tf.random_uniform_initializer(-1./math.sqrt(1024), 1./math.sqrt(1024)), name="fc5", reg=Gen_l2_reg)
                
                fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout2')
                
                logits = nf.fc_layer(fc5, model_params['fc6'], initializer=tf.random_uniform_initializer(-1./math.sqrt(1024), 1./math.sqrt(1024)), name="logits", activat_fn=None)
                
                return logits, fc5, att              
            
        elif net is "Dis":
                        
            with tf.variable_scope("BCL_Dis", reuse=reuse):     
                
                inputs = kwargs["inputs"]

                fc11 = nf.fc_layer(inputs, model_params['fc11'], name="fc11", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc12 = nf.fc_layer(fc11, model_params['fc12'], name="fc12", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc13 = nf.fc_layer(fc12, model_params['fc13'], name="fc13", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc14 = nf.fc_layer(fc13, model_params['fc14'], name="fc14", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc15 = nf.fc_layer(fc14, model_params['fc15'], name="fc15", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc16 = nf.fc_layer(fc15, model_params['fc16'], name="fc16", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc17 = nf.fc_layer(fc16, model_params['fc17'], name="fc17", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                logits = nf.fc_layer(fc17, model_params['fc18'], name="fc18", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                        
                return logits
        
    def AD_att_GAN(self, kwargs):
        
        net = kwargs["net"]

        reuse = kwargs["reuse"]

        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0)
        Gen_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        Dis_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        Cls_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 512,

                        'fc11': 768,
                        'fc12': 768,
                        'fc13': 512,
                        'fc14': 512,
                        'fc15': 256,
                        'fc16': 128,
                        'fc17': 64,
                        'fc18': 32,

                        'fc21': 768,
                        'fc22': 512,
                        'fc23': 256,
                        'fc24': 128,
                        'fc25': 64,
                        'fc26': 10,

                        }    
        
        if net is "Gen":
                                   
            with tf.variable_scope("AD_Gen", reuse=reuse):  

                inputs = kwargs["inputs"]
                
                conv11 = nf.convolution_layer(inputs, model_params['conv11'], [1,1,1,1], name="conv11", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    
                conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
                
                conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)           
                conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    
                conv34_mp, att = nf.channel_attention(conv34_mp, name='att')
                    
                bsize, a, b, c = conv34_mp.get_shape().as_list()
                conv34_mp_flatten = tf.reshape(conv34_mp, [bsize, int(np.prod(conv34_mp.get_shape()[1:]))])                    
                    
                fc4 = nf.fc_layer(conv34_mp_flatten, model_params['fc4'], initializer=init, name="fc4", reg=Gen_l2_reg)
                
                fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout1')
                
                fc5 = nf.fc_layer(fc4, model_params['fc5'], initializer=init, name="fc5", reg=Gen_l2_reg)
                
                fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout2')
                
                code = nf.fc_layer(fc5, model_params['fc6'], initializer=init, name="logits", activat_fn=None)
                
                return code, att              
            
        elif net is "Dis":
                        
            with tf.variable_scope("AD_Dis", reuse=reuse):     
                
                inputs = kwargs["inputs"]

                fc11 = nf.fc_layer(inputs, model_params['fc11'], name="fc11", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc12 = nf.fc_layer(fc11, model_params['fc12'], name="fc12", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc13 = nf.fc_layer(fc12, model_params['fc13'], name="fc13", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc14 = nf.fc_layer(fc13, model_params['fc14'], name="fc14", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc15 = nf.fc_layer(fc14, model_params['fc15'], name="fc15", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc16 = nf.fc_layer(fc15, model_params['fc16'], name="fc16", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc17 = nf.fc_layer(fc16, model_params['fc17'], name="fc17", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                logits = nf.fc_layer(fc17, model_params['fc18'], name="fc18", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                        
                return logits        

        elif net is "Cls":
                        
            with tf.variable_scope("AD_Cls", reuse=reuse):     
                
                inputs = kwargs["inputs"]
                               
                fc21 = nf.fc_layer(inputs, model_params['fc21'], name="fc21", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc22 = nf.fc_layer(fc21, model_params['fc22'], name="fc22", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc23 = nf.fc_layer(fc22, model_params['fc23'], name="fc23", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc24 = nf.fc_layer(fc23, model_params['fc24'], name="fc24", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc25 = nf.fc_layer(fc24, model_params['fc25'], name="fc25", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                logits = nf.fc_layer(fc25, model_params['fc26'], name="fc26", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                        
                return logits      

    def AD_att_GAN_v2(self, kwargs):
        
        net = kwargs["net"]

        reuse = kwargs["reuse"]

        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0)
        Gen_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        Dis_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        Cls_l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        dropout = self.dropout            
        
        model_params = {

                        'conv11': [3,3,64],
                        'conv12': [3,3,64],
                        'conv21': [3,3,128],
                        'conv22': [3,3,256],
                        'conv31': [3,3,256],
                        'conv32': [3,3,256],
                        'conv33': [3,3,256],
                        'conv34': [3,3,256],
                        'fc4': 1024,
                        'fc5': 1024,
                        'fc6': 512,

                        'fc11': 768,
                        'fc12': 768,
                        'fc13': 512,
                        'fc14': 512,
                        'fc15': 256,
                        'fc16': 128,
                        'fc17': 64,
                        'fc18': 2,

                        'fc21': 768,
                        'fc22': 512,
                        'fc23': 256,
                        'fc24': 128,
                        'fc25': 64,
                        'fc26': 10,

                        }    
        
        if net is "Gen":
                                   
            with tf.variable_scope("AD_Gen", reuse=reuse):  

                inputs = kwargs["inputs"]
                
                conv11 = nf.convolution_layer(inputs, model_params['conv11'], [1,1,1,1], name="conv11", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv12 = nf.convolution_layer(conv11, model_params['conv12'], [1,1,1,1], name="conv12", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv12_mp = tf.nn.max_pool(conv12, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    
                conv21 = nf.convolution_layer(conv12_mp, model_params['conv21'], [1,1,1,1], name="conv21", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv22 = nf.convolution_layer(conv21, model_params['conv22'], [1,1,1,1], name="conv22", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv22_mp = tf.nn.max_pool(conv22, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
                
                conv31 = nf.convolution_layer(conv22_mp, model_params['conv31'], [1,1,1,1], name="conv31", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv32 = nf.convolution_layer(conv31, model_params['conv32'], [1,1,1,1], name="conv32", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv33 = nf.convolution_layer(conv32, model_params['conv33'], [1,1,1,1], name="conv33", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)
                conv34 = nf.convolution_layer(conv33, model_params['conv34'], [1,1,1,1], name="conv34", initializer=init, reg=Gen_l2_reg, is_bn=True, is_training=self.is_training)           
                conv34_mp = tf.nn.max_pool(conv34, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    
                conv34_mp, att = nf.channel_attention(conv34_mp, name='att')
                    
                bsize, a, b, c = conv34_mp.get_shape().as_list()
                conv34_mp_flatten = tf.reshape(conv34_mp, [bsize, int(np.prod(conv34_mp.get_shape()[1:]))])                    
                    
                fc4 = nf.fc_layer(conv34_mp_flatten, model_params['fc4'], initializer=init, name="fc4", reg=Gen_l2_reg)
                
                fc4 = tf.layers.dropout(fc4, rate=dropout, training=self.is_training, name='dropout1')
                
                fc5 = nf.fc_layer(fc4, model_params['fc5'], initializer=init, name="fc5", reg=Gen_l2_reg)
                
                fc5 = tf.layers.dropout(fc5, rate=dropout, training=self.is_training, name='dropout2')
                
                code = nf.fc_layer(fc5, model_params['fc6'], initializer=init, name="logits", activat_fn=None)
                
                return code, att              
            
        elif net is "Dis":
                        
            with tf.variable_scope("AD_Dis", reuse=reuse):     
                
                inputs = kwargs["inputs"]

                fc11 = nf.fc_layer(inputs, model_params['fc11'], name="fc11", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc12 = nf.fc_layer(fc11, model_params['fc12'], name="fc12", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc13 = nf.fc_layer(fc12, model_params['fc13'], name="fc13", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc14 = nf.fc_layer(fc13, model_params['fc14'], name="fc14", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc15 = nf.fc_layer(fc14, model_params['fc15'], name="fc15", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc16 = nf.fc_layer(fc15, model_params['fc16'], name="fc16", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                fc17 = nf.fc_layer(fc16, model_params['fc17'], name="fc17", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                
                logits = nf.fc_layer(fc17, model_params['fc18'], name="fc18", activat_fn=nf.lrelu, reg=Dis_l2_reg)
                        
                return logits        

        elif net is "Cls":
                        
            with tf.variable_scope("AD_Cls", reuse=reuse):     
                
                inputs = kwargs["inputs"]
                               
                #fc21 = nf.fc_layer(inputs, model_params['fc21'], name="fc21", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                #fc22 = nf.fc_layer(fc21, model_params['fc22'], name="fc22", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc23 = nf.fc_layer(inputs, model_params['fc23'], name="fc23", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc24 = nf.fc_layer(fc23, model_params['fc24'], name="fc24", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                fc25 = nf.fc_layer(fc24, model_params['fc25'], name="fc25", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                
                logits = nf.fc_layer(fc25, model_params['fc26'], name="fc26", activat_fn=nf.lrelu, reg=Cls_l2_reg)
                        
                return logits      
            
    def AD_att_VAE(self, kwargs):
         
        model_params = {       

            "conv_1": [3,3,256],
            "conv_2": [3,3,128],
            "conv_3": [3,3,64],   
            "conv_4": [3,3,32],   
            "fc_4": 512,
            "fc_5": 256,

            #"fc_mean": 16,
            #"fc_std": 16,

            "fc_mean": 256,
            "fc_std": 256,

            "fc_1": 256,
            "fc_2": 512,                  
            "deconv_2": [3,3,256],
            "deconv_3": [3,3,256],
            "deconv_4": [3,3,128],
            "deconv_5": [3,3,64],
            "deconv_6": [3,3,32],
            "deconv_7": [3,3,3],
            "conv_5": [3,3,128],
            "conv_6": [3,3,64],   
            "conv_7": [3,3,32], 
            "conv_8": [3,3,3], 
            
            "dis_fc_1": 512,
            "dis_fc_2": 256,
            "dis_fc_3": 128,
            "dis_fc_4": 64,
            "dis_fc_5": 1,
            
            "cls_fc_1": 512,
            "cls_fc_2": 256,
            "cls_fc_3": 128,
            "cls_fc_4": 64,
            "cls_fc_5": 10,            
        }

        mode = kwargs["mode"]
        reuse = kwargs["reuse"]

        print("===================================================================")

        if mode is "encoder":                
            with tf.variable_scope("encoder", reuse=reuse):
                
                print("[Encoder] input: %s" % self.inputs.get_shape())
                
                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=tf.nn.relu)
                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=tf.nn.relu)
                conv_1_3 = nf.convolution_layer(conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3", padding='SAME', activat_fn=tf.nn.relu)                
                conv_1, conv_1_att = nf.channel_attention(conv_1_3, name='conv1_att')
                conv_1 = conv_1 + conv_1_1 + conv_1_2 + conv_1_3
                print("conv_1: %s" % conv_1.get_shape())       
                
                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=tf.nn.relu)
                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=tf.nn.relu)
                conv_2_3 = nf.convolution_layer(conv_2_2, model_params["conv_2"], [1,1,1,1], name="conv_2_3", padding='SAME', activat_fn=tf.nn.relu)               
                conv_2, conv_2_att = nf.channel_attention(conv_2_3, name='conv2_att')
                conv_2 = conv_2 + conv_2_1 + conv_2_2 + conv_2_3
                print("conv_2: %s" % conv_2.get_shape())                             
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME', activat_fn=tf.nn.relu)
                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=tf.nn.relu)
                conv_3_3 = nf.convolution_layer(conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME', activat_fn=tf.nn.relu)
                conv_3, conv_3_att = nf.channel_attention(conv_3_3, name='conv3_att')
                conv_3 = conv_3 + conv_3_1 + conv_3_2 + conv_3_3
                print("conv_3: %s" % conv_3.get_shape())                 

                conv_4_1 = nf.convolution_layer(conv_3, model_params["conv_4"], [1,1,1,1], name="conv_4_1", padding='SAME', activat_fn=tf.nn.relu)
                conv_4_2 = nf.convolution_layer(conv_4_1, model_params["conv_4"], [1,1,1,1], name="conv_4_2", padding='SAME', activat_fn=tf.nn.relu)
                conv_4_3 = nf.convolution_layer(conv_4_2, model_params["conv_4"], [1,1,1,1], name="conv_4_3", padding='SAME', activat_fn=tf.nn.relu)
                conv_4, conv_4_att = nf.channel_attention(conv_4_3, name='conv4_att')
                conv_4 = conv_4 + conv_4_1 + conv_4_2 + conv_4_3
                print("conv_4: %s" % conv_4.get_shape())                 
                
                conv_4 = tf.reshape(conv_4, [-1, int(np.prod(conv_4.get_shape()[1:]))])         

                #fc_4 = nf.fc_layer(conv_3, model_params["fc_4"], name="fc_4", activat_fn=tf.nn.relu)
                #print("fc_4: %s" % fc_4.get_shape())    
                
                #fc_5 = nf.fc_layer(fc_4, model_params["fc_5"], name="fc_5", activat_fn=tf.nn.relu)
                #print("fc_5: %s" % fc_5.get_shape())    
                
                en_mean = nf.fc_layer(conv_4, model_params["fc_mean"], name="fc_mean", activat_fn=None)
                print("en_mean: %s" % en_mean.get_shape())    
                
                en_std = nf.fc_layer(conv_4, model_params["fc_std"], name="fc_std", activat_fn=None)
                print("en_std: %s" % en_std.get_shape())    

                att = tf.concat([conv_1_att, conv_2_att, conv_3_att, conv_4_att], -1)
                print("att weight: %s" % att.get_shape())    
                
            return en_mean, en_std, att

        if mode is "decoder": 
            with tf.variable_scope("decoder", reuse=reuse):
                               
                code_layer = kwargs["code"]
                
                print("[Decoder] input: %s" % code_layer.get_shape())
    
                #fc_1 = nf.fc_layer(code_layer, model_params["fc_1"], name="fc_1", activat_fn=nf.lrelu)
                #print("fc_1: %s" % fc_1.get_shape())   
    
                #fc_2 = nf.fc_layer(code_layer, model_params["fc_2"], name="fc_2", activat_fn=nf.lrelu)
                #print("fc_2: %s" % fc_2.get_shape())   
                
                #fc_2 = tf.reshape(fc_2, [tf.shape(self.inputs)[0], 8, 8, 8])
                
                code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 4, 4, 16])

                deconv_2 = nf.deconvolution_layer(code_layer, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,2,2,1], name="deconv_2", padding='SAME', activat_fn=nf.lrelu)
                print("deconv_2: %s" % deconv_2.get_shape())                       
                
                deconv_3 = nf.deconvolution_layer(deconv_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 256], [1,2,2,1], name="deconv_3", padding='SAME', activat_fn=nf.lrelu)
                print("deconv_3: %s" % deconv_3.get_shape())                       

                deconv_4 = nf.deconvolution_layer(deconv_3, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 128], [1,2,2,1], name="deconv_4", padding='SAME', activat_fn=nf.lrelu)
                print("deconv_4: %s" % deconv_4.get_shape())      

                #deconv_5 = nf.deconvolution_layer(deconv_4, model_params["deconv_5"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,1,1,1], name="deconv_5", padding='SAME', activat_fn=nf.lrelu)
                #print("deconv_5: %s" % deconv_5.get_shape())      

                #deconv_6 = nf.deconvolution_layer(deconv_5, model_params["deconv_6"], [tf.shape(self.inputs)[0], 32, 32, 32], [1,1,1,1], name="deconv_6", padding='SAME', activat_fn=nf.lrelu)
                #print("deconv_6: %s" % deconv_6.get_shape())      

                #deconv_7 = nf.deconvolution_layer(deconv_6, model_params["deconv_7"], [tf.shape(self.inputs)[0], 32, 32, 3], [1,1,1,1], name="deconv_7", padding='SAME', activat_fn=None)
                #print("deconv_7: %s" % deconv_7.get_shape())      

                conv_5_1 = nf.convolution_layer(deconv_4, model_params["conv_5"], [1,1,1,1], name="conv_5_1", padding='SAME', activat_fn=nf.lrelu)
                conv_5_2 = nf.convolution_layer(conv_5_1, model_params["conv_5"], [1,1,1,1], name="conv_5_2", padding='SAME', activat_fn=nf.lrelu)
                conv_5_3 = nf.convolution_layer(conv_5_2, model_params["conv_5"], [1,1,1,1], name="conv_5_3", padding='SAME', activat_fn=nf.lrelu)                
                conv_5, conv_5_att = nf.channel_attention(conv_5_3, name='conv5_att')
                conv_5 = conv_5 + conv_5_1 + conv_5_2 + conv_5_3
                print("conv_5: %s" % conv_5.get_shape())       
                
                conv_6_1 = nf.convolution_layer(conv_5, model_params["conv_6"], [1,1,1,1], name="conv_6_1", padding='SAME', activat_fn=nf.lrelu)
                conv_6_2 = nf.convolution_layer(conv_6_1, model_params["conv_6"], [1,1,1,1], name="conv_6_2", padding='SAME', activat_fn=nf.lrelu)
                conv_6_3 = nf.convolution_layer(conv_6_2, model_params["conv_6"], [1,1,1,1], name="conv_6_3", padding='SAME', activat_fn=nf.lrelu)                
                conv_6, conv_6_att = nf.channel_attention(conv_6_3, name='conv6_att')
                conv_6 = conv_6 + conv_6_1 + conv_6_2 + conv_6_3
                print("conv_6: %s" % conv_6.get_shape())       

                conv_7_1 = nf.convolution_layer(conv_6, model_params["conv_7"], [1,1,1,1], name="conv_7_1", padding='SAME', activat_fn=nf.lrelu)
                conv_7_2 = nf.convolution_layer(conv_7_1, model_params["conv_7"], [1,1,1,1], name="conv_7_2", padding='SAME', activat_fn=nf.lrelu)
                conv_7_3 = nf.convolution_layer(conv_7_2, model_params["conv_7"], [1,1,1,1], name="conv_7_3", padding='SAME', activat_fn=nf.lrelu)                
                conv_7, conv_7_att = nf.channel_attention(conv_7_3, name='conv7_att')
                conv_7 = conv_7 + conv_7_1 + conv_7_2 + conv_7_3
                print("conv_7: %s" % conv_7.get_shape())       
                
                conv_8 = nf.convolution_layer(conv_7, model_params["conv_8"], [1,1,1,1], name="conv_8", padding='SAME', activat_fn=None)                
                print("conv_8: %s" % conv_8.get_shape())  
                
                return conv_8    
            
        if mode is "discriminator":
            with tf.variable_scope("discriminator", reuse=reuse):          
                
                dis_input = kwargs["dis_input"]
                
                print("[Discriminator] input: %s" % dis_input.get_shape())
                
                dis_fc_1 = nf.fc_layer(dis_input, model_params["dis_fc_1"], name="dis_fc_1", activat_fn=nf.lrelu)
                print("dis_fc_1: %s" % dis_fc_1.get_shape())                  
                
                dis_fc_2 = nf.fc_layer(dis_fc_1, model_params["dis_fc_2"], name="dis_fc_2", activat_fn=nf.lrelu)
                print("dis_fc_2: %s" % dis_fc_2.get_shape())                  

                dis_fc_3 = nf.fc_layer(dis_fc_2, model_params["dis_fc_3"], name="dis_fc_3", activat_fn=nf.lrelu)
                print("dis_fc_3: %s" % dis_fc_3.get_shape())                  

                dis_fc_4 = nf.fc_layer(dis_fc_3, model_params["dis_fc_4"], name="dis_fc_4", activat_fn=nf.lrelu)
                print("dis_fc_4: %s" % dis_fc_4.get_shape())                  

                dis_fc_5 = nf.fc_layer(dis_fc_4, model_params["dis_fc_5"], name="dis_fc_5", activat_fn=None)
                print("dis_fc_5: %s" % dis_fc_5.get_shape())      

                return dis_fc_5                           

        if mode is "classifier":
            with tf.variable_scope("classifier", reuse=reuse):          
                
                dis_input = kwargs["cls_input"]
                
                print("[Classifier] input: %s" % dis_input.get_shape())
                
                cls_fc_1 = nf.fc_layer(dis_input, model_params["cls_fc_1"], name="cls_fc_1", activat_fn=nf.lrelu)
                print("cls_fc_1: %s" % cls_fc_1.get_shape())                  
                
                cls_fc_2 = nf.fc_layer(cls_fc_1, model_params["cls_fc_2"], name="cls_fc_2", activat_fn=nf.lrelu)
                print("cls_fc_2: %s" % cls_fc_2.get_shape())                  

                cls_fc_3 = nf.fc_layer(cls_fc_2, model_params["cls_fc_3"], name="cls_fc_3", activat_fn=nf.lrelu)
                print("cls_fc_3: %s" % cls_fc_3.get_shape())                  

                cls_fc_4 = nf.fc_layer(cls_fc_3, model_params["cls_fc_4"], name="cls_fc_4", activat_fn=nf.lrelu)
                print("cls_fc_4: %s" % cls_fc_4.get_shape())                  

                cls_fc_5 = nf.fc_layer(cls_fc_4, model_params["cls_fc_5"], name="cls_fc_5", activat_fn=None)
                print("cls_fc_5: %s" % cls_fc_5.get_shape())      

                return cls_fc_5                           

    def AD_att_VAE_WEAK(self, kwargs):
         
        model_params = {       

            "conv_1": [3,3,128],
            "conv_2": [3,3,64],

            #"fc_mean": 16,
            #"fc_std": 16,

            "fc_mean": 1024,
            "fc_std": 1024,

            "fc_1": 256,
            "fc_2": 512,                  
            "deconv_2": [3,3,128],
            "deconv_3": [3,3,64],
            "deconv_4": [3,3,32],
            "conv_5": [3,3,32],
            "conv_6": [3,3,3], 
            
            "dis_fc_1": 512,
            "dis_fc_2": 256,
            "dis_fc_3": 128,
            "dis_fc_4": 64,
            "dis_fc_5": 1,
            
            "cls_fc_1": 512,
            "cls_fc_2": 256,
            "cls_fc_3": 128,
            "cls_fc_4": 64,
            "cls_fc_5": 10,    
            
            "cls_conv_1": [3,3,128],
            "cls_conv_2": [3,3,64],
            "cls_conv_3": [3,3,32],
            
        }

        mode = kwargs["mode"]
        reuse = kwargs["reuse"]

        print("===================================================================")

        if mode is "encoder":                
            with tf.variable_scope("encoder", reuse=reuse):
                
                print("[Encoder] input: %s" % self.inputs.get_shape())
                
                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=tf.nn.relu)
                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=tf.nn.relu)
                conv_1_3 = nf.convolution_layer(conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3", padding='SAME', activat_fn=tf.nn.relu)                
                conv_1, conv_1_att = nf.channel_attention(conv_1_3, name='conv1_att')
                conv_1 = conv_1 + conv_1_1
                print("conv_1: %s" % conv_1.get_shape())       
                                       
                conv_1 = tf.reshape(conv_1, [-1, int(np.prod(conv_1.get_shape()[1:]))])         
                                
                en_mean = nf.fc_layer(conv_1, model_params["fc_mean"], name="fc_mean", activat_fn=None)
                print("en_mean: %s" % en_mean.get_shape())    
                
                en_std = nf.fc_layer(conv_1, model_params["fc_std"], name="fc_std", activat_fn=None)
                print("en_std: %s" % en_std.get_shape())    

                att = conv_1_att
                print("att weight: %s" % att.get_shape())    
                
            return en_mean, en_std, att

        if mode is "decoder": 
            with tf.variable_scope("decoder", reuse=reuse):
                               
                code_layer = kwargs["code"]
                
                print("[Decoder] input: %s" % code_layer.get_shape())
                    
                code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 4, 4, 64])

                deconv_2 = nf.deconvolution_layer(code_layer, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 128], [1,2,2,1], name="deconv_2", padding='SAME', activat_fn=nf.lrelu)
                print("deconv_2: %s" % deconv_2.get_shape())                       
                
                deconv_3 = nf.deconvolution_layer(deconv_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 64], [1,2,2,1], name="deconv_3", padding='SAME', activat_fn=nf.lrelu)
                print("deconv_3: %s" % deconv_3.get_shape())                       

                deconv_4 = nf.deconvolution_layer(deconv_3, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 32], [1,2,2,1], name="deconv_4", padding='SAME', activat_fn=nf.lrelu)
                print("deconv_4: %s" % deconv_4.get_shape())      

                conv_5_1 = nf.convolution_layer(deconv_4, model_params["conv_5"], [1,1,1,1], name="conv_5_1", padding='SAME', activat_fn=nf.lrelu)
                conv_5_2 = nf.convolution_layer(conv_5_1, model_params["conv_5"], [1,1,1,1], name="conv_5_2", padding='SAME', activat_fn=nf.lrelu)
                conv_5_3 = nf.convolution_layer(conv_5_2, model_params["conv_5"], [1,1,1,1], name="conv_5_3", padding='SAME', activat_fn=nf.lrelu)                
                conv_5, conv_5_att = nf.channel_attention(conv_5_3, name='conv5_att')
                conv_5 = conv_5 + conv_5_1
                print("conv_5: %s" % conv_5.get_shape())       
                                
                conv_6 = nf.convolution_layer(deconv_4, model_params["conv_6"], [1,1,1,1], name="conv_6", padding='SAME', activat_fn=None)                
                print("conv_6: %s" % conv_6.get_shape())  
                
                return conv_6    
            
        if mode is "discriminator":
            with tf.variable_scope("discriminator", reuse=reuse):          
                
                dis_input = kwargs["dis_input"]
                
                print("[Discriminator] input: %s" % dis_input.get_shape())
                
                dis_fc_1 = nf.fc_layer(dis_input, model_params["dis_fc_1"], name="dis_fc_1", activat_fn=nf.lrelu)
                print("dis_fc_1: %s" % dis_fc_1.get_shape())                  
                
                dis_fc_2 = nf.fc_layer(dis_fc_1, model_params["dis_fc_2"], name="dis_fc_2", activat_fn=nf.lrelu)
                print("dis_fc_2: %s" % dis_fc_2.get_shape())                  

                dis_fc_3 = nf.fc_layer(dis_fc_2, model_params["dis_fc_3"], name="dis_fc_3", activat_fn=nf.lrelu)
                print("dis_fc_3: %s" % dis_fc_3.get_shape())                  

                dis_fc_4 = nf.fc_layer(dis_fc_3, model_params["dis_fc_4"], name="dis_fc_4", activat_fn=nf.lrelu)
                print("dis_fc_4: %s" % dis_fc_4.get_shape())                  

                dis_fc_5 = nf.fc_layer(dis_fc_4, model_params["dis_fc_5"], name="dis_fc_5", activat_fn=None)
                print("dis_fc_5: %s" % dis_fc_5.get_shape())      

                return dis_fc_5                           

        if mode is "classifier":
            with tf.variable_scope("classifier", reuse=reuse):          
                
                dis_input = kwargs["cls_input"]
                
                print("[Classifier] input: %s" % dis_input.get_shape())
                
                cls_fc_1 = nf.fc_layer(dis_input, model_params["cls_fc_1"], name="cls_fc_1", activat_fn=nf.lrelu)
                print("cls_fc_1: %s" % cls_fc_1.get_shape())                  
                
                cls_fc_2 = nf.fc_layer(cls_fc_1, model_params["cls_fc_2"], name="cls_fc_2", activat_fn=nf.lrelu)
                print("cls_fc_2: %s" % cls_fc_2.get_shape())                  

                cls_fc_3 = nf.fc_layer(cls_fc_2, model_params["cls_fc_3"], name="cls_fc_3", activat_fn=nf.lrelu)
                print("cls_fc_3: %s" % cls_fc_3.get_shape())                  

                cls_fc_4 = nf.fc_layer(cls_fc_3, model_params["cls_fc_4"], name="cls_fc_4", activat_fn=nf.lrelu)
                print("cls_fc_4: %s" % cls_fc_4.get_shape())                  

                cls_fc_5 = nf.fc_layer(cls_fc_4, model_params["cls_fc_5"], name="cls_fc_5", activat_fn=None)
                print("cls_fc_5: %s" % cls_fc_5.get_shape())      

                return cls_fc_5                 
            
        if mode is "reconstruct_cls":
            with tf.variable_scope("reconstruct_cls", reuse=reuse):          
                
                cls_input = kwargs["cls_input"]
                
                print("[Reconstruct_cls] input: %s" % cls_input.get_shape())
                
                cls_conv_1 = nf.convolution_layer(cls_input, model_params["cls_conv_1"], [1,2,2,1], name="cls_conv_1", padding='SAME', activat_fn=tf.nn.relu)
                print("cls_conv_1: %s" % cls_conv_1.get_shape())                  
                
                cls_conv_2 = nf.convolution_layer(cls_conv_1, model_params["cls_conv_2"], [1,2,2,1], name="cls_conv_2", padding='SAME', activat_fn=tf.nn.relu)
                print("cls_conv_2: %s" % cls_conv_2.get_shape())                  
                
                cls_conv_3 = nf.convolution_layer(cls_conv_2, model_params["cls_conv_3"], [1,2,2,1], name="cls_conv_3", padding='SAME', activat_fn=tf.nn.relu)
                print("cls_conv_3: %s" % cls_conv_3.get_shape())                  
                
                cls_conv_3_1 = nf.convolution_layer(cls_conv_3, model_params["cls_conv_3"], [1,1,1,1], name="cls_conv_3_1", padding='SAME', activat_fn=tf.nn.relu)
                cls_conv_3_2 = nf.convolution_layer(cls_conv_3_1, model_params["cls_conv_3"], [1,1,1,1], name="cls_conv_3_2", padding='SAME', activat_fn=tf.nn.relu)
                cls_conv_3_3 = nf.convolution_layer(cls_conv_3_2, model_params["cls_conv_3"], [1,1,1,1], name="cls_conv_3_3", padding='SAME', activat_fn=tf.nn.relu)
                cls_conv_3_3, cls_conv_3_att = nf.channel_attention(cls_conv_3_3, name='cls_conv_3_att')
                cls_conv_output = cls_conv_3_1 + cls_conv_3_3
                
                cls_conv_output = tf.reshape(cls_conv_output, [-1, int(np.prod(cls_conv_output.get_shape()[1:]))])       

                cls_fc_5 = nf.fc_layer(cls_conv_output, model_params["cls_fc_5"], name="cls_fc_5", activat_fn=None)
                print("cls_fc_5: %s" % cls_fc_5.get_shape())  

                return cls_fc_5               

    def AD_att_VAE_GAN(self, kwargs):
         
        model_params = {       

            "conv_1": [3,3,128],
            "conv_2": [3,3,256],
            "conv_3": [3,3,512],

            #"fc_mean": 16,
            #"fc_std": 16,

            "fc_mean": 512,
            "fc_std": 512,

            "fc_1": 256,
            "fc_2": 512,                  
            "deconv_2": [3,3,256],
            "deconv_3": [3,3,128],
            "deconv_4": [3,3,64],
            "conv_5": [3,3,32],
            "conv_6": [3,3,3], 

            "dis_conv_1": [3,3,32],
            "dis_conv_2": [3,3,64],            
            "dis_conv_3": [3,3,128],                  
            "dis_conv_4": [3,3,1],                  

            "dis_fc_1": 2048,
            "dis_fc_2": 2048,
            "dis_fc_3": 1024,
            "dis_fc_4": 512,
            "dis_fc_5": 16,
            
            "cls_fc_1": 512,
            "cls_fc_2": 256,
            "cls_fc_3": 128,
            "cls_fc_4": 64,
            "cls_fc_5": 10,    
            
            "cls_conv_1": [3,3,128],
            "cls_conv_2": [3,3,64],
            "cls_conv_3": [3,3,32],
            
        }

        mode = kwargs["mode"]
        reuse = kwargs["reuse"]

        print("===================================================================")

        if mode is "encoder":                
            with tf.variable_scope("encoder", reuse=reuse):
                
#                print("[Encoder] input: %s" % self.inputs.get_shape())
#                
#                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_1_3 = nf.convolution_layer(conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)                
#                conv_1, conv_1_att = nf.channel_attention(conv_1_3, name='conv1_att')
#                conv_1 = conv_1 + conv_1_1
#                print("conv_1: %s" % conv_1.get_shape())       
#
#                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_2_3 = nf.convolution_layer(conv_2_2, model_params["conv_2"], [1,1,1,1], name="conv_2_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)                
#                conv_2, conv_2_att = nf.channel_attention(conv_2_3, name='conv2_att')
#                conv_2 = conv_2 + conv_2_1
#                print("conv_2: %s" % conv_2.get_shape())       
#                
#                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_3_3 = nf.convolution_layer(conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)                
#                conv_3, conv_3_att = nf.channel_attention(conv_3_3, name='conv3_att')
#                conv_3 = conv_3 + conv_3_1
#                print("conv_3: %s" % conv_3.get_shape())                       
#                                       
#                conv_3 = tf.reshape(conv_3, [-1, int(np.prod(conv_3.get_shape()[1:]))])         
#                                
#                en_mean = nf.fc_layer(conv_3, model_params["fc_mean"], name="fc_mean", activat_fn=None)
#                print("en_mean: %s" % en_mean.get_shape())    
#                
#                en_std = nf.fc_layer(conv_3, model_params["fc_std"], name="fc_std", activat_fn=None)
#                print("en_std: %s" % en_std.get_shape())    
#
#                att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)
#                print("att weight: %s" % att.get_shape())    

                print("[Encoder] input: %s" % self.inputs.get_shape())
                
                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)               
                conv_1, conv_1_att = nf.channel_attention(conv_1_1, name='conv1_att')
                print("conv_1: %s" % conv_1.get_shape())       

                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_2, conv_2_att = nf.channel_attention(conv_2_1, name='conv2_att')
                print("conv_2: %s" % conv_2.get_shape())       
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_3, conv_3_att = nf.channel_attention(conv_3_1, name='conv3_att')
                print("conv_3: %s" % conv_3.get_shape())                       
                                       
                conv_3 = tf.reshape(conv_3, [-1, int(np.prod(conv_3.get_shape()[1:]))])         
                                
                en_mean = nf.fc_layer(conv_3, model_params["fc_mean"], name="fc_mean", activat_fn=None)
                print("en_mean: %s" % en_mean.get_shape())    
                
                en_std = nf.fc_layer(conv_3, model_params["fc_std"], name="fc_std", activat_fn=None)
                print("en_std: %s" % en_std.get_shape())    

                att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)
                print("att weight: %s" % att.get_shape())    
                
            return en_mean, en_std, att

        if mode is "decoder": 
            with tf.variable_scope("decoder", reuse=reuse):
                               
                code_layer = kwargs["code"]
                
#                print("[Decoder] input: %s" % code_layer.get_shape())
#                    
#                code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 4, 4, 32])
#
#                deconv_2_1 = nf.deconvolution_layer(code_layer, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,2,2,1], name="deconv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_2_2 = nf.deconvolution_layer(deconv_2_1, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,1,1,1], name="deconv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_2_3 = nf.deconvolution_layer(deconv_2_2, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,1,1,1], name="deconv_2_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_2, deconv_2_att = nf.channel_attention(deconv_2_3, name='deconv2_att')
#                deconv_2 = deconv_2 + deconv_2_1                
#                print("deconv_2: %s" % deconv_2.get_shape())                       
#                
#                deconv_3_1 = nf.deconvolution_layer(deconv_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,2,2,1], name="deconv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_3_2 = nf.deconvolution_layer(deconv_3_1, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,1,1,1], name="deconv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_3_3 = nf.deconvolution_layer(deconv_3_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,1,1,1], name="deconv_3_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_3, deconv_3_att = nf.channel_attention(deconv_3_3, name='deconv3_att')
#                deconv_3 = deconv_3 + deconv_3_1                
#                print("deconv_3: %s" % deconv_3.get_shape())                       
#
#                deconv_4_1 = nf.deconvolution_layer(deconv_3, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,2,2,1], name="deconv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_4_2 = nf.deconvolution_layer(deconv_4_1, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,1,1,1], name="deconv_4_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_4_3 = nf.deconvolution_layer(deconv_4_2, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,1,1,1], name="deconv_4_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                deconv_4, deconv_4_att = nf.channel_attention(deconv_4_3, name='deconv4_att')
#                deconv_4 = deconv_4 + deconv_4_1                
#                print("deconv_4: %s" % deconv_4.get_shape())          
#                                
#                conv_6 = nf.convolution_layer(deconv_4, model_params["conv_6"], [1,1,1,1], name="conv_6", padding='SAME', activat_fn=None)                
#                print("conv_6: %s" % conv_6.get_shape())  

                print("[Decoder] input: %s" % code_layer.get_shape())
                    
                code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 4, 4, 32])

                deconv_2_1 = nf.deconvolution_layer(code_layer, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,2,2,1], name="deconv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_2, deconv_2_att = nf.channel_attention(deconv_2_1, name='deconv2_att')
                print("deconv_2: %s" % deconv_2.get_shape())                       
                
                deconv_3_1 = nf.deconvolution_layer(deconv_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,2,2,1], name="deconv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_3, deconv_3_att = nf.channel_attention(deconv_3_1, name='deconv3_att')
                print("deconv_3: %s" % deconv_3.get_shape())                       

                deconv_4_1 = nf.deconvolution_layer(deconv_3, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,2,2,1], name="deconv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_4, deconv_4_att = nf.channel_attention(deconv_4_1, name='deconv4_att')
                print("deconv_4: %s" % deconv_4.get_shape())          
                                
                conv_6 = nf.convolution_layer(deconv_4, model_params["conv_6"], [1,1,1,1], name="conv_6", padding='SAME', activat_fn=None)                
                print("conv_6: %s" % conv_6.get_shape())  
                
                return conv_6    

        if mode is "encoder2":                
            with tf.variable_scope("encoder2", reuse=reuse):
                
#                print("[Encoder-2] input: %s" % self.inputs.get_shape())
#                
#                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_1_3 = nf.convolution_layer(conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)                
#                conv_1, conv_1_att = nf.channel_attention(conv_1_3, name='conv1_att_2')
#                conv_1 = conv_1 + conv_1_1
#                print("conv_1: %s" % conv_1.get_shape())       
#
#                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_2_3 = nf.convolution_layer(conv_2_2, model_params["conv_2"], [1,1,1,1], name="conv_2_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)                
#                conv_2, conv_2_att = nf.channel_attention(conv_2_3, name='conv2_att_2')
#                conv_2 = conv_2 + conv_2_1
#                print("conv_2: %s" % conv_2.get_shape())       
#                
#                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
#                conv_3_3 = nf.convolution_layer(conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)                
#                conv_3, conv_3_att = nf.channel_attention(conv_3_3, name='conv3_att_2')
#                conv_3 = conv_3 + conv_3_1
#                print("conv_3: %s" % conv_3.get_shape())                       
#                                       
#                conv_3 = tf.reshape(conv_3, [-1, int(np.prod(conv_3.get_shape()[1:]))])         
#                                
#                en_mean = nf.fc_layer(conv_3, model_params["fc_mean"], name="fc_mean_2", activat_fn=None)
#                print("en_mean: %s" % en_mean.get_shape())    
#                
#                en_std = nf.fc_layer(conv_3, model_params["fc_std"], name="fc_std_2", activat_fn=None)
#                print("en_std: %s" % en_std.get_shape())    
#
#                att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)
#                print("att weight: %s" % att.get_shape())    

                print("[Encoder-2] input: %s" % self.inputs.get_shape())
                
                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)               
                conv_1, conv_1_att = nf.channel_attention(conv_1_1, name='conv1_att')
                print("conv_1: %s" % conv_1.get_shape())       

                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_2, conv_2_att = nf.channel_attention(conv_2_1, name='conv2_att')
                print("conv_2: %s" % conv_2.get_shape())       
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_3, conv_3_att = nf.channel_attention(conv_3_1, name='conv3_att')
                print("conv_3: %s" % conv_3.get_shape())                       
                                       
                conv_3 = tf.reshape(conv_3, [-1, int(np.prod(conv_3.get_shape()[1:]))])         
                                
                en_mean = nf.fc_layer(conv_3, model_params["fc_mean"], name="fc_mean_2", activat_fn=None)
                print("en_mean: %s" % en_mean.get_shape())    
                
                en_std = nf.fc_layer(conv_3, model_params["fc_std"], name="fc_std_2", activat_fn=None)
                print("en_std: %s" % en_std.get_shape())    

                att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)
                print("att weight: %s" % att.get_shape())    
                
            return en_mean, en_std, att
        
        if mode is "discriminator":
            with tf.variable_scope("discriminator", reuse=reuse):          
                
                dis_input = kwargs["dis_input"]
                
                print("[Discriminator] input: %s" % dis_input.get_shape())
                
                dis_conv_1_1 = nf.convolution_layer(dis_input, model_params["dis_conv_1"], [1,2,2,1], name="dis_conv_1_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_1_2 = nf.convolution_layer(dis_conv_1_1, model_params["dis_conv_1"], [1,1,1,1], name="dis_conv_1_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_1_3 = nf.convolution_layer(dis_conv_1_2, model_params["dis_conv_1"], [1,1,1,1], name="dis_conv_1_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_1, dis_conv_1_att = nf.channel_attention(dis_conv_1_3, name='dis_conv1_att')
                dis_conv_1 = dis_conv_1 + dis_conv_1_1
                print("dis_conv_1: %s" % dis_conv_1.get_shape())     

                dis_conv_2_1 = nf.convolution_layer(dis_conv_1, model_params["dis_conv_2"], [1,2,2,1], name="dis_conv_2_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_2_2 = nf.convolution_layer(dis_conv_2_1, model_params["dis_conv_2"], [1,1,1,1], name="dis_conv_2_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_2_3 = nf.convolution_layer(dis_conv_2_2, model_params["dis_conv_2"], [1,1,1,1], name="dis_conv_2_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_2, dis_conv_2_att = nf.channel_attention(dis_conv_2_3, name='dis_conv2_att')
                dis_conv_2 = dis_conv_2 + dis_conv_2_1
                print("dis_conv_2: %s" % dis_conv_2.get_shape())     

                dis_conv_3_1 = nf.convolution_layer(dis_conv_2, model_params["dis_conv_3"], [1,2,2,1], name="dis_conv_3_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_3_2 = nf.convolution_layer(dis_conv_3_1, model_params["dis_conv_3"], [1,1,1,1], name="dis_conv_3_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_3_3 = nf.convolution_layer(dis_conv_3_2, model_params["dis_conv_3"], [1,1,1,1], name="dis_conv_3_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_3, dis_conv_3_att = nf.channel_attention(dis_conv_3_3, name='dis_conv3_att')
                dis_conv_3 = dis_conv_3 + dis_conv_3_1
                print("dis_conv_3: %s" % dis_conv_3.get_shape())     
                
                dis_conv_4 = nf.convolution_layer(dis_conv_3, model_params["dis_conv_4"], [1,1,1,1], name="dis_conv_4", padding='SAME', activat_fn=nf.lrelu)
                print("dis_conv_4: %s" % dis_conv_4.get_shape())     
                
                return dis_conv_4   

        if mode is "code_dis":
            with tf.variable_scope("code_dis", reuse=reuse):          
                
                code_dis_input = kwargs["code_dis_input"]
                
                print("[Code_Discriminator] input: %s" % code_dis_input.get_shape())
                
                dis_fc_1_1 = nf.fc_layer(code_dis_input, model_params["dis_fc_1"], name="dis_fc_1_1", activat_fn=nf.lrelu)
                dis_fc_1_2 = nf.fc_layer(dis_fc_1_1, model_params["dis_fc_1"], name="dis_fc_1_2", activat_fn=nf.lrelu)
                dis_fc_1_3 = nf.fc_layer(dis_fc_1_2, model_params["dis_fc_1"], name="dis_fc_1_3", activat_fn=nf.lrelu)
                dis_fc_1 = dis_fc_1_1 + dis_fc_1_3
                print("dis_fc_1: %s" % dis_fc_1.get_shape())                  
                
                dis_fc_2_1 = nf.fc_layer(dis_fc_1, model_params["dis_fc_2"], name="dis_fc_2_1", activat_fn=nf.lrelu)
                dis_fc_2_2 = nf.fc_layer(dis_fc_2_1, model_params["dis_fc_2"], name="dis_fc_2_2", activat_fn=nf.lrelu)
                dis_fc_2_3 = nf.fc_layer(dis_fc_2_2, model_params["dis_fc_2"], name="dis_fc_2_3", activat_fn=nf.lrelu)
                dis_fc_2 = dis_fc_2_1 + dis_fc_2_3
                print("dis_fc_2: %s" % dis_fc_2.get_shape())                  

                dis_fc_3_1 = nf.fc_layer(dis_fc_2, model_params["dis_fc_3"], name="dis_fc_3_1", activat_fn=nf.lrelu)
                dis_fc_3_2 = nf.fc_layer(dis_fc_3_1, model_params["dis_fc_3"], name="dis_fc_3_2", activat_fn=nf.lrelu)
                dis_fc_3_3 = nf.fc_layer(dis_fc_3_2, model_params["dis_fc_3"], name="dis_fc_3_3", activat_fn=nf.lrelu)
                dis_fc_3 = dis_fc_3_1 + dis_fc_3_3
                print("dis_fc_3: %s" % dis_fc_3.get_shape())                  

                dis_fc_4_1 = nf.fc_layer(dis_fc_3, model_params["dis_fc_4"], name="dis_fc_4_1", activat_fn=nf.lrelu)
                dis_fc_4_2 = nf.fc_layer(dis_fc_4_1, model_params["dis_fc_4"], name="dis_fc_4_2", activat_fn=nf.lrelu)
                dis_fc_4_3 = nf.fc_layer(dis_fc_4_2, model_params["dis_fc_4"], name="dis_fc_4_3", activat_fn=nf.lrelu)
                dis_fc_4 = dis_fc_4_1 + dis_fc_4_3
                print("dis_fc_4: %s" % dis_fc_4.get_shape())                  

                dis_fc_5 = nf.fc_layer(dis_fc_4, model_params["dis_fc_5"], name="dis_fc_5", activat_fn=nf.lrelu)
                print("dis_fc_5: %s" % dis_fc_5.get_shape())                  
                
                return dis_fc_5                    
        
    def AD_att_AE_GAN(self, kwargs):
         
        model_params = {       

            "conv_1": [3,3,128],
            "conv_2": [3,3,256],
            "conv_3": [3,3,512],

            "fc_code": 512,

            "fc_1": 256,
            "fc_2": 512,                  
            "deconv_2": [3,3,256],
            "deconv_3": [3,3,128],
            "deconv_4": [3,3,64],
            "conv_5": [3,3,32],
            "conv_6": [3,3,3], 

            "dis_conv_1": [3,3,32],
            "dis_conv_2": [3,3,64],            
            "dis_conv_3": [3,3,128],                  
            "dis_conv_4": [3,3,1],                  

            "dis_fc_1": 2048,
            "dis_fc_2": 2048,
            "dis_fc_3": 1024,
            "dis_fc_4": 512,
            "dis_fc_5": 16,
            
            "cls_fc_1": 2048,
            "cls_fc_2": 2048,
            "cls_fc_3": 1024,
            "cls_fc_4": 512,
            "cls_fc_5": 10,    
            
            "cls_conv_1": [3,3,128],
            "cls_conv_2": [3,3,64],
            "cls_conv_3": [3,3,32],
            
        }

        mode = kwargs["mode"]
        reuse = kwargs["reuse"]

        print("===================================================================")

        if mode is "encoder":                
            with tf.variable_scope("encoder", reuse=reuse):

                print("[Encoder] input: %s" % self.inputs.get_shape())
                
                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu)               
                #conv_1, conv_1_att = nf.channel_attention(conv_1_1, name='conv1_att')
                conv_1 = conv_1_1
                print("conv_1: %s" % conv_1.get_shape())       

                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_2, conv_2_att = nf.channel_attention(conv_2_1, name='conv2_att')
                conv_2 = conv_2_1
                print("conv_2: %s" % conv_2.get_shape())       
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_3, conv_3_att = nf.channel_attention(conv_3_1, name='conv3_att')
                conv_3 = conv_3_1
                print("conv_3: %s" % conv_3.get_shape())                       
                                       
                conv_3 = tf.reshape(conv_3, [-1, int(np.prod(conv_3.get_shape()[1:]))])         
                                
                en_code = nf.fc_layer(conv_3, model_params["fc_code"], name="fc_code", activat_fn=None)
                print("en_code: %s" % en_code.get_shape())    
                
                #att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)                
                #print("att weight: %s" % att.get_shape())    
                att = None                
                
            return en_code, att

        if mode is "decoder": 
            with tf.variable_scope("decoder", reuse=reuse):
                               
                code_layer = kwargs["code"]

                print("[Decoder] input: %s" % code_layer.get_shape())
                    
                code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 4, 4, 32])

                deconv_2_1 = nf.deconvolution_layer(code_layer, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,2,2,1], name="deconv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #deconv_2, deconv_2_att = nf.channel_attention(deconv_2_1, name='deconv2_att')
                deconv_2 = deconv_2_1
                print("deconv_2: %s" % deconv_2.get_shape())                       
                
                deconv_3_1 = nf.deconvolution_layer(deconv_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,2,2,1], name="deconv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #deconv_3, deconv_3_att = nf.channel_attention(deconv_3_1, name='deconv3_att')
                deconv_3 = deconv_3_1
                print("deconv_3: %s" % deconv_3.get_shape())                       

                deconv_4_1 = nf.deconvolution_layer(deconv_3, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,2,2,1], name="deconv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #deconv_4, deconv_4_att = nf.channel_attention(deconv_4_1, name='deconv4_att')
                deconv_4 = deconv_4_1
                print("deconv_4: %s" % deconv_4.get_shape())          
                                
                conv_6 = nf.convolution_layer(deconv_4, model_params["conv_6"], [1,1,1,1], name="conv_6", padding='SAME', activat_fn=None)                
                print("conv_6: %s" % conv_6.get_shape())  
                
                return conv_6    

        if mode is "encoder2":                
            with tf.variable_scope("encoder2", reuse=reuse):

                print("[Encoder-2] input: %s" % self.inputs.get_shape())
                
                conv_1_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1_1_2", padding='SAME')               
                #conv_1, conv_1_att = nf.channel_attention(conv_1_1, name='conv1_att')
                conv_1 = conv_1_1
                print("conv_1: %s" % conv_1.get_shape())       

                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_2, conv_2_att = nf.channel_attention(conv_2_1, name='conv2_att')
                conv_2 = conv_2_1
                print("conv_2: %s" % conv_2.get_shape())       
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_3, conv_3_att = nf.channel_attention(conv_3_1, name='conv3_att')
                conv_3 = conv_3_1
                print("conv_3: %s" % conv_3.get_shape())                       
                                       
                conv_3 = tf.reshape(conv_3, [-1, int(np.prod(conv_3.get_shape()[1:]))])         
                                
                en_code = nf.fc_layer(conv_3, model_params["fc_code"], name="fc_code_2", activat_fn=None)
                print("en_code: %s" % en_code.get_shape())    

                #att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)
                #print("att weight: %s" % att.get_shape())    
                att = None
                
            return en_code, att
        
        if mode is "discriminator":
            with tf.variable_scope("discriminator", reuse=reuse):          
                
                dis_input = kwargs["dis_input"]
                
                print("[Discriminator] input: %s" % dis_input.get_shape())
                
                dis_conv_1_1 = nf.convolution_layer(dis_input, model_params["dis_conv_1"], [1,2,2,1], name="dis_conv_1_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_1_2 = nf.convolution_layer(dis_conv_1_1, model_params["dis_conv_1"], [1,1,1,1], name="dis_conv_1_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_1_3 = nf.convolution_layer(dis_conv_1_2, model_params["dis_conv_1"], [1,1,1,1], name="dis_conv_1_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_1, dis_conv_1_att = nf.channel_attention(dis_conv_1_3, name='dis_conv1_att')
                dis_conv_1 = dis_conv_1 + dis_conv_1_1
                print("dis_conv_1: %s" % dis_conv_1.get_shape())     

                dis_conv_2_1 = nf.convolution_layer(dis_conv_1, model_params["dis_conv_2"], [1,2,2,1], name="dis_conv_2_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_2_2 = nf.convolution_layer(dis_conv_2_1, model_params["dis_conv_2"], [1,1,1,1], name="dis_conv_2_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_2_3 = nf.convolution_layer(dis_conv_2_2, model_params["dis_conv_2"], [1,1,1,1], name="dis_conv_2_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_2, dis_conv_2_att = nf.channel_attention(dis_conv_2_3, name='dis_conv2_att')
                dis_conv_2 = dis_conv_2 + dis_conv_2_1
                print("dis_conv_2: %s" % dis_conv_2.get_shape())     

                dis_conv_3_1 = nf.convolution_layer(dis_conv_2, model_params["dis_conv_3"], [1,2,2,1], name="dis_conv_3_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_3_2 = nf.convolution_layer(dis_conv_3_1, model_params["dis_conv_3"], [1,1,1,1], name="dis_conv_3_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_3_3 = nf.convolution_layer(dis_conv_3_2, model_params["dis_conv_3"], [1,1,1,1], name="dis_conv_3_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_3, dis_conv_3_att = nf.channel_attention(dis_conv_3_3, name='dis_conv3_att')
                dis_conv_3 = dis_conv_3 + dis_conv_3_1
                print("dis_conv_3: %s" % dis_conv_3.get_shape())     
                
                dis_conv_4 = nf.convolution_layer(dis_conv_3, model_params["dis_conv_4"], [1,1,1,1], name="dis_conv_4", padding='SAME', activat_fn=nf.lrelu)
                print("dis_conv_4: %s" % dis_conv_4.get_shape())     
                
                return dis_conv_4   

        if mode is "code_dis":
            with tf.variable_scope("code_dis", reuse=reuse):          
                
                code_dis_input = kwargs["code_dis_input"]
                
                print("[Code_Discriminator] input: %s" % code_dis_input.get_shape())
                
                dis_fc_1_1 = nf.fc_layer(code_dis_input, model_params["dis_fc_1"], name="dis_fc_1_1", activat_fn=nf.lrelu)
                dis_fc_1_2 = nf.fc_layer(dis_fc_1_1, model_params["dis_fc_1"], name="dis_fc_1_2", activat_fn=nf.lrelu)
                dis_fc_1_3 = nf.fc_layer(dis_fc_1_2, model_params["dis_fc_1"], name="dis_fc_1_3", activat_fn=nf.lrelu)
                dis_fc_1 = dis_fc_1_1 + dis_fc_1_3
                print("dis_fc_1: %s" % dis_fc_1.get_shape())                  
                
                dis_fc_2_1 = nf.fc_layer(dis_fc_1, model_params["dis_fc_2"], name="dis_fc_2_1", activat_fn=nf.lrelu)
                dis_fc_2_2 = nf.fc_layer(dis_fc_2_1, model_params["dis_fc_2"], name="dis_fc_2_2", activat_fn=nf.lrelu)
                dis_fc_2_3 = nf.fc_layer(dis_fc_2_2, model_params["dis_fc_2"], name="dis_fc_2_3", activat_fn=nf.lrelu)
                dis_fc_2 = dis_fc_2_1 + dis_fc_2_3
                print("dis_fc_2: %s" % dis_fc_2.get_shape())                  

                dis_fc_3_1 = nf.fc_layer(dis_fc_2, model_params["dis_fc_3"], name="dis_fc_3_1", activat_fn=nf.lrelu)
                dis_fc_3_2 = nf.fc_layer(dis_fc_3_1, model_params["dis_fc_3"], name="dis_fc_3_2", activat_fn=nf.lrelu)
                dis_fc_3_3 = nf.fc_layer(dis_fc_3_2, model_params["dis_fc_3"], name="dis_fc_3_3", activat_fn=nf.lrelu)
                dis_fc_3 = dis_fc_3_1 + dis_fc_3_3
                print("dis_fc_3: %s" % dis_fc_3.get_shape())                  

                dis_fc_4_1 = nf.fc_layer(dis_fc_3, model_params["dis_fc_4"], name="dis_fc_4_1", activat_fn=nf.lrelu)
                dis_fc_4_2 = nf.fc_layer(dis_fc_4_1, model_params["dis_fc_4"], name="dis_fc_4_2", activat_fn=nf.lrelu)
                dis_fc_4_3 = nf.fc_layer(dis_fc_4_2, model_params["dis_fc_4"], name="dis_fc_4_3", activat_fn=nf.lrelu)
                dis_fc_4 = dis_fc_4_1 + dis_fc_4_3
                print("dis_fc_4: %s" % dis_fc_4.get_shape())                  

                dis_fc_5 = nf.fc_layer(dis_fc_4, model_params["dis_fc_5"], name="dis_fc_5", activat_fn=nf.lrelu)
                print("dis_fc_5: %s" % dis_fc_5.get_shape())                  
                
                return dis_fc_5                            

        if mode is "code_cls":
            with tf.variable_scope("code_cls", reuse=reuse):          
                
                code_cls_input = kwargs["code_cls_input"]
                
                print("[Code_Classifier] input: %s" % code_cls_input.get_shape())
                
                cls_fc_1_1 = nf.fc_layer(code_cls_input, model_params["cls_fc_1"], name="cls_fc_1_1", activat_fn=nf.lrelu)
                cls_fc_1_2 = nf.fc_layer(cls_fc_1_1, model_params["cls_fc_1"], name="cls_fc_1_2", activat_fn=nf.lrelu)
                cls_fc_1_3 = nf.fc_layer(cls_fc_1_2, model_params["cls_fc_1"], name="cls_fc_1_3", activat_fn=nf.lrelu)
                cls_fc_1 = cls_fc_1_1 + cls_fc_1_3
                print("cls_fc_1: %s" % cls_fc_1.get_shape())                  
                
                cls_fc_2_1 = nf.fc_layer(cls_fc_1, model_params["cls_fc_2"], name="cls_fc_2_1", activat_fn=nf.lrelu)
                cls_fc_2_2 = nf.fc_layer(cls_fc_2_1, model_params["cls_fc_2"], name="cls_fc_2_2", activat_fn=nf.lrelu)
                cls_fc_2_3 = nf.fc_layer(cls_fc_2_2, model_params["cls_fc_2"], name="cls_fc_2_3", activat_fn=nf.lrelu)
                cls_fc_2 = cls_fc_2_1 + cls_fc_2_3
                print("cls_fc_2: %s" % cls_fc_2.get_shape())                  

                cls_fc_3_1 = nf.fc_layer(cls_fc_2, model_params["cls_fc_3"], name="cls_fc_3_1", activat_fn=nf.lrelu)
                cls_fc_3_2 = nf.fc_layer(cls_fc_3_1, model_params["cls_fc_3"], name="cls_fc_3_2", activat_fn=nf.lrelu)
                cls_fc_3_3 = nf.fc_layer(cls_fc_3_2, model_params["cls_fc_3"], name="cls_fc_3_3", activat_fn=nf.lrelu)
                cls_fc_3 = cls_fc_3_1 + cls_fc_3_3
                print("cls_fc_3: %s" % cls_fc_3.get_shape())                  

                cls_fc_4_1 = nf.fc_layer(cls_fc_3, model_params["cls_fc_4"], name="cls_fc_4_1", activat_fn=nf.lrelu)
                cls_fc_4_2 = nf.fc_layer(cls_fc_4_1, model_params["cls_fc_4"], name="cls_fc_4_2", activat_fn=nf.lrelu)
                cls_fc_4_3 = nf.fc_layer(cls_fc_4_2, model_params["cls_fc_4"], name="cls_fc_4_3", activat_fn=nf.lrelu)
                cls_fc_4 = cls_fc_4_1 + cls_fc_4_3
                print("cls_fc_4: %s" % cls_fc_4.get_shape())                  

                cls_fc_5 = nf.fc_layer(cls_fc_4, model_params["cls_fc_5"], name="cls_fc_5", activat_fn=nf.lrelu)
                print("cls_fc_5: %s" % cls_fc_5.get_shape())                  
                
                return cls_fc_5   
        
    def AD_att_AE_GAN_3DCode(self, kwargs):
         
        model_params = {       

            "conv_1": [3,3,128],
            "conv_2": [3,3,256],
            "conv_3": [3,3,512],

            "conv_code": [1,1,32],

            "fc_1": 256,
            "fc_2": 512,                  
            "deconv_2": [3,3,256],
            "deconv_3": [3,3,128],
            "deconv_4": [3,3,64],
            "conv_5": [3,3,32],
            "conv_6": [3,3,3], 

            "dis_conv_1": [3,3,32],
            "dis_conv_2": [3,3,64],            
            "dis_conv_3": [3,3,128],                  
            "dis_conv_4": [3,3,1],                  

            "dis_fc_1": 2048,
            "dis_fc_2": 2048,
            "dis_fc_3": 1024,
            "dis_fc_4": 512,
            "dis_fc_5": 16,
            
            "cls_fc_1": 2048,
            "cls_fc_2": 2048,
            "cls_fc_3": 1024,
            "cls_fc_4": 512,
            "cls_fc_5": 10,    
            
            "cls_conv_1": [3,3,128],
            "cls_conv_2": [3,3,64],
            "cls_conv_3": [3,3,32],
            
        }

        mode = kwargs["mode"]
        reuse = kwargs["reuse"]

        print("===================================================================")

        if mode is "encoder":                
            with tf.variable_scope("encoder", reuse=reuse):

                en_input = kwargs["en_input"]
                
                print("[Encoder] input: %s" % en_input.get_shape())
                
                conv_1_1 = nf.convolution_layer(en_input, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)               
                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)     
                conv_1_3 = nf.convolution_layer(conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)     
                #conv_1, conv_1_att = nf.channel_attention(conv_1_1, name='conv1_att')
                conv_1 = conv_1_1 + conv_1_3
                print("conv_1: %s" % conv_1.get_shape())       

                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_2_3 = nf.convolution_layer(conv_2_2, model_params["conv_2"], [1,1,1,1], name="conv_2_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_2, conv_2_att = nf.channel_attention(conv_2_1, name='conv2_att')
                conv_2 = conv_2_1 + conv_2_3
                print("conv_2: %s" % conv_2.get_shape())       
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_3_3 = nf.convolution_layer(conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_3, conv_3_att = nf.channel_attention(conv_3_1, name='conv3_att')
                conv_3 = conv_3_1 + conv_3_3
                print("conv_3: %s" % conv_3.get_shape())                       
                                      
                en_code = nf.convolution_layer(conv_3, model_params["conv_code"], [1,1,1,1], name="conv_code", padding='VALID', activat_fn=nf.lrelu)
                print("en_code: %s" % en_code.get_shape())                       
                
                #att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)                
                #print("att weight: %s" % att.get_shape())    
                att = None                
                
            return en_code, att

        if mode is "decoder": 
            with tf.variable_scope("decoder", reuse=reuse):
                               
                code_layer = kwargs["code"]

                print("[Decoder] input: %s" % code_layer.get_shape())
                    
                code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 4, 4, 32])

                deconv_2_1 = nf.deconvolution_layer(code_layer, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,2,2,1], name="deconv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_2_2 = nf.deconvolution_layer(deconv_2_1, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,1,1,1], name="deconv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_2_3 = nf.deconvolution_layer(deconv_2_2, model_params["deconv_2"], [tf.shape(self.inputs)[0], 8, 8, 256], [1,1,1,1], name="deconv_2_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #deconv_2, deconv_2_att = nf.channel_attention(deconv_2_1, name='deconv2_att')
                deconv_2 = deconv_2_1 + deconv_2_3
                print("deconv_2: %s" % deconv_2.get_shape())                       
                
                deconv_3_1 = nf.deconvolution_layer(deconv_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,2,2,1], name="deconv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_3_2 = nf.deconvolution_layer(deconv_3_1, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,1,1,1], name="deconv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_3_3 = nf.deconvolution_layer(deconv_3_2, model_params["deconv_3"], [tf.shape(self.inputs)[0], 16, 16, 128], [1,1,1,1], name="deconv_3_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #deconv_3, deconv_3_att = nf.channel_attention(deconv_3_1, name='deconv3_att')
                deconv_3 = deconv_3_1 + deconv_3_3
                print("deconv_3: %s" % deconv_3.get_shape())                       

                deconv_4_1 = nf.deconvolution_layer(deconv_3, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,2,2,1], name="deconv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_4_2 = nf.deconvolution_layer(deconv_4_1, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,1,1,1], name="deconv_4_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                deconv_4_3 = nf.deconvolution_layer(deconv_4_2, model_params["deconv_4"], [tf.shape(self.inputs)[0], 32, 32, 64], [1,1,1,1], name="deconv_4_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #deconv_4, deconv_4_att = nf.channel_attention(deconv_4_1, name='deconv4_att')
                deconv_4 = deconv_4_1 + deconv_4_3
                print("deconv_4: %s" % deconv_4.get_shape())          
                                
                conv_6 = nf.convolution_layer(deconv_4, model_params["conv_6"], [1,1,1,1], name="conv_6", padding='SAME', activat_fn=None)                
                print("conv_6: %s" % conv_6.get_shape())  
                
                return conv_6    

        if mode is "encoder2":                
            with tf.variable_scope("encoder2", reuse=reuse):

                en_input = kwargs["en_input"]
                
                print("[Encoder-2] input: %s" % en_input.get_shape())
                
                conv_1_1 = nf.convolution_layer(en_input, model_params["conv_1"], [1,2,2,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)               
                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)     
                conv_1_3 = nf.convolution_layer(conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)     
                #conv_1, conv_1_att = nf.channel_attention(conv_1_1, name='conv1_att')
                conv_1 = conv_1_1 + conv_1_3
                print("conv_1: %s" % conv_1.get_shape())       

                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_2_3 = nf.convolution_layer(conv_2_2, model_params["conv_2"], [1,1,1,1], name="conv_2_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_2, conv_2_att = nf.channel_attention(conv_2_1, name='conv2_att')
                conv_2 = conv_2_1 + conv_2_3
                print("conv_2: %s" % conv_2.get_shape())       
                
                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                conv_3_3 = nf.convolution_layer(conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training)
                #conv_3, conv_3_att = nf.channel_attention(conv_3_1, name='conv3_att')
                conv_3 = conv_3_1 + conv_3_3
                print("conv_3: %s" % conv_3.get_shape())              
                                
                en_code = nf.convolution_layer(conv_3, model_params["conv_code"], [1,1,1,1], name="conv_code", padding='VALID', activat_fn=nf.lrelu)
                print("en_code: %s" % en_code.get_shape())                       

                en_code_reshape = tf.reshape(en_code, [tf.shape(self.inputs)[0], 512])
                cls_output = nf.fc_layer(en_code_reshape, model_params["cls_fc_5"], name="en_cls_fc", activat_fn=nf.lrelu)
                print("cls_output: %s" % cls_output.get_shape())    

                #att = tf.concat([conv_1_att, conv_2_att, conv_3_att], -1)
                #print("att weight: %s" % att.get_shape())    
                att = None
                
            return en_code, cls_output, att
        
        if mode is "discriminator":
            with tf.variable_scope("discriminator", reuse=reuse):          
                
                dis_input = kwargs["dis_input"]
                
                print("[Discriminator] input: %s" % dis_input.get_shape())
                
                dis_conv_1_1 = nf.convolution_layer(dis_input, model_params["dis_conv_1"], [1,2,2,1], name="dis_conv_1_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_1_2 = nf.convolution_layer(dis_conv_1_1, model_params["dis_conv_1"], [1,1,1,1], name="dis_conv_1_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_1_3 = nf.convolution_layer(dis_conv_1_2, model_params["dis_conv_1"], [1,1,1,1], name="dis_conv_1_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_1, dis_conv_1_att = nf.channel_attention(dis_conv_1_3, name='dis_conv1_att')
                dis_conv_1 = dis_conv_1 + dis_conv_1_1
                print("dis_conv_1: %s" % dis_conv_1.get_shape())     

                dis_conv_2_1 = nf.convolution_layer(dis_conv_1, model_params["dis_conv_2"], [1,2,2,1], name="dis_conv_2_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_2_2 = nf.convolution_layer(dis_conv_2_1, model_params["dis_conv_2"], [1,1,1,1], name="dis_conv_2_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_2_3 = nf.convolution_layer(dis_conv_2_2, model_params["dis_conv_2"], [1,1,1,1], name="dis_conv_2_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_2, dis_conv_2_att = nf.channel_attention(dis_conv_2_3, name='dis_conv2_att')
                dis_conv_2 = dis_conv_2 + dis_conv_2_1
                print("dis_conv_2: %s" % dis_conv_2.get_shape())     

                dis_conv_3_1 = nf.convolution_layer(dis_conv_2, model_params["dis_conv_3"], [1,2,2,1], name="dis_conv_3_1", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_3_2 = nf.convolution_layer(dis_conv_3_1, model_params["dis_conv_3"], [1,1,1,1], name="dis_conv_3_2", padding='SAME', activat_fn=nf.lrelu)
                dis_conv_3_3 = nf.convolution_layer(dis_conv_3_2, model_params["dis_conv_3"], [1,1,1,1], name="dis_conv_3_3", padding='SAME', activat_fn=nf.lrelu)                
                dis_conv_3, dis_conv_3_att = nf.channel_attention(dis_conv_3_3, name='dis_conv3_att')
                dis_conv_3 = dis_conv_3 + dis_conv_3_1
                print("dis_conv_3: %s" % dis_conv_3.get_shape())     
                
                dis_conv_4 = nf.convolution_layer(dis_conv_3, model_params["dis_conv_4"], [1,1,1,1], name="dis_conv_4", padding='SAME', activat_fn=nf.lrelu)
                print("dis_conv_4: %s" % dis_conv_4.get_shape())     
                
                return dis_conv_4   

        if mode is "code_dis":
            with tf.variable_scope("code_dis", reuse=reuse):          
                
                code_dis_input = kwargs["code_dis_input"]
                
                code_dis_input = tf.reshape(code_dis_input, [tf.shape(self.inputs)[0], 512])
                
                print("[Code_Discriminator] input: %s" % code_dis_input.get_shape())
                
                dis_fc_1_1 = nf.fc_layer(code_dis_input, model_params["dis_fc_1"], name="dis_fc_1_1", activat_fn=nf.lrelu)
                dis_fc_1_2 = nf.fc_layer(dis_fc_1_1, model_params["dis_fc_1"], name="dis_fc_1_2", activat_fn=nf.lrelu)
                dis_fc_1_3 = nf.fc_layer(dis_fc_1_2, model_params["dis_fc_1"], name="dis_fc_1_3", activat_fn=nf.lrelu)
                dis_fc_1 = dis_fc_1_1 + dis_fc_1_3
                print("dis_fc_1: %s" % dis_fc_1.get_shape())                  
                
                dis_fc_2_1 = nf.fc_layer(dis_fc_1, model_params["dis_fc_2"], name="dis_fc_2_1", activat_fn=nf.lrelu)
                dis_fc_2_2 = nf.fc_layer(dis_fc_2_1, model_params["dis_fc_2"], name="dis_fc_2_2", activat_fn=nf.lrelu)
                dis_fc_2_3 = nf.fc_layer(dis_fc_2_2, model_params["dis_fc_2"], name="dis_fc_2_3", activat_fn=nf.lrelu)
                dis_fc_2 = dis_fc_2_1 + dis_fc_2_3
                print("dis_fc_2: %s" % dis_fc_2.get_shape())                  

                dis_fc_3_1 = nf.fc_layer(dis_fc_2, model_params["dis_fc_3"], name="dis_fc_3_1", activat_fn=nf.lrelu)
                dis_fc_3_2 = nf.fc_layer(dis_fc_3_1, model_params["dis_fc_3"], name="dis_fc_3_2", activat_fn=nf.lrelu)
                dis_fc_3_3 = nf.fc_layer(dis_fc_3_2, model_params["dis_fc_3"], name="dis_fc_3_3", activat_fn=nf.lrelu)
                dis_fc_3 = dis_fc_3_1 + dis_fc_3_3
                print("dis_fc_3: %s" % dis_fc_3.get_shape())                  

                dis_fc_4_1 = nf.fc_layer(dis_fc_3, model_params["dis_fc_4"], name="dis_fc_4_1", activat_fn=nf.lrelu)
                dis_fc_4_2 = nf.fc_layer(dis_fc_4_1, model_params["dis_fc_4"], name="dis_fc_4_2", activat_fn=nf.lrelu)
                dis_fc_4_3 = nf.fc_layer(dis_fc_4_2, model_params["dis_fc_4"], name="dis_fc_4_3", activat_fn=nf.lrelu)
                dis_fc_4 = dis_fc_4_1 + dis_fc_4_3
                print("dis_fc_4: %s" % dis_fc_4.get_shape())                  

                dis_fc_5 = nf.fc_layer(dis_fc_4, model_params["dis_fc_5"], name="dis_fc_5", activat_fn=nf.lrelu)
                print("dis_fc_5: %s" % dis_fc_5.get_shape())                  
                
                return dis_fc_5                            

        if mode is "code_cls":
            with tf.variable_scope("code_cls", reuse=reuse):          
                
                code_cls_input = kwargs["code_cls_input"]

                code_cls_input = tf.reshape(code_cls_input, [tf.shape(self.inputs)[0], 512])
                
                print("[Code_Classifier] input: %s" % code_cls_input.get_shape())
                               
                cls_fc_1_1 = nf.fc_layer(code_cls_input, model_params["cls_fc_1"], name="cls_fc_1_1", activat_fn=nf.lrelu)
                cls_fc_1_2 = nf.fc_layer(cls_fc_1_1, model_params["cls_fc_1"], name="cls_fc_1_2", activat_fn=nf.lrelu)
                cls_fc_1_3 = nf.fc_layer(cls_fc_1_2, model_params["cls_fc_1"], name="cls_fc_1_3", activat_fn=nf.lrelu)
                cls_fc_1 = cls_fc_1_1 + cls_fc_1_3
                print("cls_fc_1: %s" % cls_fc_1.get_shape())                  
                
                cls_fc_2_1 = nf.fc_layer(cls_fc_1, model_params["cls_fc_2"], name="cls_fc_2_1", activat_fn=nf.lrelu)
                cls_fc_2_2 = nf.fc_layer(cls_fc_2_1, model_params["cls_fc_2"], name="cls_fc_2_2", activat_fn=nf.lrelu)
                cls_fc_2_3 = nf.fc_layer(cls_fc_2_2, model_params["cls_fc_2"], name="cls_fc_2_3", activat_fn=nf.lrelu)
                cls_fc_2 = cls_fc_2_1 + cls_fc_2_3
                print("cls_fc_2: %s" % cls_fc_2.get_shape())                  

                cls_fc_3_1 = nf.fc_layer(cls_fc_2, model_params["cls_fc_3"], name="cls_fc_3_1", activat_fn=nf.lrelu)
                cls_fc_3_2 = nf.fc_layer(cls_fc_3_1, model_params["cls_fc_3"], name="cls_fc_3_2", activat_fn=nf.lrelu)
                cls_fc_3_3 = nf.fc_layer(cls_fc_3_2, model_params["cls_fc_3"], name="cls_fc_3_3", activat_fn=nf.lrelu)
                cls_fc_3 = cls_fc_3_1 + cls_fc_3_3
                print("cls_fc_3: %s" % cls_fc_3.get_shape())                  

                cls_fc_4_1 = nf.fc_layer(cls_fc_3, model_params["cls_fc_4"], name="cls_fc_4_1", activat_fn=nf.lrelu)
                cls_fc_4_2 = nf.fc_layer(cls_fc_4_1, model_params["cls_fc_4"], name="cls_fc_4_2", activat_fn=nf.lrelu)
                cls_fc_4_3 = nf.fc_layer(cls_fc_4_2, model_params["cls_fc_4"], name="cls_fc_4_3", activat_fn=nf.lrelu)
                cls_fc_4 = cls_fc_4_1 + cls_fc_4_3
                print("cls_fc_4: %s" % cls_fc_4.get_shape())                  

                cls_fc_5 = nf.fc_layer(cls_fc_4, model_params["cls_fc_5"], name="cls_fc_5", activat_fn=nf.lrelu)
                print("cls_fc_5: %s" % cls_fc_5.get_shape())                  
                
                return cls_fc_5           

    def RaGAN_MNIST(self, kwargs):

        reuse = kwargs["reuse"]
        
        net = kwargs["net"]
        
        DEPTH = 28
        OUTPUT_SIZE = 28
        batch_size = tf.shape(self.inputs)[0]

        model_params = {
                
                "g_fc_1": 2*2*8*DEPTH,
                "g_deconv_1": [5,5,4*DEPTH],
                "g_deconv_2": [5,5,2*DEPTH],
                "g_deconv_3": [5,5,1*DEPTH],
                "g_deconv_4": [5,5,1],

                "d_conv_1": [5,5,1*DEPTH],
                "d_conv_2": [5,5,2*DEPTH],
                "d_conv_3": [5,5,4*DEPTH],
                "d_fc_1": 1,
        }
                
        if net is "Gen":
        
            ###Generator
            g_input = self.inputs
            with tf.variable_scope("gen", reuse=reuse):     

                noise = g_input

                g_fc_1 = nf.fc_layer(noise, model_params["g_fc_1"], name="g_fc_1", activat_fn=tf.nn.relu)
                g_fc_1 = tf.reshape(g_fc_1, [batch_size, 2, 2, 8*DEPTH])
                       
                g_deconv_1 = nf.deconvolution_layer(g_fc_1, model_params["g_deconv_1"], [batch_size, 4, 4, 4*DEPTH], [1,2,2,1], name="g_deconv_1", padding='SAME', activat_fn=tf.nn.relu)
        
                g_deconv_2 = nf.deconvolution_layer(g_deconv_1, model_params["g_deconv_2"], [batch_size, 7, 7, 2*DEPTH], [1,2,2,1], name="g_deconv_2", padding='SAME', activat_fn=tf.nn.relu)        
        
                g_deconv_3 = nf.deconvolution_layer(g_deconv_2, model_params["g_deconv_3"], [batch_size, 14, 14, DEPTH], [1,2,2,1], name="g_deconv_3", padding='SAME', activat_fn=tf.nn.relu)                
        
                g_deconv_4 = nf.deconvolution_layer(g_deconv_3, model_params["g_deconv_4"], [batch_size, OUTPUT_SIZE, OUTPUT_SIZE, 1], [1,2,2,1], name="g_deconv_4", padding='SAME', activat_fn=tf.nn.sigmoid)                        
                
                return tf.reshape(g_deconv_4, [-1, 784])           

        elif net is "Dis":
            
            d_inputs = kwargs["d_inputs"]
            
            ###Discriminator
            input_gan = d_inputs 
            with tf.variable_scope("dis", reuse=reuse):     
                
                input_gan = tf.reshape(input_gan, [-1, 28, 28, 1])

                d_conv_1 = nf.convolution_layer(input_gan, model_params["d_conv_1"], [1,2,2,1], name="d_conv_1", padding='SAME', activat_fn=nf.lrelu)
        
                d_conv_2 = nf.convolution_layer(d_conv_1, model_params["d_conv_2"], [1,2,2,1], name="d_conv_2", padding='SAME', activat_fn=nf.lrelu)
        
                d_conv_3 = nf.convolution_layer(d_conv_2, model_params["d_conv_3"], [1,2,2,1], name="d_conv_3", padding='SAME', activat_fn=nf.lrelu)
               
                chanel = d_conv_3.get_shape().as_list()
                output = tf.reshape(d_conv_3, [batch_size, chanel[1]*chanel[2]*chanel[3]])
                output = nf.fc_layer(output, model_params["d_fc_1"], name="d_fc_1", activat_fn=None)
                
                return output
        
def unit_test():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"resNet_v1")
    return mz.build_model()
