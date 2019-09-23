import tensorflow as tf

import os
import sys
sys.path.append('./utility')

from utils import mkdir_p
from model import model
from utils import CelebA, InputData
import config
import argparse

#flags = tf.app.flags
#
#flags.DEFINE_integer("batch_size" , 64, "batch size")
#flags.DEFINE_integer("max_iters" , 600000, "the maxmization epoch")
#flags.DEFINE_integer("latent_dim" , 128, "the dim of latent code")
#flags.DEFINE_float("learn_rate_init" , 0.0001, "the init of learn rate")
##Please set this num of repeat by the size of your datasets.
#flags.DEFINE_integer("repeat", 10000, "the numbers of repeat for your datasets")
#flags.DEFINE_string("path", '/home/?/data/', "for example, '/home/jack/data/' is the directory of your celebA data")
#flags.DEFINE_integer("op", 0, "Training or Test")

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="example",help="Configuration name")
args = parser.parse_args()
conf = config.config(args.config).config["common"]

#FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = conf["log_dir"]
    checkpoint_dir = conf["ckpt_dir"]
       
    mkdir_p(root_log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(os.path.join(checkpoint_dir, 'best_performance'))

    model_path = checkpoint_dir

    max_iters = conf["max_iters"]
    data_repeat = conf["repeat"]
    dropout = conf["dropout"]
    ckpt_name = conf["ckpt_name"]
    test_ckpt = conf["test_ckpt"]
    train_ckpt = conf["train_ckpt"]
    restore_model = conf["restore_model"]
    restore_step = conf["restore_step"]
    model_ticket = conf["model_ticket"]
    
    is_training = conf["is_training"]
    
    learn_rate_init = conf["learn_rate_init"]
    

    print("===================================================================")
    if is_training == True:
        batch_size = conf["batch_size"]
        print("*** [Training] ***")
        print("restore_model: [{}]".format(restore_model))
        print("train_data_path: [{}]".format(conf["train_data_path"]))
        print("valid_data_path: [{}]".format(conf["valid_data_path"]))
        print("anomaly_data_path: [{}]".format(conf["anomaly_data_path"]))
        print("ckpt_name: [{}]".format(ckpt_name))
        print("max_iters: [{}]".format(max_iters))   
        print("learn_rate_init: [{}]".format(learn_rate_init))
    else:
        batch_size = 1000
        print("*** [Testing] ***")
        print("test_data_path: [{}]".format(conf["test_data_path"]))

    print("batch_size: [{}]".format(batch_size))   
    print("model_ticket: [{}]".format(model_ticket))   
    print("dropout: [{}]".format(dropout))
    print("===================================================================")
    
    if is_training == True:               
        
        cb_ob = InputData(conf["train_data_path"], conf["valid_data_path"], conf["anomaly_data_path"], None)

        MODEL = model(  batch_size=batch_size, 
                        max_iters=max_iters, 
                        repeat=data_repeat,
                        dropout=dropout,
                        model_path=model_path, 
                        data_ob=cb_ob, 
                        log_dir=root_log_dir, 
                        learnrate_init=learn_rate_init,
                        ckpt_name=ckpt_name,
                        test_ckpt=test_ckpt,
                        train_ckpt=train_ckpt,
                        restore_model=restore_model,
                        restore_step=restore_step,
                        model_ticket=model_ticket,
                        is_training=is_training)
        
        MODEL.build_model()
        MODEL.train()

    else:
        
        test_data_path = conf["test_data_path"]
        
        for path in test_data_path:
        
            cb_ob = InputData(None, None, None, path)
    
            MODEL = model(  batch_size=batch_size, 
                            max_iters=max_iters, 
                            repeat=data_repeat,
                            dropout=dropout,
                            model_path=model_path, 
                            data_ob=cb_ob, 
                            log_dir=root_log_dir, 
                            learnrate_init=learn_rate_init,
                            ckpt_name=ckpt_name,
                            test_ckpt=test_ckpt,
                            train_ckpt=train_ckpt,
                            restore_model=restore_model,
                            restore_step=restore_step,
                            model_ticket=model_ticket,
                            is_training=is_training)        
            
            MODEL.build_eval_model()
            MODEL.test()









