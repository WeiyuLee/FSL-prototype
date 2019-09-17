class config:

    def __init__(self, configuration):
		
        self.configuration = configuration
        self.config = {
                        "common":{},
        						"train":{},
                        "test":{},
                        }
        self.get_config()


    def get_config(self):

        try:
            conf = getattr(self, self.configuration)
            conf()

        except: 
            print("Can not find configuration")
            raise

    def Default_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 128
        common_config["max_iters"] = 30000
        common_config["learn_rate_init"] = 0.001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0.5
        
        #common_config["model_ticket"] = "cifar10_alexnet_att"        
        #common_config["model_ticket"] = "BCL" 
        #common_config["model_ticket"] = "BCL_att" 
        #common_config["model_ticket"] = "BCL_att_v2" 
        #common_config["model_ticket"] = "BCL_att_v3" 
        #common_config["model_ticket"] = "BCL_att_v4" 
        common_config["model_ticket"] = "BCL_att_GAN" 
        
        #common_config["ckpt_name"] = "cifar10_alexnet_att"                
        #common_config["ckpt_name"] = "BCL"       
        #common_config["ckpt_name"] = "BCL_att"      
        #common_config["ckpt_name"] = "BCL_att_01"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att_8"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att_7"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att_5"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att_3"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att_1"  
        #common_config["ckpt_name"] = "BCL_att_001_L2_att_0"  
        #common_config["ckpt_name"] = "BCL_att_0001"  
        #common_config["ckpt_name"] = "BCL_att_9"    
        #common_config["ckpt_name"] = "BCL_att_9_v2" 
        #common_config["ckpt_name"] = "BCL_att_9_v3" 
        #common_config["ckpt_name"] = "BCL_att_9_v4" 
        #common_config["ckpt_name"] = "BCL_att_9_v5" 
        #common_config["ckpt_name"] = "BCL_att_8"  
        #common_config["ckpt_name"] = "BCL_att_8_v2" 
        #common_config["ckpt_name"] = "BCL_att_8_v3" 
        #common_config["ckpt_name"] = "BCL_att_7"  
        #common_config["ckpt_name"] = "BCL_att_7_v2"  
        #common_config["ckpt_name"] = "BCL_att_7_v3" 
        #common_config["ckpt_name"] = "BCL_att_7_v4" 
        #common_config["ckpt_name"] = "BCL_att_7_v5" 
        #common_config["ckpt_name"] = "BCL_att_6"      
        #common_config["ckpt_name"] = "BCL_att_6_2"      
        #common_config["ckpt_name"] = "BCL_att_6_3"      
        #common_config["ckpt_name"] = "BCL_att_6_v2" 
        #common_config["ckpt_name"] = "BCL_att_6_v3" 
        #common_config["ckpt_name"] = "BCL_att_6_v4" 
        #common_config["ckpt_name"] = "BCL_att_6_v5" 
        #common_config["ckpt_name"] = "BCL_att_5"      
        #common_config["ckpt_name"] = "BCL_att_5_v2" 
        #common_config["ckpt_name"] = "BCL_att_5_v3" 
        #common_config["ckpt_name"] = "BCL_att_4_2"    
        #common_config["ckpt_name"] = "BCL_att_4_v2" 
        #common_config["ckpt_name"] = "BCL_att_4_v3" 
        #common_config["ckpt_name"] = "BCL_att_3"      
        #common_config["ckpt_name"] = "BCL_att_3_v2"      
        #common_config["ckpt_name"] = "BCL_att_3_v3" 
        #common_config["ckpt_name"] = "BCL_att_3_v4" 
        #common_config["ckpt_name"] = "BCL_att_2"      
        #common_config["ckpt_name"] = "BCL_att_2_v2"      
        #common_config["ckpt_name"] = "BCL_att_2_v3" 
        #common_config["ckpt_name"] = "BCL_att_1" 
        #common_config["ckpt_name"] = "BCL_att_1_v2" 
        #common_config["ckpt_name"] = "BCL_att_1_v3" 
        #common_config["ckpt_name"] = "BCL_att_1_v4"
        #common_config["ckpt_name"] = "BCL_att_0"    
        #common_config["ckpt_name"] = "BCL_att_0_v2" 
        #common_config["ckpt_name"] = "BCL_att_0_v3" 
        #common_config["ckpt_name"] = "BCL_att_0_v4" 
        common_config["ckpt_name"] = "BCL_att_GAN"  
        
        common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_9.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_8.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_7.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_6.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_5.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_4.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_3.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_2.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_1.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_0.p"
    
        common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_9.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_8.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_7.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_6.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_5.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_4.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_3.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_2.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_1.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_0.p"
        
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p"
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_9.p"
        common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_9.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_8.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_7.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_6.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_5.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_4.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_3.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_2.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_1.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_0.p"]
        
        common_config["ckpt_dir"] = "/home/sdc1/model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = "/home/sdc1/model/FSL/FSL-prototype/BCL_att_001_L2_att_3/best_performance/BCL_att_001_L2_att_3_0.4593_0.8879-7400"                
        #common_config["test_ckpt"] = "/home/sdc1/model/FSL/FSL-prototype/BCL_att_9_v4/BCL_att_9_v4-40000"        
        common_config["train_ckpt"] = [] 
        common_config["log_dir"] = "/home/sdc1/model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]                         
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0        

    def AD_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 128
        common_config["max_iters"] = 300000
        common_config["learn_rate_init"] = 0.001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0.5

        #common_config["model_ticket"] = "AD_att_GAN" 
        common_config["model_ticket"] = "AD_att_GAN_v2" 
        
        #common_config["ckpt_name"] = "AD_att_GAN_dropout"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_9_1e8"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_9_1e8_d5"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_9_1e8_d5_codeattention"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_9_1e6_d5_codeattention"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_9_1e3_d5_codeattention"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_9_1e1_d5_codeattention"  
        #common_config["ckpt_name"] = "AD_att_GAN_dropout_5"  
        common_config["ckpt_name"] = "AD_att_GAN_v2_dropout_9"          
        
        common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_9.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_8.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_7.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_6.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_5.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_4.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_3.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_2.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_1.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_0.p"
    
        common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_9.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_8.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_7.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_6.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_5.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_4.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_3.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_2.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_1.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_0.p"
        
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p"
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_9.p"
        common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_9.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_8.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_7.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_6.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_5.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_4.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_3.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_2.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_1.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_0.p"]
        
        common_config["ckpt_dir"] = "/home/sdc1/model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = "/home/sdc1/model/FSL/FSL-prototype/AD_att_GAN_v2_dropout_9/best_performance/AD_att_GAN_v2_dropout_9_0.5846_0.8667-37400"                
        common_config["train_ckpt"] = [] 
        common_config["log_dir"] = "/home/sdc1/model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]                         
        #common_config["is_training"] = True
        common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0        

    def AD_VAE_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 32
        common_config["max_iters"] = 60000
        common_config["learn_rate_init"] = 0.0001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0

        #common_config["model_ticket"] = "AD_att_VAE" 
        #common_config["model_ticket"] = "AD_att_VAE_WEAK" 
        common_config["model_ticket"] = "AD_att_VAE_GAN" 
  
        #common_config["ckpt_name"] = "AD_att_VAE_v1"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_256"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_256_iter_5"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_256_iter_5_1_100"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_512_iter_5_1_100"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_512_iter_5_1_10"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_512_iter_5_1_1"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_512_iter_5_1_10_v2"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_256_iter_5_1_1"
        #common_config["ckpt_name"] = "AD_att_VAE_v1_code_256_iter_5_1_10_v2"
        
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_512_temp"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_1024_temp"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_512_1_1"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_512_1_10"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_1024_recon_cls"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_1024_recon_cls_L1"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_1024_recon_cls_L1_att"
        #common_config["ckpt_name"] = "AD_att_VAE_WEAK_v1_code_1024_recon_cls_L1_att_D"
        
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v2"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D25"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D38"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D50"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D75"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D100"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D50_CodeD50"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D50_CodeD10"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v3_D50_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v4_D50_CodeD50"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v4_D50_CodeD10"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v4_D50_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v4_D50_CodeD01"
        
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v5_RaGAN_D50_CodeD001"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v5_RaGAN_D50_CodeD025"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v5_RaGAN_D50_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v5_RaGAN_D50_CodeD50"
        
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v6_RaGAN_D50_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v6_RaGAN_D50_CodeD1_bound"
        
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v7_RaGAN_D50_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v7_RaGAN_R50_D1_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v7_RaGAN_R1_D50_CodeD1"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v7_RaGAN_R1_D50_CodeD1_seperate"
        #common_config["ckpt_name"] = "AD_att_VAE_GAN_v7_RaGAN_R1_D50_CodeD1_seperate_bn"
        common_config["ckpt_name"] = "AD_att_VAE_GAN_v7_RaGAN_R1_D50_CodeD1_seperate_bn_2"
        
        common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_train_class_9.p"
        common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p"
        common_config["anomaly_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_5.p"
        
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_9.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_8.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_7.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_6.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_5.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_4.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_3.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_2.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_1.p"
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_0.p"
    
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_9.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_8.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_7.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_6.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_5.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_4.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_3.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_2.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_1.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_0.p"
        
        common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p", "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_5.p"]
        
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p"
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_9.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_9.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_8.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_7.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_6.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_5.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_4.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_3.p"]
        #common_config["test_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_2.p"
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_1.p"]
        #common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/preprocessed/test.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed/preprocess_train_0.p"]
        
        common_config["ckpt_dir"] = "/home/sdc1/model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = "/home/sdc1/model/FSL/FSL-prototype/AD_att_VAE_GAN_v5_RaGAN_D50_CodeD1/best_performance/AD_att_VAE_GAN_v5_RaGAN_D50_CodeD1-8000"                
        common_config["train_ckpt"] = [] 
        common_config["log_dir"] = "/home/sdc1/model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]                         
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0      
        
    def Temp_att_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 600000
        common_config["latent_dim"] = 256
        common_config["learn_rate_init"] = 0.0001
        common_config["repeat"] = 10000
        
        #common_config["model_ticket"] = "Cat_ae_att_dist"
        #common_config["model_ticket"] = "Cat_vaegan_att_dist"     
        #common_config["model_ticket"] = "Cat_vaegan_att"     
        common_config["model_ticket"] = "Cat_ae_att_dist_z"        
        #common_config["ckpt_name"] = "Cat_att_ckpt"     
        #common_config["ckpt_name"] = "Cat_att_ckpt_att_loss_2"
        #common_config["ckpt_name"] = "Cat_att_ckpt_test"
        #common_config["ckpt_name"] = "Cat_ae_att_ckpt_att_loss"        
        #common_config["ckpt_name"] = "Cat_ae_att_ckpt"        
        #common_config["ckpt_name"] = "Cat_ae_att_ckpt_att_loss_2"        
        #common_config["ckpt_name"] = "Cat_ae_att_ckpt_2"        
        #common_config["ckpt_name"] = "Cat_vaegan_att"     
        common_config["ckpt_name"] = "Cat_ae_att_ckpt_z"           
        
        common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cat/cats_bigger_than_64x64/"
        common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/temp_cat", "/home/sdc1/dataset/FSL/temp_cat2", "/home/sdc1/dataset/FSL/temp_celeba", "/home/sdc1/dataset/FSL/temp_dog"]
        common_config["sample_path"] = "/home/sdc1/model/FSL/vae-gan-tensorflow/sample/" + common_config["ckpt_name"]       
        common_config["test_sample_path"] = "/home/sdc1/model/FSL/vae-gan-tensorflow/test_sample/" + common_config["ckpt_name"]       
        common_config["ckpt_dir"] = "/home/sdc1/model/FSL/vae-gan-tensorflow/temp/"       
        common_config["test_ckpt"] = "/home/sdc1/model/FSL/vae-gan-tensorflow/temp/Cat_ae_att_ckpt_z-38000"                
        common_config["train_ckpt"] = []
        common_config["log_dir"] = "/home/sdc1/model/FSL/vae-gan-tensorflow/log/" + common_config["ckpt_name"]                         
        common_config["op"] = 1       
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0        