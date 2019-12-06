from uuid import getnode as get_mac

class config:

    def __init__(self, configuration):
		
        self.configuration = configuration
        self.config = {
                        "common":{},
        						"train":{},
                        "test":{},
                        }

        self.mac = get_mac()
        if self.mac == 189250941727334:
            self.default_path = "/data/wei/"
        elif self.mac == 229044592702658:
            self.default_path = "/home/sdc1/"

        self.get_config()

    def get_config(self):

        try:
            conf = getattr(self, self.configuration)
            conf()

        except: 
            print("Can not find configuration")
            raise
  
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
        #common_config["model_ticket"] = "AD_att_VAE_GAN" 
        #common_config["model_ticket"] = "AD_att_AE_GAN" 
        #common_config["model_ticket"] = "AD_att_AE_GAN_3DCode" 
        #common_config["model_ticket"] = "AD_att_AE_GAN_3DCode_32x32"
        #common_config["model_ticket"] = "AD_att_AE_GAN_CLS"
        common_config["model_ticket"] = "AD_att_AE_GAN_CLS_DISE"

        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_ATTmap"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_ATTmap_v2"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_ATTmap_v2_64x64"
        
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v2"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v2_tanh"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v3_tanh"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v4_tanh"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v5_tanh"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v6_tanh"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_32x32_2stage_train_v7_tanh"
        #common_config["ckpt_name"] = "AD_AE_GAN_3DCode_Temp_64x64"
        
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v1"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v2"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v3"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v4_woDise"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v4_woDise_38cls"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v4_woDise_25cls"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v4_woDise_10cls"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_v4_woDise_5cls"
        
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_DISE_v1_25cls_25floss"
        #common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_DISE_v1_50cls_25floss_25cdis"
        common_config["ckpt_name"] = "AD_att_AE_GAN_CLS_DISE_v1_0cls_25floss_50cdis"
        
        common_config["anomaly_class"] = 9
        
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_train_class_9.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p"
        #common_config["anomaly_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug_64x64/pr_test_class_9.p"
        #common_config["anomaly_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p"
        
        common_config["train_cls_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug"
        common_config["valid_cls_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug"
        common_config["anomaly_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p"
        
        common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_9.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_8.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_7.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_6.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_5.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_4.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_3.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_2.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_1.p"
        #common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_0.p"
    
        common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_9.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_8.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_7.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_6.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_5.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_4.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_3.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_2.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_1.p"
        #common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_0.p"
        
        common_config["test_data_path"] = [self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p", self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_7.p"]
        
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
        
        common_config["ckpt_dir"] = self.default_path + "model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/FSL/FSL-prototype/AD_AE_GAN_v1_R50_D1_2/AD_AE_GAN_v1_R50_D1_2-60000"                
        common_config["train_ckpt"] = self.default_path + "model/FSL/FSL-prototype/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 6001  

    def AD_DISE_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 500000
        common_config["learn_rate_init"] = 0.0001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0.5

        #common_config["model_ticket"] = "AD_DISE"
        #common_config["model_ticket"] = "AD_CLS_DISE"        
        #common_config["model_ticket"] = "AD_CLS_DISE2"        
        common_config["model_ticket"] = "AD_CLS_DISE3"        

        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_1MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_2MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"               
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_5MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"       
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_10MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_25MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"               
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_50CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"      
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_10CL_10DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid" ###############################
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_10CL_25DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"      
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_100MSE_10CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid"
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid_LR"       
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_50CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid_LR"
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50L1_50CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid_LR"
        
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_10CL_10DL_1D_64BCH_128DIM_CON_allcls_sigmoid_cifar"      
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_64BCH_128DIM_CON_allcls_sigmoid_cifar_LR"      
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50L1_25CL_1DL_1D_BN_LIN_DRP05_64BCH_128DIM_CON_allcls_sigmoid_cifar_LR"
        
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_retrain"       
        #common_config["ckpt_name"] = "AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_cifar_retrain"       
        
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50MSE_25CL_1DL_1D_1KL"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50MSE_25CL_1DL_1D_1KL_cifar"       
        
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DL_1D_1D2"   
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DL_1D_1D2_cifar"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_cifar"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar"       ###
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DL_1D_1D2_128DIM_cifar"       
        
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_0D_1D2_128DIM"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_0D_1D2_128DIM_cifar"       

        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano0"                    
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano0_zt"            
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano1"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano2"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano3"              
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano4"                    
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano5"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano6"      
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano7"      
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano8"      
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_256DIM_cifar_ano9_zt"       
        
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_64DIM_cifar_ano9_zt"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_64DIM_cifar_ano9_zt_weakD"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_256DIM_cifar_ano9_zt_weakD"   
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_256DIM_cifar_ano9_zt_weakD_testspeed"   
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_2nd_256DIM_cifar_ano9_zt_weakD"
        
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt"              
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_256DIM_cifar_ano9_zt"              
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_64DIM_cifar_ano9_zt_ChangeBatch"      
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt_ChangeBatch"             
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v2_50L1_25CL_1DLL1_1D_1D2_256DIM_cifar_ano9_zt_ChangeBatch"             
        
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_75L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt"      
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_100L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_100L1_25CL_1DLL1_5D_1D2_128DIM_cifar_ano9_zt"    
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_200L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt"       
        #common_config["ckpt_name"] = "AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt_woadv"       
        
        #common_config["ckpt_name"] = "AD_CLS_DISE3_v1_50L1_25CL_1DLL1_1D_128DIM_cifar_ano9_zt"     
        #common_config["ckpt_name"] = "AD_CLS_DISE3_v1_50L1_25CL_1DLL1_1D_128DIM_cifar_ano9_zt_2fakesrc"     
        common_config["ckpt_name"] = "AD_CLS_DISE3_v1_50L1_25CL_1DLL1_1D_128DIM_cifar_ano9_zt_2fakesrc_lr"     
        
        #common_config["anomaly_class"] = [0,1,3,4,5,7,8,9]
        common_config["anomaly_class"] = 9
        
#        # SVHN ==================================================================================================================        
#        common_config["lat_dim"] = 128
#        common_config["train_cls_data_path"] = self.default_path + "dataset/FSL/SVHN/pr_single_class"
#        common_config["valid_cls_data_path"] = self.default_path + "dataset/FSL/SVHN/pr_single_class"
#        
#        common_config["train_data_path"] = self.default_path + "dataset/FSL/SVHN/preprocessed/preprocess_train_9.p"
#        common_config["valid_data_path"] = self.default_path + "dataset/FSL/SVHN/preprocessed/preprocess_test_9.p"        
#        common_config["anomaly_data_path"] = self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_9.p"                
#        common_config["test_data_path"] = [  self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_0.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_1.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_2.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_3.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_4.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_5.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_6.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_7.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_8.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_9.p"  ]
#        common_config["output_dir"] = self.default_path + "dataset/FSL/SVHN/FSL/" + common_config["ckpt_name"]                                  

        # Cifar-10 ==============================================================================================================
        #common_config["lat_dim"] = 256
        common_config["lat_dim"] = 128
        #common_config["lat_dim"] = 64
        common_config["train_cls_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class"
        common_config["valid_cls_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class"

        common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_" + str(common_config["anomaly_class"]) + ".p"
        common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_" + str(common_config["anomaly_class"]) + ".p"        
        common_config["anomaly_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_" + str(common_config["anomaly_class"]) + ".p"                
        common_config["test_data_path"] = [ self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_0.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_1.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_2.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_3.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_4.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_5.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_6.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_7.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_8.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p" ]
        common_config["output_dir"] = self.default_path + "dataset/FSL/Cifar-10/FSL/" + common_config["ckpt_name"]                                  
        
        # =======================================================================================================================
        common_config["ckpt_dir"] = self.default_path + "model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/FSL/FSL-prototype/AD_CLS_DISE2_v1_100L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt/AD_CLS_DISE2_v1_100L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt-284000"               
        common_config["train_ckpt"] = self.default_path + "model/FSL/FSL-prototype/AD_CLS_DISE2_v1_75L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt/AD_CLS_DISE2_v1_75L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt-60000"
        common_config["log_dir"] = self.default_path + "model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]        
       
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 60000

    def AD_BASELINE_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 500000
        common_config["learn_rate_init"] = 0.0001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0.5

        common_config["model_ticket"] = "AD_CLS_BASELINE"        

        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano0"               
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano1"               
        common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano2"               
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano3"               
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano4"          
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano5"          
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano6"        
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano7"     
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano8"     
        #common_config["ckpt_name"] = "AD_CLS_BASELINE_v1_ano9"                    

        #common_config["anomaly_class"] = [0,1,3,4,5,7,8,9]
        common_config["anomaly_class"] = 2

#        # SVHN ==================================================================================================================        
#        common_config["lat_dim"] = 128
#        common_config["train_cls_data_path"] = self.default_path + "dataset/FSL/SVHN/pr_single_class"
#        common_config["valid_cls_data_path"] = self.default_path + "dataset/FSL/SVHN/pr_single_class"
#        
#        common_config["train_data_path"] = self.default_path + "dataset/FSL/SVHN/preprocessed/preprocess_train_9.p"
#        common_config["valid_data_path"] = self.default_path + "dataset/FSL/SVHN/preprocessed/preprocess_test_9.p"        
#        common_config["anomaly_data_path"] = self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_9.p"                
#        common_config["test_data_path"] = [  self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_0.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_1.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_2.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_3.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_4.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_5.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_6.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_7.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_8.p",
#                                             self.default_path + "dataset/FSL/SVHN/pr_single_class/pr_test_class_9.p"  ]
#        common_config["output_dir"] = self.default_path + "dataset/FSL/SVHN/FSL/" + common_config["ckpt_name"]                                  

        # Cifar-10 ==============================================================================================================
        common_config["lat_dim"] = []
        common_config["train_cls_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class"
        common_config["valid_cls_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class"

        common_config["train_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_train_" + str(common_config["anomaly_class"]) + ".p"
        common_config["valid_data_path"] = self.default_path + "dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_" + str(common_config["anomaly_class"]) + ".p"        
        common_config["anomaly_data_path"] = self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_" + str(common_config["anomaly_class"]) + ".p"                
        common_config["test_data_path"] = [ self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_0.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_1.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_2.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_3.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_4.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_5.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_6.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_7.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_8.p",
                                            self.default_path + "dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p" ]
        common_config["output_dir"] = self.default_path + "dataset/FSL/Cifar-10/FSL/" + common_config["ckpt_name"]                                  
        
        # =======================================================================================================================
        common_config["ckpt_dir"] = self.default_path + "model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/FSL/FSL-prototype/AD_CLS_BASELINE_v1_ano2/AD_CLS_BASELINE_v1_ano2-500000"               
        common_config["train_ckpt"] = self.default_path + "model/FSL/FSL-prototype/AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar/AD_CLS_DISE2_v1_50L1_25CL_1DLL1_1D_1D2_128DIM_cifar-462000"
        common_config["log_dir"] = self.default_path + "model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]        
       
        #common_config["is_training"] = True
        common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 462000

    def GANomaly_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 32
        common_config["max_iters"] = 60000
        common_config["learn_rate_init"] = 0.0001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0

        common_config["model_ticket"] = "GANomaly" 
       
        #common_config["ckpt_name"] = "GANomaly_v1_MNIST"
        #common_config["ckpt_name"] = "GANomaly_v1_CIFAR10"
        common_config["ckpt_name"] = "GANomaly_v1_512_CIFAR10"
        #common_config["ckpt_name"] = "GANomaly_v1_lrelu"
               
        #common_config["train_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_train_class_9.p"
        #common_config["valid_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p"
        common_config["anomaly_data_path"] = "/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p"
        
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
        
        common_config["test_data_path"] = ["/home/sdc1/dataset/FSL/Cifar-10/pr_single_class_aug/pr_test_class_9.p", "/home/sdc1/dataset/FSL/Cifar-10/preprocessed_aug/preprocess_test_9.p"]
        
        common_config["ckpt_dir"] = "/home/sdc1/model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = "/home/sdc1/model/FSL/FSL-prototype/AD_AE_GAN_3DCode_v3_512_R50_C10_10cls/AD_AE_GAN_3DCode_v3_512_R50_C10_10cls-60000"                
        common_config["train_ckpt"] = "/home/sdc1/model/FSL/FSL-prototype/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = "/home/sdc1/model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]                         
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0      

    def RaGAN_MNIST_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 60000
        common_config["learn_rate_init"] = 0.001
        common_config["repeat"] = 10000
        common_config["dropout"] = 0

        common_config["model_ticket"] = "RaGAN_MNIST" 
  
        common_config["ckpt_name"] = "RaGAN_MNIST_v1"
        
        common_config["train_data_path"] = None
        common_config["valid_data_path"] = None
        common_config["anomaly_data_path"] = None   
        common_config["test_data_path"] = [None, None]

        common_config["ckpt_dir"] = "/data/wei/model/FSL/FSL-prototype/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = []
        common_config["train_ckpt"] = [] 
        common_config["log_dir"] = "/data/wei/model/FSL/FSL-prototype/log/" + common_config["ckpt_name"]                         
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