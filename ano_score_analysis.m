clc; clear; close all;

%dataset = 'SVHN';
%model_name = 'AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_retrain';

dataset = 'Cifar-10';
model_name = 'AD_CLS_DISE2_v1_50L1_25CL_1DL_1D_1D2_cifar';

normal_class_num = 10;
anomaly_class_idx = 10;

fprintf(['Anomaly Class:' num2str(anomaly_class_idx-1) '\n']);

normal_score = [];
normal_z1 = [];
normal_z1_2nd = [];
anomaly_score = [];
anomaly_z1 = [];
anomaly_z1_2nd = [];
for i=1:normal_class_num
    curr_data = csvread([ 'T:/data/wei/dataset/FSL/' dataset '/FSL/' model_name '/pr_test_class_' num2str(i-1) ...
                          '/pr_test_class_' num2str(i-1) '_ano_score.csv']);    
    if i ~= anomaly_class_idx                               
        normal_score = [normal_score; curr_data(:, 1)];
        normal_z1 = [normal_z1; max(curr_data(:, 2:11), [], 2)];
        normal_z1_2nd = [normal_z1_2nd; max(curr_data(:, 12:21), [], 2)];
    else
        anomaly_score = [anomaly_score; curr_data(:, 1)];
        anomaly_z1 = [anomaly_z1; max(curr_data(:, 2:11), [], 2)];
        anomaly_z1_2nd = [anomaly_z1_2nd; max(curr_data(:, 12:21), [], 2)];
    end
end

nor_rdn_idx = randi([1 length(normal_score)], 256, 1);
ano_rdn_idx = randi([1 length(anomaly_score)], 256, 1);

figure;
h = histogram(normal_score(nor_rdn_idx), 100);
hold on
histogram(anomaly_score(ano_rdn_idx), h.BinEdges);

figure;
h = histogram(normal_z1(nor_rdn_idx), 100);
hold on
histogram(anomaly_z1(ano_rdn_idx), h.BinEdges);

figure;
h = histogram(normal_z1_2nd(nor_rdn_idx), 100);
hold on
histogram(anomaly_z1_2nd(ano_rdn_idx), h.BinEdges);

P = length(normal_score);
N = length(anomaly_score);
ROC_TPR = [];
ROC_FPR = [];
AUC = 0;
for i = 0:0.001:max(anomaly_score)
    
    scroe_thres = i;
    
    TP = sum(normal_score < scroe_thres);
    FN = sum(normal_score >= scroe_thres);
    FP = sum(anomaly_score < scroe_thres);
    TN = sum(anomaly_score >= scroe_thres);

    Acc = (TP + TN) / (P + N);
    Recall = TP / P;
    Precision = TP / (TP + FP);
    F1 = 2*TP / (2*TP + FN + FP);    
    
    ROC_TPR = [ROC_TPR; TP/(TP+FN)];
    ROC_FPR = [ROC_FPR; FP/(FP+TN)];
    
    if i > 0
        AUC = AUC + (ROC_TPR(end)+ROC_TPR(end-1))*(ROC_FPR(end)-ROC_FPR(end-1))/2; 
    end
end

figure;
plot(ROC_FPR, ROC_TPR)

fprintf("AUC: %f\n", AUC);
