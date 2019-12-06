clc; clear; close all;

%dataset = 'SVHN';
%model_name = 'AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_retrain';

dataset = 'Cifar-10';
model_name = 'AD_CLS_BASELINE_v1_ano2';

normal_class_num = 10;
anomaly_class_idx = 2;

fprintf(['Anomaly Class:' num2str(anomaly_class_idx) '\n']);

normal_score = [];
normal_z1 = [];
normal_z1_2nd = [];
anomaly_score = [];
anomaly_z1 = [];
anomaly_z1_2nd = [];
for i=1:normal_class_num
    curr_data = csvread([ 'T:/data/wei/dataset/FSL/' dataset '/FSL/' model_name '/pr_test_class_' num2str(i-1) ...
                          '/pr_test_class_' num2str(i-1) '_ano_score.csv']);    
    if (i-1) ~= anomaly_class_idx                               
        normal_z1 = [normal_z1; max(curr_data(:, 1:10), [], 2)];
    else
        anomaly_z1 = [anomaly_z1; max(curr_data(:, 1:10), [], 2)];
    end
end

nor_rdn_idx = randi([1 length(normal_z1)], 256, 1);
ano_rdn_idx = randi([1 length(anomaly_z1)], 256, 1);

figure;
h = histogram(normal_z1(nor_rdn_idx), 100);
hold on
histogram(anomaly_z1(ano_rdn_idx), h.BinEdges);
title("z1");

P = length(normal_z1);
N = length(anomaly_z1);
ROC_TPR = [];
ROC_FPR = [];
AUC = 0;
for i = 0:0.001:max(max(normal_z1), max(anomaly_z1))
    
    score_thres = i;
    
    TP = sum(normal_z1 >= score_thres);
    FN = sum(normal_z1 < score_thres);
    FP = sum(anomaly_z1 >= score_thres);
    TN = sum(anomaly_z1 < score_thres);

    Acc = (TP + TN) / (P + N);
    Recall = TP / P;
    Precision = TP / (TP + FP);
    F1 = 2*TP / (2*TP + FN + FP);    
    
    ROC_TPR = [ROC_TPR; TP/(TP+FN)];
    ROC_FPR = [ROC_FPR; FP/(FP+TN)];
    
    if i > 0
        AUC = AUC + (ROC_TPR(end)+ROC_TPR(end-1))*(ROC_FPR(end-1)-ROC_FPR(end))/2; 
    end
end

figure;
plot(ROC_FPR, ROC_TPR)
title("ROC");

fprintf("AUC: %f\n", AUC);
