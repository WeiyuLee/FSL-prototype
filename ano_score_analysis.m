clc; clear; close all;

%dataset = 'SVHN';
%model_name = 'AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_retrain';

dataset = 'Cifar-10';
model_name = 'AD_CLS_DISE2_v1_100L1_25CL_1DLL1_1D_1D2_128DIM_cifar_ano9_zt';

normal_class_num = 10;
anomaly_class_idx = 9;

fprintf(['Anomaly Class:' num2str(anomaly_class_idx) '\n']);

normal_score = [];
normal_score2 = [];
normal_z1 = [];
normal_z1_2nd = [];
normal_content_loss = [];
anomaly_score = [];
anomaly_score2 = [];
anomaly_z1 = [];
anomaly_z1_2nd = [];
anomaly_content_loss = [];
for i=1:normal_class_num
    curr_data = csvread([ 'T:/data/wei/dataset/FSL/' dataset '/FSL/' model_name '/pr_test_class_' num2str(i-1) ...
                          '/pr_test_class_' num2str(i-1) '_ano_score.csv']);    
    if (i-1) ~= anomaly_class_idx                               
        normal_score = [normal_score; curr_data(:, 1)];
        %normal_score2 = [normal_score; sum((curr_data(:, 2:11) - curr_data(:, 12:21)).^2, 2)];
        normal_z1 = [normal_z1; max(curr_data(:, 2:11), [], 2)];
        normal_z1_2nd = [normal_z1_2nd; max(curr_data(:, 12:21), [], 2)];
        %normal_content_loss = [normal_content_loss; curr_data(:, 22)];
    else
        anomaly_score = [anomaly_score; curr_data(:, 1)];
        %anomaly_score2 = [anomaly_score2; sum((curr_data(:, 2:11) - curr_data(:, 12:21)).^2, 2)];
        anomaly_z1 = [anomaly_z1; max(curr_data(:, 2:11), [], 2)];
        anomaly_z1_2nd = [anomaly_z1_2nd; max(curr_data(:, 12:21), [], 2)];
        %anomaly_content_loss = [anomaly_content_loss; curr_data(:, 22)];
    end
end

nor_rdn_idx = randi([1 length(normal_score)], 256, 1);
ano_rdn_idx = randi([1 length(anomaly_score)], 256, 1);

figure;
h = histogram(normal_score(nor_rdn_idx), 100);
hold on
histogram(anomaly_score(ano_rdn_idx), h.BinEdges);
title("Anomaly Score");

% figure;
% h = histogram(normal_score2(nor_rdn_idx), 100);
% hold on
% histogram(anomaly_score2(ano_rdn_idx), h.BinEdges);
% title("Anomaly Score2");

figure;
h = histogram(normal_z1(nor_rdn_idx), 100);
hold on
histogram(anomaly_z1(ano_rdn_idx), h.BinEdges);
title("z1");

figure;
h = histogram(normal_z1_2nd(nor_rdn_idx), 100);
hold on
histogram(anomaly_z1_2nd(ano_rdn_idx), h.BinEdges);
title("z1 2nd");

% figure;
% h = histogram(normal_content_loss(nor_rdn_idx), 100);
% hold on
% histogram(anomaly_content_loss(ano_rdn_idx), h.BinEdges);
% title("Content loss");

P = length(normal_score);
N = length(anomaly_score);
ROC_TPR = [];
ROC_FPR = [];
AUC = 0;
count = 0;

lamda = 0;
normal_target = (1-lamda)*normal_score + lamda*normal_z1;
anomaly_target = (1-lamda)*anomaly_score + lamda*anomaly_z1;

min_val = min(min(normal_target), min(anomaly_target));
max_val = max(max(normal_target), max(anomaly_target));
normal_target = (normal_target - min_val) / (max_val - min_val);
anomaly_target = (anomaly_target - min_val) / (max_val - min_val);

figure;
h = histogram(normal_target(nor_rdn_idx), 100);
hold on
histogram(anomaly_target(ano_rdn_idx), h.BinEdges);
title("target");

for i = min(min(normal_target), min(anomaly_target)):0.00001:max(max(normal_target), max(anomaly_target))
    
    scroe_thres = i;
        
    TP = sum(normal_target < scroe_thres);
    FN = sum(normal_target >= scroe_thres);
    FP = sum(anomaly_target < scroe_thres);
    TN = sum(anomaly_target >= scroe_thres);

    Acc = (TP + TN) / (P + N);
    Recall = TP / P;
    Precision = TP / (TP + FP);
    F1 = 2*TP / (2*TP + FN + FP);    
    
    ROC_TPR = [ROC_TPR; TP/(TP+FN)];
    ROC_FPR = [ROC_FPR; FP/(FP+TN)];
    
    if count > 0
        AUC = AUC + (ROC_TPR(end)+ROC_TPR(end-1))*abs(ROC_FPR(end)-ROC_FPR(end-1))/2; 
    end
    count = count + 1 ;
end

figure;
plot(ROC_FPR, ROC_TPR)
title("ROC");

fprintf("AUC: %f\n", AUC);
