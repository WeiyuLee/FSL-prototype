clc; clear; close all;

dataset = 'SVHN';
model_name = 'AD_VAE_BASELINE_v1_50L1_1e1KL_SVHN';

%dataset = 'Cifar-10';
%model_name = 'AD_CLS_BASELINE_v1_ano2';

normal_class_num = 10;
anomaly_class_idx = 2;

fprintf(['Anomaly Class:' num2str(anomaly_class_idx) '\n']);

normal_content_loss = [];
normal_z = [];
normal_z_vector = [];

anomaly_content_loss = [];
anomaly_z = [];
anomaly_z_vector = [];
for i=1:normal_class_num
    curr_data = csvread([ 'T:/data/wei/dataset/FSL/' dataset '/FSL/' model_name '/pr_test_class_' num2str(i-1) ...
                          '/pr_test_class_' num2str(i-1) '_ano_score.csv']);    
    if (i-1) ~= anomaly_class_idx                               
        normal_z = [normal_z; max(curr_data(:, 1:256), [], 2)];
        normal_z_vector = [normal_z_vector; curr_data(:, 1:256)];
    else
        anomaly_z = [anomaly_z; max(curr_data(:, 1:256), [], 2)];
        anomaly_z_vector = [anomaly_z_vector; curr_data(:, 1:256)];
    end
end

% -------------------------------------------------------------------------
normal_c = repmat([0 0.4470 0.7410], 500, 1);
anomaly_c = repmat([0.6350 0.0780 0.1840], 500, 1);
c = [normal_c; anomaly_c];

fprintf('z t-sne...\n');
z1 = [normal_z_vector(1:500, :); anomaly_z_vector(1:500, :)];
Y1 = tsne(z1,'Algorithm','exact','NumDimensions',3, 'Distance','euclidean');
figure; scatter3(Y1(:,1), Y1(:,2), Y1(:,3), 30, c, 'filled', 'MarkerEdgeColor', [1,1,1]);
%title('z vector');
% -------------------------------------------------------------------------

nor_rdn_idx = randi([1 length(normal_z)], 256, 1);
ano_rdn_idx = randi([1 length(anomaly_z)], 256, 1);

figure;
h = histogram(normal_z(nor_rdn_idx), 100);
hold on
histogram(anomaly_z(ano_rdn_idx), h.BinEdges);
title("z1");

fprintf('AUC...\n');
P = length(normal_z);
N = length(anomaly_z);
ROC_TPR = [];
ROC_FPR = [];
AUC = 0;
for i = 0:0.001:max(max(normal_z), max(anomaly_z))
    
    score_thres = i;
    
    TP = sum(normal_z >= score_thres);
    FN = sum(normal_z < score_thres);
    FP = sum(anomaly_z >= score_thres);
    TN = sum(anomaly_z < score_thres);

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
