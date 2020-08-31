clc; clear; close all;

%dataset = 'SVHN';
%model_name = 'AD_CLS_DISE_v8_50MSE_25CL_1DL_1D_retrain';
%lat_dim = 256;

%dataset = 'Cifar-10';
%model_name = 'AD_VAE_DISE3_v1_50L1_10CL_1DL_1D_256DIM_SVHN_ano2_VAE_drop025_1StageTrain_1e1_SingleDis_znCode2_AUG';
%lat_dim = 256;

dataset = 'MNIST';
model_name = 'AD_VAE_DISE3_v1_50L1_10CL_1DL_1D_256DIM_MNIST_ano2_VAE_drop025_1StageTrain_1e1_SingleDis_znCode2_AUG';
lat_dim = 64;

normal_class_num = 10;
anomaly_class_idx = 2;

fprintf(['Anomaly Class:' num2str(anomaly_class_idx) '\n']);

normal_score = [];
normal_z1 = [];
normal_z1_vector = [];
normal_z2_vector = [];
normal_content_loss = [];

anomaly_score = [];
anomaly_z1 = [];
anomaly_z1_vector = [];
anomaly_z2_vector = [];
anomaly_content_loss = [];

for i=1:normal_class_num
    curr_data = csvread([ 'T:/data/wei/dataset/FSL/' dataset '/FSL/' model_name '/pr_test_class_' num2str(i-1) ...
                          '/pr_test_class_' num2str(i-1) '_ano_score.csv']);    
    if (i-1) ~= anomaly_class_idx                               
        normal_score = [normal_score; curr_data(:, 1)];
        normal_z1 = [normal_z1; max(curr_data(:, 2:11), [], 2)];       
        normal_z1_vector = [normal_z1_vector; curr_data(:, 2:11)];       
        normal_z2_vector = [normal_z2_vector; curr_data(:, 12:12+lat_dim-1)];
        normal_content_loss = [normal_content_loss; curr_data(:, end)];
    else
        anomaly_score = [anomaly_score; curr_data(:, 1)];
        anomaly_z1 = [anomaly_z1; max(curr_data(:, 2:11), [], 2)];       
        anomaly_z1_vector = [anomaly_z1_vector; curr_data(:, 2:11)];       
        anomaly_z2_vector = [anomaly_z2_vector; curr_data(:, 12:12+lat_dim-1)];
        anomaly_content_loss = [anomaly_content_loss; curr_data(:, end)];
    end
end

% -------------------------------------------------------------------------
% normal_c = repmat([0 0.4470 0.7410], 500, 1);
% anomaly_c = repmat([0.6350 0.0780 0.1840], 500, 1);
% c = [normal_c; anomaly_c];
% 
% fprintf('c t-sne...\n');
% z1 = [normal_z1_vector(1:500, :); anomaly_z1_vector(1:500, :)];
% Y1 = tsne(z1,'Algorithm','exact','NumDimensions',3, 'Distance','euclidean');
% figure; scatter3(Y1(:,1), Y1(:,2), Y1(:,3), 30, c, 'filled', 'MarkerEdgeColor', [1,1,1]);
% title('c vector');
% 
% fprintf('z t-sne...\n');
% z2 = [normal_z2_vector(1:500, :); anomaly_z2_vector(1:500, :)];
% Y2 = tsne(z2,'Algorithm','exact','NumDimensions',3);
% figure; scatter3(Y2(:,1), Y2(:,2), Y2(:,3), 30, c, 'filled', 'MarkerEdgeColor', [1,1,1]);
% title('z vector');
% -------------------------------------------------------------------------

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

% figure;
% h = histogram(normal_chi2_z1(nor_rdn_idx), 100);
% hold on
% histogram(anomaly_chi2_z1(ano_rdn_idx), h.BinEdges);
% title("chi2 z1");

figure;
h = histogram(normal_content_loss(nor_rdn_idx), 100);
hold on
histogram(anomaly_content_loss(ano_rdn_idx), h.BinEdges);
title("Content loss");

fprintf('AUC...\n');
%lamda = 0.78;
%lamda = 0;
best_AUC= 0;
interval = 0.001;
for lamda = 0:0.01:1

    P = length(normal_score);
    N = length(anomaly_score);
    ROC_TPR = zeros(int32(1/interval+1), 1);
    ROC_FPR = zeros(int32(1/interval+1), 1);
    AUC = 0;
    count = 0;
    
    normal_target = (1-lamda)*(1./normal_content_loss) + lamda*normal_z1;
    anomaly_target = (1-lamda)*(1./anomaly_content_loss) + lamda*anomaly_z1;

    min_val = min(min(normal_target), min(anomaly_target));
    max_val = max(max(normal_target), max(anomaly_target));
    normal_target = (normal_target - min_val) ./ (max_val - min_val);
    anomaly_target = (anomaly_target - min_val) ./ (max_val - min_val);

    for i = min(min(normal_target), min(anomaly_target)):interval:max(max(normal_target), max(anomaly_target))

        scroe_thres = i;

        TP = sum(normal_target > scroe_thres);
        FN = sum(normal_target <= scroe_thres);
        FP = sum(anomaly_target > scroe_thres);
        TN = sum(anomaly_target <= scroe_thres);

        %Acc = (TP + TN) / (P + N);
        %Recall = TP / P;
        %Precision = TP / (TP + FP);
        %F1 = 2*TP / (2*TP + FN + FP);    

        ROC_TPR(count+1) = TP/(TP+FN);
        ROC_FPR(count+1) = FP/(FP+TN);

        if count > 0
            AUC = AUC + (ROC_TPR(count+1)+ROC_TPR(count))*abs(ROC_FPR(count+1)-ROC_FPR(count))/2; 
        end
        count = count + 1;
        
    end
    
    if best_AUC < AUC
        best_AUC = AUC;
        best_ROC_TPR = ROC_TPR;
        best_ROC_FPR = ROC_FPR;
        best_lamda = lamda;
    end
    
end

fprintf("lamda: %f\n", best_lamda);

normal_target = (1-best_lamda)*(1./normal_content_loss) + best_lamda*normal_z1;
anomaly_target = (1-best_lamda)*(1./anomaly_content_loss) + best_lamda*anomaly_z1;

min_val = min(min(normal_target), min(anomaly_target));
max_val = max(max(normal_target), max(anomaly_target));
normal_target = (normal_target - min_val) ./ (max_val - min_val);
anomaly_target = (anomaly_target - min_val) ./ (max_val - min_val);

figure;
h = histogram(normal_target(nor_rdn_idx), 100);
hold on
histogram(anomaly_target(ano_rdn_idx), h.BinEdges);
title("target");

figure;
plot(best_ROC_FPR, best_ROC_TPR)
title("ROC");

fprintf("AUC: %f\n", best_AUC);

% -------------------------------------------------------------------------
function value = chi2(O, E)

    value = sum(((O-E)./(E+0.000001)).^2, 2);

end