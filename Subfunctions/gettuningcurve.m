function [tuning_curve_sets, NNS_set] = gettuningcurve(iter, image_sets_standard, image_sets_control1, image_sets_control2)
net = alexnet; % analyzeNetwork(net)
isANOVA1 = 1; 
rand_layers_ind = [2, 6, 10, 12 14];
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
layers_set = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};
tuning_curve_sets = zeros(iter, length(number_sets),length(number_sets));
NNS_set = zeros(1,iter);
for iterind = 1:iter
disp(iterind)
net_rand_perm = Randomizeweight_permute(net, rand_layers_ind);

%% Step 3. define layer of interest and calculate response
LOI = layers_set{5}; %  interested in the final layer
response_tot_standard_RP = getactivation(net_rand_perm, LOI, image_sets_standard);
response_tot_control1_RP = getactivation(net_rand_perm, LOI, image_sets_control1);
response_tot_control2_RP = getactivation(net_rand_perm, LOI, image_sets_control2);
% get total response matrix
response_tot_RP = cat(2,response_tot_standard_RP, response_tot_control1_RP, response_tot_control2_RP);

%% Step 4. get p-values from response
pvalues_RP = getpv(response_tot_RP);
pvalues2_RP = getpvforeach(response_tot_RP);

%% Step 5. Analyze p-values to find number selective neurons
% pv1 = pvalues_RP(1,:); pv2 = pvalues_RP(2,:);pv3 = pvalues_RP(3,:); 
if isANOVA1
pv1 = pvalues_RP(1,:); pv2 = pvalues_RP(2,:);pv3 = pvalues_RP(3,:); 
p_th = 0.01;p_th2 = 0.01;p_th3 = 0.01;
pv5 = pvalues2_RP(1,:);pv6 = pvalues2_RP(3,:);
pv4 = pvalues2_RP(2,:); 
ind1 = (pv1<p_th);
ind2 = (pv2>p_th2);
ind3 = (pv3>p_th2);
ind4 = pv4<p_th3; % 1-way anova for area
ind5 = pv5<p_th3;ind6 = pv6<p_th3;
ind_NS = find(ind1.*ind2.*ind3.*ind4.*ind5.*ind6); % indices of number selective units
else
p_th = 0.01;p_th2 = 0.01;
ind1 = (pv1<p_th);
ind2 = (pv2>p_th2);
ind3 = (pv3>p_th2);
ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
end
NNS_pretrained_RP = length(ind_NS); % number of NS for pretrained net ~ 1499

%% Step 6. get tuning curves
resp_mean_RP = squeeze((mean(response_tot_RP, 2))); resp_std_RP = squeeze(std(response_tot_RP, 0,2));
resp_mean_standard_RP = squeeze((mean(response_tot_standard_RP, 2))); resp_std_standard = squeeze(std(response_tot_standard_RP, 0,2));
resp_mean_control1_RP = squeeze((mean(response_tot_control1_RP, 2))); resp_std_control1 = squeeze(std(response_tot_control1_RP, 0,2));
resp_mean_control2_RP = squeeze((mean(response_tot_control2_RP, 2))); resp_std_control2 = squeeze(std(response_tot_control2_RP, 0,2));
% indtmp = randi(length(ind_NS));
% figure;hold on;
% errorbar(number_sets, resp_mean(:,ind_NS(indtmp)), resp_std(:,ind_NS(indtmp))/sqrt(image_iter), 'k' )
% errorbar(number_sets, resp_mean_standard(:,ind_NS(indtmp)), resp_std_standard(:,ind_NS(indtmp))/sqrt(image_iter), 'b' )
% errorbar(number_sets, resp_mean_control1(:,ind_NS(indtmp)), resp_std_control1(:,ind_NS(indtmp))/sqrt(image_iter), 'r' )
% errorbar(number_sets, resp_mean_control2(:,ind_NS(indtmp)), resp_std_control2(:,ind_NS(indtmp))/sqrt(image_iter), 'g')
% legend({'tot','std','ctr1', 'ctr2'});% scatter(number_sets, resp_mean(:,randi(43264)))

%% Step 8. get preferred number
response_NS_tot_RP = response_tot_RP(:,:,ind_NS);
response_NS_mean_RP = squeeze(mean(response_NS_tot_RP, 2));
[M,PNind_RP] = max(response_NS_mean_RP);


%% Figure S1. average tuning curves for permuted net
tuning_curve_RP = zeros(length(number_sets),length(number_sets));
tuning_curve_std_RP = zeros(length(number_sets),length(number_sets));
for ii = 1:length(number_sets)
    tuning_curvetmp = response_NS_mean_RP(:,find(PNind_RP==ii));
    cvtmp = mean(tuning_curvetmp, 2);
    cvstdtmp = std(tuning_curvetmp, [],2)/sqrt(size(tuning_curvetmp,2));
    
    atmp = 1/(max(cvtmp)-min(cvtmp));
    btmp = -min(cvtmp)/(max(cvtmp)-min(cvtmp));
    % tuning_curve(ii,:) = rescale(cvtmp);
    tuning_curve_RP(ii,:) = atmp*cvtmp+btmp;
    tuning_curve_std_RP(ii,:)  = atmp*cvstdtmp+btmp;

end
NNS_set(iterind) = length(ind_NS);
tuning_curve_sets(iterind, :, :) = tuning_curve_RP;
end


end