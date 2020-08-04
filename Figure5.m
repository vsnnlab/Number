% =========================================================================================================================================================
% Demo codes for
% "Spontaneous generation of number sense in untrained deep neural networks"
% Gwangsu Kim, Jaeson Jang, Seungdae Baek, Min Song, and Se-Bum Paik*
%
% *Contact: sbpaik@kaist.ac.kr
%
% Prerequirements
% 1) MATLAB 2018b or later version
% 2) Installation of the Deep Learning Toolbox (https://www.mathworks.com/products/deep-learning.html)
% 3) Installation of the pretrained AlexNet (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)

% This code performs a demo simulation for Fig. 5 in the manuscript.
% =========================================================================================================================================================


close all;clc;clear;

toolbox_chk

disp('Demo codes for Figure 5 of "Spontaneous generation of number sense in untrained deep neural networks"')
disp(' ')
disp('* It performs a demo version (a fewer set of stimuli than in the paper) of simulation using a single condition of the network.')
disp('  (# images for each condition: 50 -> 10 , # repetition of simulation: 100 (or 1000 for summation model) -> 2)')
disp('* Expected running time is about 5 minutes, but may vary by system conditions.')
disp(' ')

%% Figure 5

%% Setting options for simulation
generatedat = 1;
iter = 2;   % Number of randomized networks for analysis
issavefig = 1;  % Save fig files?


%% Setting file dir
pathtmp = pwd; % Setting current dir.
% mkdir ([pathtmp '/Figs']); % Generating folder to save fig. files
addpath(genpath(pathtmp));

%% Setting parameters

rand_layers_ind = [2, 6, 10, 12 14];    % Index of convolutional layer of AlexNet
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]; % Candidiate numerosities of stimulus
p_th1 = 0.01; p_th2 = 0.01; p_th3 = 0.01;  % Significance levels for two-way ANOVA
image_iter = 10;  % Number of images for a given condition
layer_investigate = [4,5]; % Indices of convolutional layer to investigate
layer_last = layer_investigate(2);   %
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256]; % Size of Alexnet
layers_name = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};
LOI = 'relu4'; % Name of layer at which the activation will be measured
L4investigatePN = [1 16]; % Investigated units in Conv4 (1:decreasing units / 16:increasing units defined by criteria used in Stoianov & Zorzi 2012) 
L5investigatePN = 1:16; % Investigated PN in Conv5

egN = 10; % # of inc/dec units for visualization
nametmp = ['Demo']; % Suffix

%% Loading pretrained network
net = alexnet; % Loading network from deep learning toolbox
% analyzenetwork(net); % Checking the network architecture

% Network initialization condition
stdfacs = [1]; % Factor multiplied to weight (variation of weight)
vercand = [1]; %1:he normal, 2:lecun normal, 3:he uniform, 4:lecun uniform

if generatedat
    %% Generating stimulus set
    
    disp(['[1/4] Generating stimulus set...'])
    % Stimulus set (Nasr 2019)
    [image_sets_standard, image_sets_control1, image_sets_control2, polyxy]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    % Area-varying set (Stoianov & zorzi 2012)
    radius_cand = 2*sqrt(1:8);
    [image_sets_zorzi, image_sets_zorzi_stat] = Stimulus_generation_Zorzi2(number_sets, image_iter, radius_cand);
    
    %% Searching increasing/decreasing units in the network
    disp('[2/4] Searching increasing/decreasing units in the network...')
    itereg = 1; %
    iter_resp_mean_n = cell(1,1,itereg);
    iter_units_PN = cell(1,1,itereg);
    iter_units_areanumcoef = cell(1,1,itereg);
    iter_units_R2s = cell(1,1,itereg);
    
    for iterind = 1:itereg
        for stdfacind = 1:length(stdfacs)
            
            stdfac = stdfacs(stdfacind);
            for verind = 1:length(vercand)
                ver = vercand(verind);
                
                [net_init, ~, networkweights, ~] = Initializeweight_he2(net, rand_layers_ind, ver, stdfac); % Initialized network
                net_test = net_init;
                
                [units_PN_rand, resp_mean, ~, ~, ~, ~] = ...
                    getNumberSensefromNet3(net_test, image_sets_standard, image_sets_control1,image_sets_control2...
                    , p_th1, p_th2, p_th3, LOI, number_sets, 0, 1);
                
                iter_resp_mean_n{stdfacind, verind,iterind} = resp_mean;
                iter_units_PN{stdfacind, verind,iterind} = units_PN_rand;
                response_tot_zorzi = getactivation(net_test, LOI, image_sets_zorzi);
                
                image_sets_zorzi_areas = squeeze(image_sets_zorzi_stat(1,:,:))';
                image_sets_zorzi_nums = [];
                for ii = 1:16
                    tmp = ii*ones(1,size(image_sets_zorzi_areas,2));
                    image_sets_zorzi_nums = cat(1, image_sets_zorzi_nums, number_sets(tmp));
                end
                
                areastmp = rescale(log((image_sets_zorzi_areas(:))));
                numstmp = rescale(log((image_sets_zorzi_nums(:))));
                
                Coeffs = zeros(3,size(response_tot_zorzi, 3));
                R2s = zeros(1,size(response_tot_zorzi, 3));
                
                for ii = 1:size(response_tot_zorzi, 3)
                    resptmp = squeeze(response_tot_zorzi(:,:,ii));
                    resptmp = (rescale(resptmp(:)));
                    X = [ones(size(areastmp)) areastmp numstmp];
                    [b,~,~,~,stats ] = regress(resptmp, X);
                    
                    R2s(ii) = stats(1);
                    Coeffs(:,ii) = b;
                end
                
                areacoef = Coeffs(2,:);
                numcoef = Coeffs(3,:);
                units_areacoef = areacoef;
                units_numcoef = numcoef;
                
                iter_units_areanumcoef{iterind} = units_areacoef+1i*units_numcoef;
                iter_units_R2s{iterind} = R2s;
                
            end
            
        end
    end
else
end

%% Generating/analyzing backtracking data

if generatedat
    disp('[3/4] Generating backtracking data...')
    generateBacktrackingdata_zorziincdec(generatedat,rand_layers_ind...
        , number_sets, iter, net, layers_name, pathtmp, layer_investigate, ...
        p_th1, p_th2, p_th3, layer_last, array_sz, ...
        image_sets_standard, image_sets_control1, image_sets_control2, image_sets_zorzi, image_sets_zorzi_stat, nametmp)
    
    %% Analyzing backtracking data
    disp('[4/4] Analyzing backtracking data')
    
    [weightdist_foreachL5PN_tot, weightdist_control_tot, incdecportion_foreachL5PN_tot...
        , weightdist_foreachL5selective_tot, incdecportion_foreachL5selective_tot...
        , weightdist_foreachL5Nonselective_tot, incdecportion_foreachL5Nonselective_tot, ...
        incdecportion_foreachL5_tot] = analyzeBacktrackingdata(iter,  ...
        pathtmp, L4investigatePN, L5investigatePN, nametmp);
else
%    load([pathtmp '/Dataset/Data/Results_Backtracking'])
end

%% Calculating weights of increasing/decreasing units (in L4) connected to L5 neurons
if generatedat
    meantmp = 0; % Mean of weight
    [connectedweights, control1s, NS_wmean, NNS_wmean] = ...
        analyzeConnections(meantmp, L5investigatePN, L4investigatePN...
        , weightdist_foreachL5PN_tot, weightdist_control_tot, ...
        weightdist_foreachL5Nonselective_tot, iter, 0);
else
%    load([pathtmp '/Dataset/Data/Results_weightbias'])
end


%% --------------------------------------------------- Summation model

%% Setting parameters
% Sigma of lognormal distribution sampled from the acitivity of decreasing and increasing units in the untrained AlexNet
log_sig_dec_mean = 1.94;    % Mean of sigma sampled from decreasing units
log_sig_dec_std = 1.5;     % STD of sigma sampled from decreasing units
log_sig_inc_mean = 2.19;    % for increasing units
log_sig_inc_std = 1.5;

% Distribution of convolutional weights
mean_weight = 0;    % Mean of weights
std_weight = 0.0065;    % STD of weights

% List of tested numerosities
num_list = [1 2:2:30];
num_list_log = log2(num_list);
num_list_ext = [num_list, 0];

% Network specification
nunit = 60;                     % Number of decreasing and increasing units
nneuron = 10000;                 % Number of output neurons
ntrial = 2; 1000;               % Number of simulations

bb_weight = linspace(-0.03, 0.03, 61);    % Bins for histogram of weights
bb_sigma = linspace(-10, 20, 101);

xx_axis = -5:35;
ncon = 100;

color_dec = [237 30 121]/255;
color_inc = [0 113 188]/255;
color_linear = [241 90 36]/255;
color_log = [0 146 69]/255;

load('rnb.mat', 'rnb')

%% Variables for saving simulation results
w_dec_save = zeros(ntrial, nunit, nneuron);
w_inc_save = zeros(ntrial, nunit, nneuron);

tun_dec = zeros(ntrial, length(num_list), nunit);    % Tuning curves of each unit
tun_inc = zeros(ntrial, length(num_list), nunit);

tun_all = zeros(length(num_list), ntrial, length(num_list));
tun_all_save = zeros(ntrial, length(num_list), nneuron);
pn_all_save = zeros(ntrial, nneuron);

pn_all = zeros(ntrial, length(num_list));
sigma_all = zeros(ntrial, length(num_list));
sigma_log_all = zeros(ntrial, length(num_list));

w_dec_mean = zeros(ntrial, length(num_list));
w_inc_mean = zeros(ntrial, length(num_list));

sigma_Weber = zeros(size(num_list));
sigma_Weber_log = zeros(size(num_list));

%% Generating tuning curves
log_sig_dec = randn(nunit, ntrial)*log_sig_dec_std + log_sig_dec_mean;  % Randomly generated sigma of tuning curves
log_sig_inc = randn(nunit, ntrial)*log_sig_inc_std + log_sig_inc_mean;

for pp = 1:ntrial
    for ii = 1:nunit
        tun_dec(pp, :, ii) = exp(-((num_list_log-log2(1)).^2)/(2*(log_sig_dec(ii, pp)).^2));   % Log-normal
        tun_inc(pp, :, ii) = exp(-((num_list_log-log2(30)).^2)/(2*(log_sig_inc(ii, pp)).^2));
    end
end

%% Weighted summation
for pp = 1:ntrial
    w_dec = zeros(nunit, nneuron);
    w_inc = zeros(nunit, nneuron);
    
    tun_map = zeros(length(num_list), nneuron);
    pn_map = zeros(nneuron, 1);
    sigma_map = zeros(nneuron, 1);
    sigma_log_map = zeros(nneuron, 1);
    
    for ii = 1:nneuron
        w_dec(:, ii) = mean_weight + randn(nunit, 1)*std_weight;
        w_inc(:, ii) = mean_weight + randn(nunit, 1)*std_weight;
    end
    
    res_1_temp = squeeze(tun_dec(pp,:,:))*squeeze(w_dec(:, :));
    res_30_temp = squeeze(tun_inc(pp,:,:))*squeeze(w_inc(:, :));
    
    res_temp =  res_1_temp + res_30_temp;
    
    res_temp(res_temp<0) = 0;
    
    tun_map(:, :) = res_temp;
    [val_temp, ind_temp] = max(res_temp);
    ind_temp(sum(res_temp) == 0) = length(num_list_ext);
    pn_map(:) = num_list_ext(ind_temp);
    
    bb = [0 1 2:2:30];
    pn_dist = zeros(length(num_list), 1);
    for ii = 1:length(num_list)
        pn_dist(ii) = sum(pn_map(:)==num_list(ii));
    end
    pn_dist = pn_dist/sum(pn_dist);
    pn_all(pp, :) = pn_dist;
    
    for ii = 1:length(num_list)
        tun_all(ii, pp, :) = mean(tun_map(:, pn_map == num_list(ii)), 2);
    end
    
    tun_all_save(pp, :, :) = tun_map;
    pn_all_save(pp, :) = pn_map;
    
    %% Average tuning curves
    tun_temp = reshape(tun_map, [length(num_list), nneuron]);
    [max_val, max_ind] = max(tun_temp, [], 1);
    max_ind(max_val==0) = nan;
    
    sigma_norm = nan(length(num_list), 1);
    sigma_norm_log = nan(length(num_list), 1);
    
    xtmp = num_list;
    xtmp_log = num_list_log;
    options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
    
    for ii = 1:length(num_list)
        sel_ind = find(max_ind == ii);
        temp = nanmean(tun_temp(:, sel_ind), 2);
        temp = (temp-min(temp))/(max(temp)-min(temp));
        
        xtmp = num_list;
        if ~isnan(temp)
            [f, gof] = fit(xtmp.', temp, 'gauss1', options);
            
            sigma_norm(ii) = f.c1/sqrt(2);
            
            xtmp = num_list_log;
            [f_log, gof_log] = fit(xtmp.', temp, 'gauss1', options);
            
            sigma_norm_log(ii) = f_log.c1/sqrt(2);
        end
    end
    
    sigma_all(pp, :) = sigma_norm;
    sigma_log_all(pp, :) = sigma_norm_log;
    
    %% Weight bias
    for ii = 1:length(num_list)
        w_dec_temp = (w_dec(:, (pn_map == num_list(ii))) - mean(w_dec(:)))/std(w_dec(:));
        w_inc_temp = (w_inc(:, (pn_map == num_list(ii))) - mean(w_inc(:)))/std(w_inc(:));
        
        w_dec_mean(pp, ii) = mean(w_dec_temp(:));
        w_inc_mean(pp, ii) = mean(w_inc_temp(:));
    end
    
    w_dec_save(pp, :, :) = w_dec;
    w_inc_save(pp, :, :) = w_inc;
end



%% Plotting results
trial_ind = 1;
num_shown_tun = 20;

tun_single_trial = squeeze(tun_all_save(trial_ind, :, :));
pn_single_trial = squeeze(pn_all_save(trial_ind, :));

figure('units','normalized','outerposition',[0 0.25 1 0.5])
% Tuning curves of decreasing units in Conv4 of the untrained AlexNet
subplot(2,6,1); hold on;
title('Decreasing units in Conv4')

% Tuning curves of increasing units in Conv4 of the untrained AlexNet
subplot(2,6,7); hold on;
title('Increasing units in Conv4')

% Examples of decreasing tuning curves
subplot(2,6,2); hold on;
plot(squeeze(tun_dec(1,:,randperm(nunit, num_shown_tun))), 'color', color_dec);
title('Model curves for decreasing activities')
ylabel('Response (A.U.)')
set(gca, 'xtick', [1 6 11 16], 'xticklabel', [1 10 20 30], 'ytick', [0 1])

% Examples of increasing tuning curves
subplot(2,6,8); hold on;
plot(squeeze(tun_inc(1,:,randperm(nunit, num_shown_tun))), 'color', color_inc);
title('Model curves for increasing activities')
xlabel('Numerosity')
ylabel('Response (A.U.)')
set(gca, 'xtick', [1 6 11 16], 'xticklabel', [1 10 20 30], 'ytick', [0 1])

% Example tuning curves
subplot_ind = [5:8, 17:20, 29:32, 41:44];
xtick_ind = [41:44];
ytick_ind = [5 17 29 41];
for ii = 1:length(num_list)
    tun_each_num = tun_single_trial(:, pn_single_trial==num_list(ii));
    norm_tun_each_num = tun_each_num./repmat(max(tun_each_num), [length(num_list), 1]);
    
    subplot(4,12,subplot_ind(ii)); hold on;
    %     plot(norm_tun_each_num, 'color', [0.8 0.8 0.8 0.5])
    plot(mean(norm_tun_each_num,2), 'k')
    set(gca, 'xtick', [], 'ytick', [])
    if ismember(subplot_ind(ii), xtick_ind); set(gca, 'xtick', [1 6 11 16], 'xticklabel', [1 10 20 30]); end
    if ismember(subplot_ind(ii), ytick_ind); set(gca, 'ytick', [0 1]); end
    if ismember(subplot_ind(ii), xtick_ind) && ismember(subplot_ind(ii), ytick_ind); xlabel('Numerosity'); ylabel('Normalized response'); end
    title(['PN = ' num2str(num_list(ii))])
    if ii == 1; title(['Weighted summation: PN = ' num2str(num_list(ii))]); end
end

%% Distribution of preferred numerosity
subplot(2,6,5); hold on;
load('PN_monkey')   % Loding monkey data from Nieder et al., 2007

pn_dist = zeros(length(num_list), 1);
for ii = 1:length(num_list)
    pn_dist(ii) = mean(pn_all(:, ii));
end
pn_dist_crop = pn_dist(2:end-1);
pn_dist_crop = pn_dist_crop/sum(pn_dist_crop);

PN_monkey_crop = PN_monkey(2:end-1);
PN_monkey_crop = PN_monkey_crop/sum(PN_monkey_crop);

plot(num_list(2:end-1), pn_dist_crop, 'k')

plot(num_list(2:end-1), PN_monkey_crop, 'color', [0 0.7 0])
set(gca, 'xtick', 0:10:30, 'ytick', 0:0.05:0.35, 'yticklabel', {[0], [], [0.1], [], [0.2], [], [0.3], []})
title('Fig. 5d: PN distribution')
legend({'Summation model', 'Monkey data'})
legend boxoff
xlabel('Preferred numerosity')
ylabel('Probability')
xlim([0 30]); ylim([0 0.3])

%% Weber-Fechner law
subplot(2,6,6); hold on;
sigma_model_linear = nanmean(sigma_all,1);
sigma_model_log = nanmean(sigma_log_all,1);

load('Data_Nieder2007.mat')
sig_monkey_linear = sig_linear;
sig_monkey_log = sig_log;

scatter(num_list, sig_monkey_linear, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', 'none')
scatter(num_list, sig_monkey_log, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', 'none')

scatter(num_list, sigma_model_linear, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear);
scatter(num_list, sigma_model_log, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log);

scatter(-0.5, 15, 20, 'o', 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear)
text(1, 15, 'Linear', 'fontsize', 8)
scatter(-0.5, 13.3, 20, 'o', 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log)
text(1, 13.3, 'Log', 'fontsize', 8)
scatter(-0.5, 11.7, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')
text(1, 11.7, 'Model', 'fontsize', 8)
scatter(-0.5, 10.1, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none')
text(1, 10.1, 'Monkey', 'fontsize', 8)

fit_Weber_linear = fit(num_list.', sigma_model_linear.', 'poly1');
fit_Weber_log = fit(num_list.', sigma_model_log.', 'poly1');

plot(xx_axis, fit_Weber_linear.p1*xx_axis + fit_Weber_linear.p2, 'color', color_linear)
plot(xx_axis, fit_Weber_log.p1*xx_axis + fit_Weber_log.p2, 'color', color_log)
title('Fig. 5d: Weber-Fechner law')
xlabel('Preferred numerosity')
ylabel('\sigma of Gaussian fit')
xlim([-2 32]); ylim([-2 16])
set(gca, 'xtick', 0:10:30, 'ytick', 0:8:16)

%% Calculating weight bias across numerosities
subplot(2,6,12); hold on;
weight_diff = zeros(ntrial, length(num_list));
for pp = 1:ntrial
    
    w_dec_temp = squeeze(w_dec_save(pp,:,:));
    w_inc_temp = squeeze(w_inc_save(pp,:,:));
    pn_temp = squeeze(pn_all_save(pp,:));
    
    for ii = 1:length(num_list)
        w_dec_temp2 = w_dec_temp(:, (pn_temp == num_list(ii)));
        w_inc_temp2 = w_inc_temp(:, (pn_temp == num_list(ii)));
        
        weight_diff(pp,ii) = nanmean(w_dec_temp2(:)) - nanmean(w_inc_temp2(:));
    end
end

fill([num_list, fliplr(num_list)], [nanmean(weight_diff,1)+nanstd(weight_diff,0,1), fliplr(nanmean(weight_diff,1)-nanstd(weight_diff,0,1))], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
plot(num_list, nanmean(weight_diff,1), 'k');
plot([0 num_list(end)], [0 0], 'k--')
xlim([0 30]);
% ylim([-1 1]*(1.5)*(10^(-3)));
title('Fig. 5f: Input bias')
xlabel('Preferred numerosity')
ylabel('W_{Dec} - W_{Inc}')

%% 5b. Decreasing and increasing units
numsampletmp = 10; % # of inc/dec units for visualization
units_PN = iter_units_PN{trial_ind};
units_areacoef = real(iter_units_areanumcoef{trial_ind});
units_numcoef = imag(iter_units_areanumcoef{trial_ind});
R2s = iter_units_R2s{trial_ind};
ind1 = R2s>0.1; ind2 = abs(units_areacoef)<0.1; ind = ind1 & ind2;
ind_inc = ind & (units_numcoef>0); ind_dec = ind & (units_numcoef<0);
resp_mean_n = (iter_resp_mean_n{trial_ind});
subplot(2,6,1); hold on;
incunits_meantuningcurve = rescale(mean(resp_mean_n(:,ind_inc), 2));
indtmps = datasample(find(ind_inc), numsampletmp);
for ii = 1:length(indtmps)
    p1 = plot(number_sets, rescale(resp_mean_n(:,indtmps(ii))));
    p1.Color(4) = 0.25;
end
plot(number_sets, incunits_meantuningcurve, 'k'); hold on
xlabel('Numerosity'); ylabel('Response (A.U.)'); title('Increasing units in Conv4')
subplot(2,6,7); hold on;
decunits_meantuningcurve = rescale(mean(resp_mean_n(:,ind_dec), 2));
indtmps = datasample(find(ind_dec), numsampletmp);
for ii = 1:length(indtmps)
    p1 = plot(number_sets, rescale(resp_mean_n(:,indtmps(ii))));
    p1.Color(4) = 0.25;
end
plot(number_sets, decunits_meantuningcurve, 'k'); hold on
xlabel('Numerosity'); ylabel('Response (A.U.)'); title('Decreasing units in Conv4')

%% Figure 5f. dec, inc weight bias
check1std = squeeze(std(connectedweights));
diffcheck1 = squeeze(connectedweights(:,:,1)-connectedweights(:,:,2));
diffcontrol1 = squeeze(control1s(:,:,1)-control1s(:,:,2));
% Wilcoxon signed rank test
p1 = signrank(squeeze(connectedweights(:,3,1)), squeeze(connectedweights(:,3,2)), 'tail', 'right');
p2 = signrank(squeeze(connectedweights(:,13,1)), squeeze(connectedweights(:,13,2)), 'tail', 'left');
subplot(2,6,12); hold on
l3 = shadedErrorBar(number_sets, mean(diffcheck1, 1), std(diffcheck1, [],1), 'Lineprops', 'r');
ltmp = plot(num_list, nanmean(weight_diff,1), 'k');
legend([ltmp, l3.mainLine], {'Summation model','Untrained AlexNet'})
