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

% This code performs a demo simulation for Figs. 1,2,3, and 4 in the manuscript.
% =========================================================================================================================================================


close all;clc;clear;

toolbox_chk

disp('Demo codes for Figures 1-4 of "Spontaneous generation of number sense in untrained deep neural networks"')
disp(' ')
disp('* It performs a demo version (a fewer set of stimuli than in the paper) of simulation using a single condition of the network.')
disp('  (# images for each condition: 50 -> 10 , # repetition of simulation: 100 -> 2)')
disp('* Expected running time is about 5 minutes, but may vary by system conditions.')
disp(' ')

%% Setting options for simulation
generatedat = 1;
iter = 2; % Number of networks for analysis
% issavefig = 1; % save figure

%% Setting file dir
pathtmp = pwd; % please set dir of Codes_200123 folder
% mkdir ([pathtmp '/Figs']); % generate folder to save fig. files
addpath(genpath(pathtmp));

%% Setting parameters

rand_layers_ind = [2, 6, 10, 12 14];    % Index of convolutional layer of AlexNet
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]; % Candidiate numerosities of stimulus
LOI = 'relu5';  % Name of layer at which the activation will be measured
image_iter = 10;  % Number of images for a given condition, N = 50 in the manuscript
p_th1 = 0.01; p_th2 = 0.01; p_th3 = 0.01;  % Significance levels for two-way ANOVA


% Paramters of comparison task
iterforeachN = 50; % Number of training pairs for each numerosity, N = 100 in the manuscript
iterforeachN_val = 50; % Number of testing pairs for each numerosity, N = 100 in the manuscript
sampling_numbers = 256; % Number of units used in comparison task
averageiter = 10; % Number of unit sampling trials of comparison task
ND = -30:30; % Numerical distance
radius_cand = 2*sqrt(1:8); % Stimulus radius of numerosity = 30 of area-varying stimulus set

% Training conditions : 1: all, 2: area congruent, 3: density congruent, 4:
% dot size congruent, 5: area incongruent, 6: density incongruent, 7: dot
% size incongruent
setNo_trs = 1:4;

% Network initialization condition
vercand = [1]; % 1:he normal, 2:lecun normal, 3:he uniform, 4:lecun uniform
stdfacs = [1]; % Factor multiplied to weight (variation of weight)

%% Loading pretrained network
net = alexnet;  % load pretrained network

if generatedat
    %% Generating stimulus set
    
    disp(['[1/4] Generating a stimulus set...'])
    % Stimulus set (Nasr 2019)
    [image_sets_standard, image_sets_control1, image_sets_control2, polyxy]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    
    % Training data pool
    image_sets_standardT = image_sets_standard;
    image_sets_control1T = image_sets_control1;
    image_sets_control2T =  image_sets_control2;
    % Validation data pool
    [image_sets_standardV, image_sets_control1V, image_sets_control2V]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    
    % Area-varying set (Stoianov & zorzi 2012)
    % Training data pool
    [image_sets_zorziT, image_sets_zorziT_stat] = Stimulus_generation_Zorzi2(number_sets, image_iter, radius_cand);
    
    % Validation data pool
    [image_sets_zorziV, image_sets_zorziV_stat] = Stimulus_generation_Zorzi2(number_sets, image_iter, radius_cand);
    
    %% Calculating Number selectivity for pretrained networks
    
    % Variables
    % Regarding network and number neurons
    iter_resp_mean_n= cell(length(stdfacs), length(vercand),iter); % Mean resp of this network
    iter_units_PN = cell(length(stdfacs), length(vercand),iter); % PN of this network
    iter_units_PN2 = cell(length(stdfacs), length(vercand),iter); % PN of this network
    iter_tuning_curve_n = cell(length(stdfacs), length(vercand),iter); % Tuning curve of this network
    iter_resp_std_n = cell(length(stdfacs), length(vercand),iter); % Response of this network
    iter_networkweights = cell(length(stdfacs), length(vercand), iter); % Weight of this network
    iter_units_sigmas = cell(length(stdfacs), length(vercand), iter); % Fitted Gaussian width
    
    % Regarding numerosity comaparison task
    
    % using Nasr 2019
    iter_correct_original = cell(length(stdfacs), length(vercand), iter); % Tuning curves of correct trials
    iter_wrong_original = cell(length(stdfacs), length(vercand), iter); % Tuning curves of incorret trials
    iter_correctness_inputnumerosity_original = cell(length(stdfacs), length(vercand), iter); % Confusing matrix of performance
    iter_congincong_dat_original = cell(length(stdfacs), length(vercand), iter); % Data for congruent-incongruent task
    
    % using Stoianov & Zorzi 2012
    iter_performance = cell(length(stdfacs), length(vercand), iter); % Performance of the network
    iter_correct = cell(length(stdfacs), length(vercand), iter); % Tuning curves of correct trials
    iter_wrong = cell(length(stdfacs), length(vercand), iter); % Tuning curves of incorret trials
    iter_correctness_inputnumerosity = cell(length(stdfacs), length(vercand), iter); % Confusing matrix of performance
    iter_congincong_dat = cell(length(stdfacs), length(vercand), iter); % Data for congruent-incongruent task
    
    
    for iterind = 1:iter
        
        for stdfacind = 1:length(stdfacs)
            
            stdfac = stdfacs(stdfacind);
            for verind = 1:length(vercand)
                vertmp = vercand(verind);
                
                %% Initializing network and finding number neurons in the network
                disp(['[2/4] Calculating number selectivity in randomly initialized networks...'])
                
                [net_test, ~, networkweights, ~] = Initializeweight_he2(net, rand_layers_ind, vertmp, stdfac); % Initialized network
                iter_networkweights{stdfacind, verind, iterind} = networkweights; % Save network weights
                
                [units_PN_rand, resp_mean, resp_std, tuning_curve, ~, ind_NNS_initialized, response_tot] = ...
                    getNumberSensefromNet(net_test, image_sets_standard, image_sets_control1,image_sets_control2...
                    , p_th1, p_th2, p_th3, LOI, number_sets, 0, 1);
                
                NSind_rand = find(units_PN_rand>0);
                NNSind_rand = find(ind_NNS_initialized);
                iter_resp_mean_n{stdfacind, verind,iterind} = resp_mean;
                iter_units_PN{stdfacind, verind,iterind} = units_PN_rand;
                iter_tuning_curve_n{stdfacind, verind,iterind} = tuning_curve;
                iter_resp_std_n{stdfacind, verind,iterind} = resp_std;
                
                %% Cross validation
                [~,PN_sample_set] = max(resp_mean(:,units_PN_rand>0));
                
                % PN from validation stimulus
                [~, resp_meant, ~, ~, ~, ~] = ...
                    getNumberSensefromNet3(net_test, image_sets_standardV, image_sets_control1V,image_sets_control2V...
                    , p_th1, p_th2, p_th3, LOI, number_sets, 0, 1);
                [~,PN_test_set] = max(resp_meant(:,units_PN_rand>0));
                
                % PN from different sets
                tmp1 = squeeze(mean(response_tot(:,1:image_iter, :), 2)); 
                tmp2 = squeeze(mean(response_tot(:,image_iter+1:image_iter*2, :), 2)); 
                tmp3 = squeeze(mean(response_tot(:,image_iter*2+1:image_iter*3, :), 2));
                [~,PN_set1] = max(tmp1(:,NSind_rand)); 
                [~,PN_set2] = max(tmp2(:,NSind_rand)); 
                [~,PN_set3] = max(tmp3(:,NSind_rand));
                
                
                %% ------ Numerosity comparison task using Nasr 2019 stimulus set
                disp(['[3/4] Simulating numerosity comparison task (normal test)...'])
                
                % Generating sample, test pairs, and simple statistics
                [image_sets_sample, image_sets_test, label_tr, ~, ~, ~] = ...
                    imgToSampleTest_setN_congincongtag(image_sets_standardT, image_sets_control1T, image_sets_control2T, [1,2,3], iterforeachN, number_sets);
                
                [image_sets_sampleV, image_sets_testV, label_val, areas_val, sets_val, nums_val] = ...
                    imgToSampleTest_setN_congincongtag(image_sets_standardV, image_sets_control1V, image_sets_control2V, [1,2,3], iterforeachN_val, number_sets);
                
                % Training with simple img + SVM
                YlabelT = categorical(label_tr{2}(:));
                YlabelV = categorical(label_val{2}(:));
                image_rawT = reshape(image_sets_sample, [size(image_sets_sample,1)*size(image_sets_sample,2), ...
                    size(image_sets_sample,3)*size(image_sets_sample,4)]);
                
                image_rawV = reshape(image_sets_sampleV, [size(image_sets_sampleV,1)*size(image_sets_sampleV,2), ...
                    size(image_sets_sampleV,3)*size(image_sets_sampleV,4)]);
                
                % Getting response to test/validation stimulus in randomized network
                response_tot_sample = getactivation(net_test, LOI, image_sets_sample); response_tot_test = getactivation(net_test, LOI, image_sets_test);
                respmat = cell(1,2); respmat{1} = response_tot_sample; respmat{2} = response_tot_test;
                resp_rand = respmat;
                
                response_tot_sample = getactivation(net_test, LOI, image_sets_sampleV); response_tot_test = getactivation(net_test, LOI, image_sets_testV);
                respmat = cell(1,2); respmat{1} = response_tot_sample; respmat{2} = response_tot_test;
                resp_rand_val = respmat;
                
                
                %% Training and testing
                
                perf_dat = cell(1,length(setNo_trs)); correct_dat = cell(1,length(setNo_trs)); wrong_dat = cell(1,length(setNo_trs)); congincong_dat = cell(6,length(setNo_trs));
                
                for setindtmp = 1% :length(setNo_trs)
                    
                    indused_tr = ones(1,length(YlabelT))>0; % Stimulus used for training
                    
                    congincong_dat{1,setindtmp} = areas_val;
                    congincong_dat{4,setindtmp} = sets_val;
                    congincong_dat{5,setindtmp} = nums_val;
                    performance_mat = zeros(4, averageiter, length(sampling_numbers)); % dim1: none(reomved), perm, nonselective, controlperm(removed)
                    answer_mat = cell(5,averageiter, length(sampling_numbers)); % dim1: none(removed), initialized-selective, nonselective, controlperm(removed), rawimg(modified),
                    tuning_curves_correct_set = cell(averageiter, length(sampling_numbers));
                    tuning_curves_wrong_set = cell(averageiter, length(sampling_numbers));
                    
                    % performance for raw image
                    image_rawT2 = image_rawT(:,indused_tr);
                    YlabelT2 = YlabelT(indused_tr, 1);
                    Mdl_raw = fitclinear(squeeze(image_rawT2)', YlabelT2);
                    [label, score] = predict(Mdl_raw, squeeze(image_rawV)');
                    answer_mat{5,1,1} = double(string(label))+1i*double(string(YlabelV));
                    
                    % Simulating network performance
                    correctness_inputnumerosity_dat = cell(averageiter, length(sampling_numbers), 3);
                    for ii = 1:averageiter
                        for kk = 1:length(sampling_numbers)
                            sampling_unitsN = sampling_numbers(kk);
                            sampleind_perm = datasample(NSind_rand, sampling_unitsN, 'Replace', false);
                            sampleind_perm2 = datasample( NNSind_rand, sampling_unitsN, 'Replace', false);
                            
                            resp_rand_part = cell(1,2); resp_rand_val_part = cell(1,2);
                            resp_rand_part2 = cell(1,2); resp_rand_val_part2 = cell(1,2);
                            
                            
                            % For selective
                            tmp = resp_rand{1}; tmp = tmp(:,:,sampleind_perm); resp_rand_part{1} = tmp;
                            tmp = resp_rand{2}; tmp = tmp(:,:,sampleind_perm); resp_rand_part{2} = tmp;
                            tmp = resp_rand_val{1}; tmp = tmp(:,:,sampleind_perm); resp_rand_val_part{1} = tmp;
                            tmp = resp_rand_val{2}; tmp = tmp(:,:,sampleind_perm); resp_rand_val_part{2} = tmp;
                            
                            % For non-selective
                            tmp = resp_rand{1}; tmp = tmp(:,:,sampleind_perm2); resp_rand_part2{1} = tmp;
                            tmp = resp_rand{2}; tmp = tmp(:,:,sampleind_perm2); resp_rand_part2{2} = tmp;
                            tmp = resp_rand_val{1}; tmp = tmp(:,:,sampleind_perm2); resp_rand_val_part2{1} = tmp;
                            tmp = resp_rand_val{2}; tmp = tmp(:,:,sampleind_perm2); resp_rand_val_part2{2} = tmp;
                            
                            % Non-selective
                            [XTrain_pm, YTrain_pm, ~] = getdataformat2(resp_rand_part2, label_tr); %% Get training dataset
                            XTrain_pm = XTrain_pm(:,:,:, indused_tr);
                            YTrain_pm = YTrain_pm(indused_tr,1);
                            [XVal_pm, YVal_pm, ~] = getdataformat2(resp_rand_val_part2, label_val); %% Get validation dataset
                            Mdl_pm = fitclinear(squeeze(XTrain_pm)', YTrain_pm);
                            [label, ~] = predict(Mdl_pm, squeeze(XVal_pm)');
                            correct_ratio_pm = length(find(label==YVal_pm))/length(YVal_pm);
                            performance_mat(3, ii, kk) = correct_ratio_pm;
                            answer_mat{3,ii,kk} = double(string(label))+1i*double(string(YVal_pm));
                            
                            % Selective
                            [XTrain_pm, YTrain_pm, labelsTrain_pm] = getdataformat2(resp_rand_part, label_tr); %% Get training dataset
                            XTrain_pm = XTrain_pm(:,:,:, indused_tr);
                            YTrain_pm = YTrain_pm(indused_tr,1);
                            [XVal_pm, YVal_pm, labelsVal_pm] = getdataformat2(resp_rand_val_part, label_val); %% Get validation dataset
                            Mdl_pm = fitclinear(squeeze(XTrain_pm)', YTrain_pm);
                            [label, score] = predict(Mdl_pm, squeeze(XVal_pm)');
                            correct_ratio_pm = length(find(label==YVal_pm))/length(YVal_pm);
                            performance_mat(2, ii, kk) = correct_ratio_pm;
                            answer_mat{2,ii,kk} = double(string(label))+1i*double(string(YVal_pm));
                            
                            %% Comparing tuning curves of correct/incorrect trial
                            indcorrect = find(label==YVal_pm); % Correct pair
                            indwrong = find(~(label==YVal_pm)); % Incorrect pair
                            indcorr = (label==YVal_pm);
                            
                            PNlabel = units_PN_rand(sampleind_perm);
                            tmp = label_val{1};
                            numlabel_sample = squeeze(tmp(1,:,:)); numlabel_sample = numlabel_sample(:);
                            numlabel_test = squeeze(tmp(2,:,:)); numlabel_test = numlabel_test(:);
                            
                            correctness_inputnumerosity_dat{ii, kk,1} = indcorr;
                            correctness_inputnumerosity_dat{ii, kk,2} = numlabel_sample;
                            correctness_inputnumerosity_dat{ii, kk,3} = numlabel_test;
                            
                            % tuning curves for correct/incorrect
                            tuning_curves_correct = zeros(sampling_unitsN, length(ND));
                            tuning_curves_wrong = zeros(sampling_unitsN, length(ND));
                            for ij = 1:sampling_unitsN % # of units
                                PNtmp = number_sets(PNlabel(ij));
                                tmp = resp_rand_val_part{1};
                                resp_sampleunit = squeeze(tmp(:,:,ij))';
                                resp_sampleunit = resp_sampleunit(:);
                                tmp = resp_rand_val_part{2};
                                resp_testunit = squeeze(tmp(:,:,ij))';
                                resp_testunit = resp_testunit(:);
                                resp_correct = cell(1,length(ND));
                                resp_wrong = cell(1,length(ND));
                                for jj = 1:length(indcorrect) % average for correct trial
                                    indtmp = indcorrect(jj);
                                    resptmp_sample = resp_sampleunit(indtmp);
                                    resptmp_test = resp_testunit(indtmp);
                                    sampleNtmp = number_sets(numlabel_sample(indtmp));
                                    testNtmp = number_sets(numlabel_test(indtmp));
                                    
                                    NDtmp_sample = sampleNtmp-PNtmp;
                                    NDindtmp_sample = 31+NDtmp_sample;
                                    resp_correct{NDindtmp_sample} = [resp_correct{NDindtmp_sample}, resptmp_sample];
                                end
                                correct_resptmp = zeros(1,length(ND));
                                for jj = 1:length(resp_correct)
                                    correct_resptmp(jj) = mean(resp_correct{jj});
                                end
                                tuning_curves_correct(ij,:) = correct_resptmp/max(correct_resptmp); % Tuning curve for correct trial
                                
                                for jj = 1:length(indwrong)
                                    indtmp = indwrong(jj);
                                    resptmp_sample = resp_sampleunit(indtmp);
                                    sampleNtmp = number_sets(numlabel_sample(indtmp));
                                    
                                    NDtmp_sample = sampleNtmp-PNtmp;
                                    NDindtmp_sample = 31+NDtmp_sample;
                                    resp_wrong{NDindtmp_sample} = [resp_wrong{NDindtmp_sample}, resptmp_sample];
                                end
                                wrong_resptmp = zeros(1,length(ND));
                                for jj = 1:length(resp_wrong)
                                    wrong_resptmp(jj) = mean(resp_wrong{jj});
                                end
                                tuning_curves_wrong(ij,:) = wrong_resptmp/max(wrong_resptmp); % Tuning curve for wrong trial
                            end
                            
                            tuning_curves_correct_set{ii,kk} = tuning_curves_correct;
                            tuning_curves_wrong_set{ii,kk} = tuning_curves_wrong;
                            
                            
                        end
                    end
                    congincong_dat{6,setindtmp} = answer_mat;
                    
                    
                    aveTNCV_correct = zeros(length(sampling_numbers),averageiter, length(ND));
                    aveTNCV_wrong = zeros(length(sampling_numbers),averageiter, length(ND));
                    for kk = 1:length(sampling_numbers)
                        for ii = 1:averageiter
                            tuning_curves_tmp = tuning_curves_correct_set{ii,kk};
                            cvmean = nanmean(tuning_curves_tmp, 1);
                            scaletmp = cvmean(31);
                            tmp = cvmean/scaletmp;
                            aveTNCV_correct(kk, ii, :) = tmp;
                            
                            tuning_curves_tmp = tuning_curves_wrong_set{ii,kk};
                            cvmean = nanmean(tuning_curves_tmp, 1);
                            tmp = cvmean/scaletmp;
                            aveTNCV_wrong(kk, ii, :) = tmp;
                            
                        end
                        correct = squeeze(nanmean(aveTNCV_correct(kk,:,:),2)); std1 = squeeze(nanstd(aveTNCV_correct(kk,:,:),[],2));
                        wrong = squeeze(nanmean(aveTNCV_wrong(kk,:,:),2)); std2 = squeeze(nanstd(aveTNCV_wrong(kk,:,:),[],2));
                        
                    end
                    
                    for kk = 1:length(sampling_numbers)
                        performance_pret_permtmp = squeeze(performance_mat(:,:,kk));
                        
                    end
                    
                    performance_pret_perm_sets(1, 1) = mean(performance_pret_permtmp(1,:));
                    performance_pret_perm_sets(1, 2) = mean(performance_pret_permtmp(2,:));
                    performance_pret_perm_sets(1, 3) = mean(performance_pret_permtmp(3,:));
                    performance_pret_perm_sets(1, 4) = mean(performance_pret_permtmp(4,:));
                    
                    perf_dat{setindtmp} = performance_pret_perm_sets;
                    correct_dat{setindtmp} = correct;
                    wrong_dat{setindtmp} = wrong;
                    
                end
                iter_correct_original{stdfacind, verind, iterind} = correct_dat;
                iter_wrong_original{stdfacind, verind, iterind} = wrong_dat;
                iter_correctness_inputnumerosity_original{stdfacind, verind, iterind} = correctness_inputnumerosity_dat;
                iter_congincong_dat_original{stdfacind, verind, iterind} = congincong_dat;
                
                
                
                %% ----- Numerosity comparison task using Stoianov & Zorzi 2012 stimulus set (congruent - incongruent test)
                disp(['[4/4] Simulating numerosity comparison task (congruent - incongruent test)...'])
                % Generating train sets
                [image_sets_sample, image_sets_test, label_tr, areas_tr, dens_tr, dotsz_tr, sets_tr, nums_tr] = ...
                    imgToSampleTest_zorzi_congincongtag(image_sets_zorziT, image_sets_zorziT_stat, iterforeachN, number_sets);
                
                % Low-level statistics of images
                areaS = squeeze(areas_tr(1,:,:)); areaS = areaS(:); areaT = squeeze(areas_tr(2,:,:)); areaT = areaT(:);
                densS = squeeze(dens_tr(1,:,:)); densS = densS(:); densT = squeeze(dens_tr(2,:,:)); densT = densT(:);
                dotszS = squeeze(dotsz_tr(1,:,:)); dotszS = dotszS(:); dotszT = squeeze(dotsz_tr(2,:,:)); dotszT = dotszT(:);
                numS = squeeze(nums_tr(1,:,:)); numS = numS(:); numT = squeeze(nums_tr(2,:,:)); numT = numT(:);
                setS = squeeze(sets_tr(1,:,:)); setS = setS(:); setT = squeeze(sets_tr(2,:,:)); setT = setT(:);
                
                congruentind_sz = ((areaS>=areaT & numS>numT) | (areaS<=areaT & numS<numT)) | (setS==setT); % just because same area is sometimes considered different in matlab..
                incongruentind_sz = ~congruentind_sz;% & ~(setS==setT);
                congruentind_dens = (densS<=densT & numS>numT) | (densS>=densT & numS<numT);% & (setS~=setT);
                incongruentind_dens = ~congruentind_dens;
                congruentind_dotsz = (dotszS>=dotszT & numS>numT) | (dotszS<=dotszT & numS<numT);% & (setS~=setT);
                incongruentind_dotsz = ~congruentind_dotsz;
                
                
                % Generating validation sets
                [image_sets_sampleV, image_sets_testV, label_val, areas_val,dens_val, dotsz_val,  sets_val, nums_val] = ...
                    imgToSampleTest_zorzi_congincongtag(image_sets_zorziV, image_sets_zorziV_stat, iterforeachN, number_sets);
                
                % Training with simple image + SVM
                
                YlabelT = categorical(label_tr{2}(:));
                YlabelV = categorical(label_val{2}(:));
                image_rawT = reshape(image_sets_sample, [size(image_sets_sample,1)*size(image_sets_sample,2), ...
                    size(image_sets_sample,3)*size(image_sets_sample,4)]);
                
                image_rawV = reshape(image_sets_sampleV, [size(image_sets_sampleV,1)*size(image_sets_sampleV,2), ...
                    size(image_sets_sampleV,3)*size(image_sets_sampleV,4)]);
                
                
                % Getting response to test/validation stimulus in randomized network
                response_tot_sample = getactivation(net_test, LOI, image_sets_sample); response_tot_test = getactivation(net_test, LOI, image_sets_test);
                respmat = cell(1,2); respmat{1} = response_tot_sample; respmat{2} = response_tot_test;
                resp_rand = respmat;
                
                response_tot_sample = getactivation(net_test, LOI, image_sets_sampleV); response_tot_test = getactivation(net_test, LOI, image_sets_testV);
                respmat = cell(1,2); respmat{1} = response_tot_sample; respmat{2} = response_tot_test;
                resp_rand_val = respmat;
                
                
                %% Training and testing
                
                perf_dat = cell(1,length(setNo_trs)); correct_dat = cell(1,length(setNo_trs)); wrong_dat = cell(1,length(setNo_trs)); congincong_dat = cell(6,length(setNo_trs));
                
                for setindtmp = 1:length(setNo_trs)
                    
                    if setindtmp == 1
                        indused_tr = ones(1,length(areaS))>0;
                    elseif setindtmp == 2
                        indused_tr = congruentind_sz;
                    elseif setindtmp ==3
                        indused_tr = congruentind_dens;
                    elseif setindtmp ==4
                        indused_tr = congruentind_dotsz;
                    elseif setindtmp == 5
                        indused_tr = incongruentind_sz;
                    elseif setindtmp == 6
                        indused_tr = incongruentind_dens;
                    elseif setindtmp == 7
                        indused_tr = incongruentind_dotsz;
                    end
                    congincong_dat{1,setindtmp} = areas_val;
                    congincong_dat{2,setindtmp} = dens_val;
                    congincong_dat{3,setindtmp} = dotsz_val;
                    congincong_dat{4,setindtmp} = sets_val;
                    congincong_dat{5,setindtmp} = nums_val;
                    
                    performance_mat = zeros(4, averageiter, length(sampling_numbers)); % dim1: none(reomved), perm, nonselective, controlperm(removed)
                    answer_mat = cell(5,averageiter, length(sampling_numbers)); % dim1: none(removed), initialized-selective, nonselective, controlperm(removed), rawimg(modified),
                    tuning_curves_correct_set = cell(averageiter, length(sampling_numbers));
                    tuning_curves_wrong_set = cell(averageiter, length(sampling_numbers));
                    
                    % Performance using raw img pixel values
                    image_rawT2 = image_rawT(:,indused_tr);
                    YlabelT2 = YlabelT(indused_tr, 1);
                    
                    Mdl_raw = fitclinear(squeeze(image_rawT2)', YlabelT2);
                    [label, score] = predict(Mdl_raw, squeeze(image_rawV)');
                    answer_mat{5,1,1} = double(string(label))+1i*double(string(YlabelV));
                    
                    % Performance of units in network
                    for ii = 1:averageiter
                        for kk = 1:length(sampling_numbers)
                            sampling_unitsN = sampling_numbers(kk);
                            sampleind_perm = datasample(NSind_rand, sampling_unitsN, 'Replace', false);
                            sampleind_perm2 = datasample( NNSind_rand, sampling_unitsN, 'Replace', false);
                            resp_pretrained_part = cell(1,2); resp_pretrained_val_part = cell(1,2);
                            resp_rand_part = cell(1,2); resp_rand_val_part = cell(1,2);
                            resp_rand_part2 = cell(1,2); resp_rand_val_part2 = cell(1,2);
                            
                            % for selective
                            tmp = resp_rand{1}; tmp = tmp(:,:,sampleind_perm); resp_rand_part{1} = tmp;
                            tmp = resp_rand{2}; tmp = tmp(:,:,sampleind_perm); resp_rand_part{2} = tmp;
                            tmp = resp_rand_val{1}; tmp = tmp(:,:,sampleind_perm); resp_rand_val_part{1} = tmp;
                            tmp = resp_rand_val{2}; tmp = tmp(:,:,sampleind_perm); resp_rand_val_part{2} = tmp;
                            
                            % for non-selective
                            tmp = resp_rand{1}; tmp = tmp(:,:,sampleind_perm2); resp_rand_part2{1} = tmp;
                            tmp = resp_rand{2}; tmp = tmp(:,:,sampleind_perm2); resp_rand_part2{2} = tmp;
                            tmp = resp_rand_val{1}; tmp = tmp(:,:,sampleind_perm2); resp_rand_val_part2{1} = tmp;
                            tmp = resp_rand_val{2}; tmp = tmp(:,:,sampleind_perm2); resp_rand_val_part2{2} = tmp;
                            
                            % non-selective
                            [XTrain_pm, YTrain_pm, ~] = getdataformat2(resp_rand_part2, label_tr); %% get training dataset
                            %                         if trainWithOnlyCong
                            XTrain_pm = XTrain_pm(:,:,:, indused_tr);
                            YTrain_pm = YTrain_pm(indused_tr,1);
                            %                         end
                            [XVal_pm, YVal_pm, ~] = getdataformat2(resp_rand_val_part2, label_val); %% get validation dataset
                            Mdl_pm = fitclinear(squeeze(XTrain_pm)', YTrain_pm);
                            [label, ~] = predict(Mdl_pm, squeeze(XVal_pm)');
                            correct_ratio_pm = length(find(label==YVal_pm))/length(YVal_pm);
                            performance_mat(3, ii, kk) = correct_ratio_pm;
                            answer_mat{3,ii,kk} = double(string(label))+1i*double(string(YVal_pm));
                            
                            % selective
                            [XTrain_pm, YTrain_pm, labelsTrain_pm] = getdataformat2(resp_rand_part, label_tr); %% get training dataset
                            %                         if trainWithOnlyCong
                            XTrain_pm = XTrain_pm(:,:,:, indused_tr);
                            YTrain_pm = YTrain_pm(indused_tr,1);
                            %                         end
                            [XVal_pm, YVal_pm, labelsVal_pm] = getdataformat2(resp_rand_val_part, label_val); %% get validation dataset
                            
                            Mdl_pm = fitclinear(squeeze(XTrain_pm)', YTrain_pm);
                            [label, score] = predict(Mdl_pm, squeeze(XVal_pm)');
                            correct_ratio_pm = length(find(label==YVal_pm))/length(YVal_pm);
                            performance_mat(2, ii, kk) = correct_ratio_pm;
                            answer_mat{2,ii,kk} = double(string(label))+1i*double(string(YVal_pm));
                            
                            % comparing correct/incorrect trial
                            indcorrect = find(label==YVal_pm);
                            indwrong = find(~(label==YVal_pm));
                            indcorr = (label==YVal_pm);
                            
                            PNlabel = units_PN_rand(sampleind_perm);
                            tmp = label_val{1};
                            numlabel_sample = squeeze(tmp(1,:,:)); numlabel_sample = numlabel_sample(:);
                            numlabel_test = squeeze(tmp(2,:,:)); numlabel_test = numlabel_test(:);
                            correctness_inputnumerosity_dat{ii, kk,1} = indcorr;
                            correctness_inputnumerosity_dat{ii, kk,2} = numlabel_sample;
                            correctness_inputnumerosity_dat{ii, kk,3} = numlabel_test;
                            
                            % tuning curves for correct/incorrect trials
                            tuning_curves_correct = zeros(sampling_unitsN, length(ND));
                            tuning_curves_wrong = zeros(sampling_unitsN, length(ND));
                            for ij = 1:sampling_unitsN % # of units
                                PNtmp = number_sets(PNlabel(ij));
                                tmp = resp_rand_val_part{1};
                                resp_sampleunit = squeeze(tmp(:,:,ij))';
                                resp_sampleunit = resp_sampleunit(:);
                                tmp = resp_rand_val_part{2};
                                resp_testunit = squeeze(tmp(:,:,ij))';
                                resp_testunit = resp_testunit(:);
                                resp_correct = cell(1,length(ND));
                                resp_wrong = cell(1,length(ND));
                                for jj = 1:length(indcorrect) % average for correct trial
                                    indtmp = indcorrect(jj);
                                    resptmp_sample = resp_sampleunit(indtmp);
                                    resptmp_test = resp_testunit(indtmp);
                                    sampleNtmp = number_sets(numlabel_sample(indtmp));
                                    testNtmp = number_sets(numlabel_test(indtmp));
                                    
                                    NDtmp_sample = sampleNtmp-PNtmp;
                                    %         NDtmp_test = testNtmp-PNtmp;
                                    NDindtmp_sample = 31+NDtmp_sample;
                                    %         NDindtmp_test = 31+NDtmp_test;
                                    resp_correct{NDindtmp_sample} = [resp_correct{NDindtmp_sample}, resptmp_sample];
                                    %         resp_correct{NDindtmp_test} = [resp_correct{NDindtmp_test}, resptmp_test];
                                end
                                correct_resptmp = zeros(1,length(ND));
                                for jj = 1:length(resp_correct)
                                    correct_resptmp(jj) = mean(resp_correct{jj});
                                end
                                tuning_curves_correct(ij,:) = correct_resptmp/max(correct_resptmp);
                                
                                for jj = 1:length(indwrong)
                                    indtmp = indwrong(jj);
                                    resptmp_sample = resp_sampleunit(indtmp);
                                    sampleNtmp = number_sets(numlabel_sample(indtmp));
                                    
                                    NDtmp_sample = sampleNtmp-PNtmp;
                                    NDindtmp_sample = 31+NDtmp_sample;
                                    resp_wrong{NDindtmp_sample} = [resp_wrong{NDindtmp_sample}, resptmp_sample];
                                end
                                wrong_resptmp = zeros(1,length(ND));
                                for jj = 1:length(resp_wrong)
                                    wrong_resptmp(jj) = mean(resp_wrong{jj});
                                end
                                tuning_curves_wrong(ij,:) = wrong_resptmp/max(wrong_resptmp);
                            end
                            
                            tuning_curves_correct_set{ii,kk} = tuning_curves_correct;
                            tuning_curves_wrong_set{ii,kk} = tuning_curves_wrong;
                            
                            
                        end
                    end
                    congincong_dat{6,setindtmp} = answer_mat;
                    
                    aveTNCV_correct = zeros(length(sampling_numbers),averageiter, length(ND));
                    aveTNCV_wrong = zeros(length(sampling_numbers),averageiter, length(ND));
                    for kk = 1:length(sampling_numbers)
                        for ii = 1:averageiter
                            tuning_curves_tmp = tuning_curves_correct_set{ii,kk};
                            cvmean = nanmean(tuning_curves_tmp, 1);
                            scaletmp = cvmean(31);
                            tmp = cvmean/scaletmp;
                            aveTNCV_correct(kk, ii, :) = tmp;
                            
                            tuning_curves_tmp = tuning_curves_wrong_set{ii,kk};
                            cvmean = nanmean(tuning_curves_tmp, 1);
                            tmp = cvmean/scaletmp;
                            aveTNCV_wrong(kk, ii, :) = tmp;
                            
                        end
                        correct = squeeze(nanmean(aveTNCV_correct(kk,:,:),2)); std1 = squeeze(nanstd(aveTNCV_correct(kk,:,:),[],2));
                        wrong = squeeze(nanmean(aveTNCV_wrong(kk,:,:),2)); std2 = squeeze(nanstd(aveTNCV_wrong(kk,:,:),[],2));
                        
                    end
                    
                    for kk = 1:length(sampling_numbers)
                        performance_pret_permtmp = squeeze(performance_mat(:,:,kk));
                        
                    end
                    
                    performance_pret_perm_sets(1, 1) = mean(performance_pret_permtmp(1,:));
                    performance_pret_perm_sets(1, 2) = mean(performance_pret_permtmp(2,:));
                    performance_pret_perm_sets(1, 3) = mean(performance_pret_permtmp(3,:));
                    performance_pret_perm_sets(1, 4) = mean(performance_pret_permtmp(4,:));
                    
                    perf_dat{setindtmp} = performance_pret_perm_sets;
                    correct_dat{setindtmp} = correct;
                    wrong_dat{setindtmp} = wrong;
                    
                end
                iter_performance{stdfacind, verind,iterind} = perf_dat;
                iter_correct{stdfacind, verind, iterind} = correct_dat;
                iter_wrong{stdfacind, verind, iterind} = wrong_dat;
                iter_correctness_inputnumerosity{stdfacind, verind, iterind} = correctness_inputnumerosity_dat;
                iter_congincong_dat{stdfacind, verind, iterind} = congincong_dat;
                
            end
        end
    end
  
else
%     load('Results_tot_200516') %load data
%     load('Image_sets')
end


%% ------------------------------------------------------------------ Figure 1

%% Figure 1d. Stimulus set
figure('units','normalized','outerposition',[0 0.25 0.5 0.5]); %set(gcf,'Visible', 'off')
%% standard
indtmp = randi(size(image_sets_standard,3));
subplot(3,5,1);tmp = squeeze(image_sets_standard(:,:, indtmp,3));imagesc(tmp);colormap(gray);axis image xy off
subplot(3,5,2);tmp = squeeze(image_sets_standard(:,:, indtmp,9));imagesc(tmp);colormap(gray);axis image xy off
title('Fig. 1d: Stimulus encoding numerosity')
subplot(3,5,3);tmp = squeeze(image_sets_standard(:,:, indtmp,15));imagesc(tmp);colormap(gray);axis image xy off
%% control 1
subplot(3,5,6);tmp = squeeze(image_sets_control1(:,:, indtmp,3));imagesc(tmp);colormap(gray);axis image xy off
subplot(3,5,7);tmp = squeeze(image_sets_control1(:,:, indtmp,9));imagesc(tmp);colormap(gray);axis image xy off
subplot(3,5,8);tmp = squeeze(image_sets_control1(:,:, indtmp,15));imagesc(tmp);colormap(gray);axis image xy off
%% control 2
polyxx = squeeze(polyxy(:,:,:,1)); polyyy = squeeze(polyxy(:,:,:,2));
subplot(3,5,11);tmp = squeeze(image_sets_control2(:,:, indtmp,3));imagesc(tmp);colormap(gray);axis image xy off
pgon = polyshape(squeeze(polyxx(3,indtmp, :,:)),squeeze(polyyy(3,indtmp, :,:)));hold on;plot(pgon)
subplot(3,5,12);tmp = squeeze(image_sets_control2(:,:, indtmp,9));imagesc(tmp);colormap(gray);axis image xy off
pgon = polyshape(squeeze(polyxx(9,indtmp, :,:)),squeeze(polyyy(9,indtmp, :,:)));hold on;plot(pgon)
subplot(3,5,13);tmp = squeeze(image_sets_control2(:,:, indtmp,15));imagesc(tmp);colormap(gray);axis image xy off
pgon = polyshape(squeeze(polyxx(15,indtmp, :,:)),squeeze(polyyy(15,indtmp, :,:)));hold on;plot(pgon)

% if issavefig; savefig([pathtmp '/Figs/1d']); end

%% Figure 1f. Number neurons in untrained AlexNet

if ~generatedat
    [net_test, ~, networkweights, ~] = Initializeweight_he2(net, rand_layers_ind, 1,1); % Initialized network
end
vis_N = 1;
%figure; %set(gcf,'Visible', 'off')
for layers = 2
    tmp = net.Layers(layers).Weights; sztmp = 1:size(tmp,4); 
    indtmpp=datasample(sztmp, vis_N,'Replace', false);
    for ind = 1:length(indtmpp)
        tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
        subplot(3,5,4);imagesc(squeeze(tmp(:,:,1,indtmpp(ind))));axis image off;caxis([-caxtmp caxtmp])
        title('Pretrained')
    end
    tmp = net_test.Layers(layers).Weights;sztmp = 1:size(tmp,4);
    indtmpp=datasample(sztmp, vis_N,'Replace', false);
    for ind = 1:length(indtmpp)
        tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
        ax=subplot(3,5,5);imagesc(squeeze(tmp(:,:,1,indtmpp(ind))));axis image off;caxis(ax, [-caxtmp caxtmp])
        title('Untrained')
    end
    colormap(gray);
end

PN_ex = [1, 5, 12, 16];  % Example PN for visualization
units_PN = iter_units_PN{iter}; resp_mean = iter_resp_mean_n{iter}; resp_std = iter_resp_std_n{iter};
resp_totmean_tmp = zeros(length(number_sets), length(PN_ex)); resp_totstd_tmp = zeros(length(number_sets), length(PN_ex));
for ii = 1:length(number_sets)
    PNtmp = ii;
    indcand = find(units_PN ==PNtmp);
    if length(indcand)>0
        indcand2 = datasample(indcand, 1);
        resp_totmean_tmp(:,ii) = resp_mean(:, indcand2);
        resp_totstd_tmp(:,ii) = resp_std(:, indcand2)/sqrt(3*image_iter);
    end
end

for ii = 1:length(PN_ex)
    jj = ii+8;    if ii>=3; jj = ii+11; end
    
    subplot(3,5,jj)
    hold on
    shadedErrorBar(number_sets, resp_totmean_tmp(:,PN_ex(ii)), resp_totstd_tmp(:,PN_ex(ii)))
    xlabel('Numerosity');ylabel('Resp.')
    title(['Fig. 1f: PN = ' num2str(number_sets(PN_ex(ii)))])
end
%sgtitle('Figure 1. Number neurons in randomized AlexNet')
% if issavefig; savefig([pathtmp '/Figs/1f']); end


%% ------------------------------------------------------------------ Figure 2

%% Figure 2a. Preferred numerosity (PN) in repeated trials with new stimulus sets
figure('units','normalized','outerposition',[0 0.25 0.8 0.5]);
subplot(2,4,1)
tmp1 = 2*(PN_sample_set-1);tmp1(tmp1==0) = 0;
tmp2 = 2*(PN_test_set-1);tmp2(tmp2==0)=0;
heattmp = full(sparse(ceil(tmp1/2)+1, ceil(tmp2/2)+1, 1));
heattmp2 = heattmp';
for kk = 1:size(heattmp2,2)
    heattmp2(:,kk) = heattmp2(:,kk);%/sum(heattmp2(:,kk));
end
imagesc(log10(heattmp2));axis xy image;colormap('hot');%caxis([0 0.3]);colorbar
xticks([5 10 15]);xticklabels([8 18 28])
yticks([5 10 15]);yticklabels([8 18 28])

[r,p] = corrcoef(PN_sample_set, PN_test_set);
% disp(['r = ' num2str(r(2)), ', p = ' num2str(p(2))])
title(['Fig. 2a: PN in repeated trials with new stimulus sets']);box off;xlabel('Fig. 2a: Numerical distance');ylabel('Ratio')
xlabel('PN_{sample}');ylabel('PN_{test}');colormap hot

indrand1 = randperm(length(PN_sample_set));
indrand2 = randperm(length(PN_sample_set));
disttmp_perm = (abs(number_sets(PN_sample_set(indrand1))-number_sets(PN_test_set(indrand2))));
disttmp = abs(number_sets(PN_sample_set)-number_sets(PN_test_set));

subplot(2,4,2)
edges = 0:2:31;
histogram(disttmp, edges, 'normalization', 'probability', 'EdgeColor', 'none', 'Facecolor', 'r')
hold on; histogram(disttmp_perm, edges, 'normalization', 'probability', 'edgecolor', 'none', 'Facecolor', [0.5 0.5 0.5])
plot([mean(disttmp), mean(disttmp)], [0 0.3], 'r--')
plot([mean(disttmp_perm), mean(disttmp_perm)], [0 0.3], 'k--')
p = ranksum(disttmp, disttmp_perm, 'tail', 'left'); box off
xlabel('PN_{original} - PN_{new}'); ylabel('Ratio'); title('Numerical distance')
legend({'Number neurons', 'Shuffled'})
% if issavefig; savefig([pathtmp '/Figs/2a']); end


%% PN distribution
PNdist_perm_tot = zeros(iter, 16); %PNdist_pret_tot = zeros(iter, 16);
NSnums = zeros(iter, 1);
for iterind = 1:iter
   
    units_PN = iter_units_PN{iterind};
   
    for jj = 1:length(number_sets)
        PNdist_perm_tot(iterind,jj) = sum(units_PN==jj)/sum(units_PN>0);
    end
    NSnums(iterind) = sum(units_PN>0);
end

load('PN_monkey') % PN distribution of monkey, from Nieder 2007

%figure; %set(gcf,'Visible', 'off')
subplot(2,4,3)
b1 = plot(number_sets, PN_monkey, 'g'); alpha (0.3); hold on;
b2 = shadedErrorBar(number_sets, mean(PNdist_perm_tot,1), std(PNdist_perm_tot,[],1), 'lineprops', 'r'); alpha (0.3);%b2.EdgeColor = 'none'; %/sum(PN_popul_RP));
% b3 = plot(number_sets, PN_pretrained); alpha (0.3)
xlabel('Preferred numerosity');ylabel('portion')
title('Fig. 2b: Preferred numerosity')
legend([b1, b2.mainLine], {'Monkey', 'Untrained'}); box off%legend boxoff; 
% if issavefig; savefig([pathtmp '/Figs/2b']); end


%% Average tuning curve

tuning_curve_sum = zeros(length(number_sets),length(number_sets));
sig_lin_per_tot = zeros(iter, 16);
R2_lin_per_tot = zeros(iter, 16);
sig_log_per_tot = zeros(iter, 16);
R2_log_per_tot = zeros(iter, 16);
for iterind = 1:iter
    % get tuning curve data
    tuning_curve = iter_tuning_curve_n{iterind};
    tuning_curve_sum = tuning_curve_sum+tuning_curve;
    
    % get Weber's law data
    TNCV_tmp = tuning_curve;
    xtmp = number_sets;
    [sig_lin_per, R2_lin_per] = Weberlaw(TNCV_tmp, xtmp);
    sig_lin_per_tot(iterind,:) = sig_lin_per;
    R2_lin_per_tot(iterind,:) = R2_lin_per;
    xtmp = log2(number_sets);
    [sig_log_per, R2_log_per] = Weberlaw(TNCV_tmp, xtmp);
    sig_log_per_tot(iterind,:) = sig_log_per;
    R2_log_per_tot(iterind,:) = R2_log_per;
end
tuning_curve_ave_per = tuning_curve_sum/iter;

load('coltmp', 'colortmp')
options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
subplot(2,4,5)
xtmp = number_sets;sigmas = zeros(1,length(number_sets)); R2 = zeros(1,length(number_sets)); TNtmp = tuning_curve_ave_per;
for ii = 1:length(number_sets)
    hh=plot(number_sets, TNtmp(ii,:), 'Color', colortmp(ii,:));
    ytmp = TNtmp(ii,:);
    if isnan(ytmp)
        sigmas(ii) = nan;
    else
%         f = fit(xtmp.', ytmp.', 'gauss1', options);
%         sigmas(ii) = f.c1/2; xfinetmp = 1:0.01:30;
%         tmp = f.a1*exp(-((xfinetmp-f.b1)/f.c1).^2);
%         ymean = nanmean(ytmp);
%         SStot = sum((ytmp-ymean).^2); %     SSres = sum((ytmp-tmp).^2); %     R2(ii) = 1-SSres./SStot;
    end
    hold on
end
xlabel('Numerosity');ylabel('Normalized response (A.U.)'); title('Fig. 2c: Linear scale'); box off

subplot(2,4,6) %set(gcf,'Visible', 'off'); hold on
xtmp = log2(number_sets);sigmas = zeros(1,length(number_sets)); R2 = zeros(1,length(number_sets)); TNtmp = tuning_curve_ave_per;
for ii = 1:length(number_sets)
    
    if ii ==1
    btmp = ii;
    else
    btmp = (ii-1)*2;
    end
    options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 log2(30) 100]);
    
   % options = fitoptions('gauss1', 'Lower', [0 btmp 0], 'Upper', [1.5 btmp 100]);
    
    hh=plot(xtmp, TNtmp(ii,:), 'Color', colortmp(ii,:)); hold on
    ytmp = TNtmp(ii,:);
    if isnan(ytmp)
        sigmas(ii) = nan;
    else
    end
end
xlabel('Numerosity');ylabel('Normalized response (A.U.)'); title('Fig. 2c: Log scale'); box off
xticks([0:5]);xticklabels({'1', '2', '4', '8', '16', '32'})

subplot(2,4,7); hold on
bar([1], [mean(R2_lin_per_tot(:))], 'FaceColor', [249 119 85]/255)
bar([2], [mean(R2_log_per_tot(:))], 'FaceColor', [44 169 111]/255)
errorbar([1,2], [mean(R2_lin_per_tot(:)), mean(R2_log_per_tot(:))], [std(R2_lin_per_tot(:)), std(R2_log_per_tot(:))], 'linestyle', 'none')
box off; xticks([1,2]); xticklabels({'Linear', 'Log'});ylabel('Goodness of fit, r^2'); title('Fig. 2c: Goodness of fit');
% if issavefig; savefig([pathtmp '/Figs/2c']); end

%% Weber-Fechner law
load('Data_Nieder2007') % tuning width data for monkey, from Nieder 2007

%figure; hold on; %set(gcf,'Visible', 'on'); hold on
subplot(2,4,[8]); hold on
sig_lin_per = mean(sig_lin_per_tot, 1);
sig_log_per = mean(sig_log_per_tot, 1);
stdlin = std(sig_lin_per_tot, [],1);
stdlog = std(sig_log_per_tot, [],1);
s1 = scatter(number_sets, sig_lin_per, 'fill', 'MarkerFacecolor', [249 119 85]/255);
p1 = polyfit(number_sets, sig_lin_per, 1);
%errorbar(number_sets, sig_lin_per, stdlin, 'LineStyle', 'none');
plot(number_sets, p1(1)*number_sets+p1(2), 'color', [249 119 85]/255)
s2 = scatter(number_sets, sig_log_per, 'fill', 'MarkerFaceColor', [44 169 111]/255);
%errorbar(number_sets, sig_log_per, stdlog, 'LineStyle', 'none')
p2 = polyfit(number_sets, sig_log_per, 1);
plot(number_sets, p2(1)*number_sets+p2(2), 'color', [44 169 111]/255)
xlabel('Numerosity');ylabel('Sigma of Gaussian fit');ylim([0 16])
[r1,pv1] = corrcoef(number_sets, sig_lin_per);
[r2,pv2] = corrcoef(number_sets, sig_log_per);
title('Fig. 2d: Weber-Fechner law')
% s3 = scatter(x_axis, sig_linear, 'k'); s4 = scatter(x_axis, sig_log, 'r'); title('Fig. 2d: Weber-Fechner law')
%legend([s1, s2, s3, s4], {'Init (lin)', 'Init (log)', 'Monkey (lin)', 'Monkey (log)'}, ...
legend([s1, s2], {'Linear', 'Log'}, ...
    'Location', 'best');%legend boxoff
% if issavefig; savefig([pathtmp '/Figs/2d']); end


%% ------------------------------------------------------------------ Figure 3

%% Fig 3b. Task performance
figure('units','normalized','outerposition',[0 0.25 0.5 0.5]);
subplot(2,3,[1,4])
kk = 1;
setindtmp = 1;
iter = 1;
tmp1 = 0; tmp3 = 0; tmp4 = 0;
for iterind = 1:iter
    congincong_dat = iter_congincong_dat_original{1,1,iterind};
    answer_mat = congincong_dat{6,setindtmp};
    
    % performance using pretrained-number, initialized-number, initialized-nonselective
    perftmp_cong = zeros(3,averageiter);
    for ii = 1:averageiter
        anstmp = answer_mat{1,ii,kk};
        perftmp_cong(1,ii) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
        anstmp = answer_mat{2,ii,kk};
        perftmp_cong(2,ii) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
        anstmp = answer_mat{3,ii,kk};
        perftmp_cong(3,ii) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
        
    end
    
    % performance using raw image
    anstmp = answer_mat{5,1,1};
    tmp0(iterind) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
    
    tmp1(iterind) = mean(perftmp_cong(1,:));
    tmp3(iterind) = mean(perftmp_cong(2,:));
    tmp4(iterind) = mean(perftmp_cong(3,:));
    
end

hold on
bar(1, 100*mean(tmp3), 0.25, 'FaceColor', [254 67 78]/255);
errorbar([1], 100*[mean(tmp3)], 100*[std(tmp3)], 'k','LineStyle', 'none')
bar(2, 100*mean(tmp4), 0.25, 'FaceColor', 'w', 'EdgeColor', [254 67 78]/255)
errorbar([2], 100*[mean(tmp4)], 100*[std(tmp4)], 'k','LineStyle', 'none')
bar(3, 100*mean(tmp0), 0.25, 'FaceColor', [0.5 0.5 0.5])
errorbar([3], 100*[mean(tmp0)], 100*[std(tmp0)], 'k','LineStyle', 'none')
%p(setindtmp) = ranksum(tmp0, tmp3, 'tail', 'left');
ptmp12 = ranksum(tmp3, tmp4, 'tail', 'right');
ptmp13 = ranksum(tmp3, tmp0, 'tail', 'right');
ylim([40 100]);
title(['Fig 3b: Task performance']);ylabel('Performance (%)');%ylim([40 80]);yticks([40, 60, 80])
xticks([1,2,3]); xticklabels({'Number neuron response', 'Non-selective neuron response', 'Image pixel information'})
xtickangle(45)
% if issavefig; savefig([pathtmp '/Figs/3b']); end

%% Figure 3c-d. Confusing matrix and numerical distance effect

correctness_inputnumerosity_dat = cell(iter, averageiter, length(sampling_numbers), 3); % Reorganize data
for iterind = 1:iter
    tmp = iter_correctness_inputnumerosity_original{iterind};
    for ii = 1:averageiter
        for kk = 1:length(sampling_numbers)
            correctness_inputnumerosity_dat{iterind, ii, kk, 1} = tmp{ii,kk,1};
            correctness_inputnumerosity_dat{iterind, ii, kk, 2} = tmp{ii,kk,2};
            correctness_inputnumerosity_dat{iterind, ii, kk, 3} = tmp{ii,kk,3};
        end
    end
end

Ntmp = length(number_sets)*iterforeachN_val*averageiter;
[correctratiomat, correcttotdist, correctratiodist] = getPerformanceforeachcombinationofnumerosities(Ntmp, sampling_numbers, ...
    correctness_inputnumerosity_dat, iter, averageiter, iterforeachN_val, number_sets); % All sets used

subplot(2,3,2)
imagesc(correctratiomat);axis image;xlabel('Numerosity of stimulus A');ylabel('Numerosity of stimulus B');axis xy;colormap('hot');colorbar
title('Fig. 3c: Performance for each combination')
% if issavefig; savefig([pathtmp '/Figs/3c']); end

p = signrank(correcttotdist(:,2), 0.5*ones(1,length(correcttotdist(:,2))), 'tail', 'right');
subplot(2,3,3);hold on
e1=shadedErrorBar(2:2:28, 100*nanmean(correcttotdist(:, 2:2:28), 1), 100*nanstd(correcttotdist(:, 2:2:28), [], 1), 'lineprops', 'r');
xlabel('Numerical distance');ylabel('Performance (%)');
% ptmp = signrank(correcttotdist2(:,2), 0.5+zeros(1,length(correcttotdist2)), 'tail', 'right');
title('Fig. 3d: Numerical distance effect');%legend([e1.mainLine, e2.mainLine], {'All sets used', 'Set 2 only'})
% if issavefig; savefig([pathtmp '/Figs/3d']); end


%% Figure 3e. Number neurons response

stdind = 1; verind = 1;
correct_sets = zeros(iter, length(ND));
wrong_sets = zeros(iter, length(ND));
for iterind = 1:iter
    correct_dat = iter_correct_original{stdind, verind, iterind};
    correct_sets(iterind, :) = correct_dat{1};
    wrong_dat = iter_wrong_original{stdind, verind, iterind};
    wrong_sets(iterind, :) = wrong_dat{1};
    
end

subplot(2,3,5)
tmp1 = nanmean(correct_sets, 1); std1 = nanstd(correct_sets, [],1);
l1 = shadedErrorBar(ND(1:2:61), tmp1(1:2:61), std1(1:2:61), 'Lineprops', 'b');hold on
tmp2 = nanmean(wrong_sets, 1); std2 = nanstd(wrong_sets, [],1);
l2 = shadedErrorBar(ND(1:2:61), tmp2(1:2:61),std2(1:2:61), 'Lineprops', 'k');
xlabel('Numerical distance');ylabel('Normalized response (A.U.)');
ptmp = ranksum(correct_sets(:, 31), wrong_sets(:, 31), 'tail', 'right');
legend([l1.mainLine l2.mainLine], {'Correct', 'Incorrect'}, 'Location', 'best')
title('Fig. 3e: Number neurons response')
subplot(2,3,6)
hold on;
bar([1], mean([correct_sets(:,31)], 1), 'Facecolor', [56 80 248]/255)
bar([2], mean([wrong_sets(:,31)], 1), 'Facecolor', [102, 102, 102]/255)
errorbar([1,2], mean([correct_sets(:,31), wrong_sets(:,31)], 1), std([correct_sets(:,31), wrong_sets(:,31)], [], 1), 'linestyle', 'none')
xticks([1,2]);xticklabels({'Correct', 'Incorrect'})
% if issavefig; savefig([pathtmp '/Figs/3e']); end

%% --------------------------------------------- Figure 4

%% Figure 4a. Correlation between numerosity and total area
[training_areas,  training_sets, training_Nums] = ...
    getimgstat(image_sets_standard, image_sets_control1, image_sets_control2, number_sets);

figure('units','normalized','outerposition',[0 0.25 1 0.5]); 
subplot(1,4,1);hold on
training_Nums_set2 = training_Nums(1,image_iter+1:image_iter*2,:);
training_Nums_set1 = training_Nums(1,[1:image_iter],:);
training_Nums_set3 = training_Nums(1,[image_iter*2+1:image_iter*3],:);
training_areas_set2 = training_areas(1,image_iter+1:image_iter*2,:);
training_areas_set1 = training_areas(1,[1:image_iter],:);
training_areas_set3 = training_areas(1,[image_iter*2+1:image_iter*3],:);
c1 = scatter( 0.1*randn(1,length(training_Nums_set1(:)))'+number_sets(training_Nums_set1(:))', training_areas_set1(:), '.','Markeredgecolor', [12 159 81]/255);
c2 = scatter( 0.1*randn(1,length(training_Nums_set3(:)))'+number_sets(training_Nums_set3(:))', training_areas_set3(:), '.','Markeredgecolor', [0 113 178]/255);
c3 = scatter( 0.1*randn(1,length(training_Nums_set2(:)))'+number_sets(training_Nums_set2(:))', training_areas_set2(:), '.','Markeredgecolor', [227 156 83]/255);
legend([c1, c2, c3], {'Same dot size', 'Same convex hull', 'Same total area'}, 'location', 'best')
xlabel('Numerosity');ylabel('Total area (pixel^2)');yticks([0:1000:5000])
[r,p] = corrcoef(training_Nums(:), training_areas(:));
title('Fig. 4a: Correlation between numerosity and total area')
% if issavefig; savefig([pathtmp '/Figs/4a']); end


%% Figures 4d, f

for setindtmp = [2,4,3]
    subplot(1,4,setindtmp)
    tmp1 = 0; tmp3 = 0; tmp4 = 0;
    for iterind = 1:iter
        congincong_dat = iter_congincong_dat{1,1,iterind};
        answer_mat = congincong_dat{6,setindtmp};
        areas_val = congincong_dat{1,setindtmp};
        denss_val = congincong_dat{2,setindtmp};
        dotszs_val = congincong_dat{3,setindtmp};
        sets_val = congincong_dat{4,setindtmp};
        nums_val = congincong_dat{5,setindtmp};
        
        % Statistics of testing stimulus
        areaS = squeeze(areas_val(1,:,:)); areaS = areaS(:);
        areaT = squeeze(areas_val(2,:,:)); areaT = areaT(:);
        numS = squeeze(nums_val(1,:,:)); numS = numS(:);
        numT = squeeze(nums_val(2,:,:)); numT = numT(:);
        setS = squeeze(sets_val(1,:,:)); setS = setS(:);
        setT = squeeze(sets_val(2,:,:)); setT = setT(:);
        densS = squeeze(denss_val(1,:,:)); densS = densS(:);
        densT = squeeze(denss_val(2,:,:)); densT = densT(:);
        dotszS = squeeze(dotszs_val(1,:,:)); dotszS = dotszS(:);
        dotszT = squeeze(dotszs_val(2,:,:)); dotszT = dotszT(:);
        
        congruentind_sz = ((areaS>=areaT & numS>numT) | (areaS<=areaT & numS<numT)) | (setS==setT); % just because same area is sometimes problematic in matlab..
        incongruentind_sz = ~congruentind_sz;% & ~(setS==setT);
        congruentind_dens = (densS<=densT & numS>numT) | (densS>=densT & numS<numT);% & (setS~=setT);
        incongruentind_dens = ~congruentind_dens;
        congruentind_dotsz = (dotszS>=dotszT & numS>numT) | (dotszS<=dotszT & numS<numT);% & (setS~=setT);
        incongruentind_dotsz = ~congruentind_dotsz;
        
        congruentind_all = congruentind_sz & congruentind_dens & congruentind_dotsz;
        congruentind_inverse = ~(incongruentind_sz & incongruentind_dens & incongruentind_dotsz);
        incongruentind_all = incongruentind_sz & incongruentind_dens & incongruentind_dotsz;
        
        if setindtmp == 1
            indused_tr = congruentind_sz;
        elseif setindtmp == 2
            indused_tr = congruentind_sz;
        elseif setindtmp ==3
            indused_tr = congruentind_dens;
        elseif setindtmp ==4
            indused_tr = congruentind_dotsz;
        end
        
        perftmp_cong = zeros(3,averageiter);
        perftmp_incong = zeros(3,averageiter);
        for ii = 1:averageiter
            
            anstmp = answer_mat{2,ii,kk};
            anstmp = anstmp(~indused_tr);
            perftmp_incong(2,ii) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
            anstmp = answer_mat{3,ii,kk};
            anstmp = anstmp(~indused_tr);
            perftmp_incong(3,ii) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
            
        end
        anstmp = answer_mat{5,1,1};
        anstmp = anstmp(congruentind_sz);
        tmp0(iterind) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
        anstmp = answer_mat{5,1,1};
        anstmp = anstmp(incongruentind_sz);
        perf_rawimg(iterind) = sum(real(anstmp)==imag(anstmp))/length(anstmp);
        tmp3_incong(iterind) = mean(perftmp_incong(2,:));
        tmp4_incong(iterind) = mean(perftmp_incong(3,:));
        
    end
    
    hold on
    bar(1, 100*mean(tmp3_incong), 0.25, 'FaceColor', [254 67 78]/255);
    errorbar(1, 100*[mean(tmp3_incong)], 100*[std(tmp3_incong)], 'k','LineStyle', 'none')
    
    bar(2, 100*mean(tmp4_incong), 0.25, 'FaceColor', 'w', 'EdgeColor', [254 67 78]/255)
    errorbar(2, 100*[mean(tmp4_incong)], 100*[std(tmp4_incong)], 'k','LineStyle', 'none')
    bar(3, 100*mean(perf_rawimg), 0.25, 'FaceColor', [0.5 0.5 0.5])
    errorbar(3, 100*[mean(perf_rawimg)], 100*[std(perf_rawimg)], 'k','LineStyle', 'none')
    ylim([20 100])
    xticks([1,2,3]);xticklabels({'Number neuron response', 'Non-selective neuron response', 'Image pixel information'})
    xtickangle(45);
    controlled_features = {'', 'Area', 'Density', 'Dot size'};
    title(['Figs. 4d, f: ' controlled_features{setindtmp} ' control']);ylabel('Performance (%)');%ylim([40 80]);yticks([40, 60, 80])
    plot([0 4], [50 50], 'k--')
end
% sgtitle(['Figure 4: Task performance'])
% if issavefig; savefig([pathtmp '/Figs/3d,f']); end
