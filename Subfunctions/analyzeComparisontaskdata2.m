function [performance_pret_perm_sets, correctness_inputnumerosity_dat, correct_sets, wrong_sets] ...
    = analyzeComparisontaskdata2(iter, sampling_numbers, averageiter, ND,  savedir)


%% SVM test using set2, indiv response

correct_sets = zeros(iter, length(ND));
wrong_sets = zeros(iter, length(ND));
performance_pret_perm_sets = zeros(iter, 4);
correctness_inputnumerosity_dat = cell(iter, averageiter, length(sampling_numbers),3);

for iterind = 1:iter
    
    load([savedir '\Data_for_Comparisontask_onlyset2_iter_' num2str(iterind)])
    
    performance_pret_perm = zeros(4, averageiter, length(sampling_numbers)); % dim1: pret, perm, controlpret, controlperm
    tuning_curves_correct_set = cell(averageiter, length(sampling_numbers));
    tuning_curves_wrong_set = cell(averageiter, length(sampling_numbers));
    %   response_NS_sets_fortask = cell(iter, 2, 2); % 3-dim : iteration, pretrained/permuted, test/validation
    for ii = 1:averageiter
        for kk = 1:length(sampling_numbers)
            sampling_unitsN = sampling_numbers(kk);
            %disp(ii);
            sampleind_pret = datasample(NSind_pretrained, sampling_unitsN, 'Replace', false);
            sampleind_perm = datasample(NSind_permuted, sampling_unitsN, 'Replace', false);
            %         NSind_pretrained2 = datasample(NSind_pretrained, sampling_unitsN, 'Replace', false);
            %         NSind_permuted2 = datasample(NSind_permuted, sampling_unitsN, 'Replace', false);
            resp_pretrained_part = cell(1,2); resp_pretrained_val_part = cell(1,2);
            resp_permuted_part = cell(1,2); resp_permuted_val_part = cell(1,2);
            tmp = resp_pretrained{1}; tmp = tmp(:,:,sampleind_pret); resp_pretrained_part{1} = tmp;
            tmp = resp_pretrained{2}; tmp = tmp(:,:,sampleind_pret); resp_pretrained_part{2} = tmp;
            tmp = resp_pretrained_val{1}; tmp = tmp(:,:,sampleind_pret); resp_pretrained_val_part{1} = tmp;
            tmp = resp_pretrained_val{2}; tmp = tmp(:,:,sampleind_pret); resp_pretrained_val_part{2} = tmp;
            tmp = resp_permuted{1}; tmp = tmp(:,:,sampleind_perm); resp_permuted_part{1} = tmp;
            tmp = resp_permuted{2}; tmp = tmp(:,:,sampleind_perm); resp_permuted_part{2} = tmp;
            tmp = resp_permuted_val{1}; tmp = tmp(:,:,sampleind_perm); resp_permuted_val_part{1} = tmp;
            tmp = resp_permuted_val{2}; tmp = tmp(:,:,sampleind_perm); resp_permuted_val_part{2} = tmp;
            
            
            [XTrain, YTrain, labelsTrain] = getdataformat2(resp_pretrained_part, label_test); %% get training dataset
            [XVal, YVal, labelsVal] = getdataformat2(resp_pretrained_val_part, label_val); %% get validation dataset
            ntmp = size(XTrain, 1);idx = randperm(ntmp);XTrain = XTrain(idx, :, :, :);
            Mdl_ctr = fitcsvm(squeeze(XTrain)', YTrain);
            [label, score] = predict(Mdl_ctr, squeeze(XVal)');
            correct_ratio = length(find(label==YVal))/length(YVal);
            performance_pret_perm(3, ii, kk) = correct_ratio;
            
            [XTrain_pm, YTrain_pm, labelsTrain_pm] = getdataformat2(resp_permuted_part, label_test_perm); %% get training dataset
            [XVal_pm, YVal_pm, labelsVal_pm] = getdataformat2(resp_permuted_val_part, label_val_perm); %% get validation dataset
            ntmp = size(XTrain_pm, 1); idx = randperm(ntmp); XTrain_pm = XTrain_pm(idx, :, :, :);
            Mdl_pm = fitcsvm(squeeze(XTrain_pm)', YTrain_pm);
            [label, score] = predict(Mdl_pm, squeeze(XVal_pm)');
            correct_ratio_pm = length(find(label==YVal_pm))/length(YVal_pm);
            performance_pret_perm(4, ii, kk) = correct_ratio_pm;
            
            [XTrain, YTrain, labelsTrain] = getdataformat2(resp_pretrained_part, label_test); %% get training dataset
            [XVal, YVal, labelsVal] = getdataformat2(resp_pretrained_val_part, label_val); %% get validation dataset
            Mdl = fitcsvm(squeeze(XTrain)', YTrain);
            [label, score] = predict(Mdl, squeeze(XVal)');
            correct_ratio = length(find(label==YVal))/length(YVal);
            performance_pret_perm(1, ii, kk) = correct_ratio;
            
            [XTrain_pm, YTrain_pm, labelsTrain_pm] = getdataformat2(resp_permuted_part, label_test_perm); %% get training dataset
            [XVal_pm, YVal_pm, labelsVal_pm] = getdataformat2(resp_permuted_val_part, label_val_perm); %% get validation dataset
            Mdl_pm = fitcsvm(squeeze(XTrain_pm)', YTrain_pm);
            [label, score] = predict(Mdl_pm, squeeze(XVal_pm)');
            correct_ratio_pm = length(find(label==YVal_pm))/length(YVal_pm);
            performance_pret_perm(2, ii, kk) = correct_ratio_pm;
            
            %% compare correct/incorrect trial
            indcorrect = find(label==YVal_pm);
            indwrong = find(~(label==YVal_pm));
            indcorr = (label==YVal_pm);
            PNlabel = units_PN_permuted(sampleind_perm);
            tmp = label_val_perm{1};
            numlabel_sample = squeeze(tmp(1,:,:)); numlabel_sample = numlabel_sample(:);
            numlabel_test = squeeze(tmp(2,:,:)); numlabel_test = numlabel_test(:);
            correctness_inputnumerosity_dat{iterind, ii, kk,1} = indcorr;
            correctness_inputnumerosity_dat{iterind, ii, kk,2} = numlabel_sample;
            correctness_inputnumerosity_dat{iterind, ii, kk,3} = numlabel_test;
            
        end
    end
    
    %% 2I drawing
    %figure
    for kk = 1:length(sampling_numbers)
        %subplot(1,length(sampling_numbers), kk)
        performance_pret_permtmp = squeeze(performance_pret_perm(:,:,kk));
        
        % bar([1,2], [mean(performance_pret_permtmp(1,:)),mean(performance_pret_permtmp(2,:))])
        % hold on;errorbar([1,2], [mean(performance_pret_permtmp(1,:)),mean(performance_pret_permtmp(2,:))], [std(performance_pret_permtmp(1,:)),std(performance_pret_permtmp(2,:))])
        % ptmp = ranksum(performance_pret_permtmp(1,:), performance_pret_permtmp(2,:));
        % title(['p = ' num2str(ptmp) ', used N = ' num2str(sampling_numbers(kk))]);ylim([0.4 1])
        % plot([0 3], [0.5 0.5], 'r')
    end
    % ' 1:pret, 2:perm, errorbar represents std'
    
%     correct_sets(iterind, :) = correct;
%     wrong_sets(iterind, :) = wrong;
    performance_pret_perm_sets(iterind, 1) = mean(performance_pret_permtmp(1,:));
    performance_pret_perm_sets(iterind, 2) = mean(performance_pret_permtmp(2,:));
    performance_pret_perm_sets(iterind, 3) = mean(performance_pret_permtmp(3,:));
    performance_pret_perm_sets(iterind, 4) = mean(performance_pret_permtmp(4,:));
end

end