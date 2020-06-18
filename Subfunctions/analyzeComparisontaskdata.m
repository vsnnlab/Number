function [performance_pret_perm_sets, correctness_inputnumerosity_dat, correct_sets, wrong_sets] ...
    = analyzeComparisontaskdata(iter, pathtmp, generatecomparisondata, ...
    sampling_numbers, averageiter, ND, number_sets, savedir)
if generatecomparisondata

correct_sets = zeros(iter, length(ND));
wrong_sets = zeros(iter, length(ND));
performance_pret_perm_sets = zeros(iter, 4);
correctness_inputnumerosity_dat = cell(iter, averageiter, length(sampling_numbers),3);
 
for iterind = 1:iter
   
    load([savedir '\Data_for_Comparisontask_iter_' num2str(iterind)])
     
    performance_pret_perm = zeros(4, averageiter, length(sampling_numbers)); % dim1: pret, perm, controlpret, controlperm
    tuning_curves_correct_set = cell(averageiter, length(sampling_numbers));
    tuning_curves_wrong_set = cell(averageiter, length(sampling_numbers));
    
    for ii = 1:averageiter
        for kk = 1:length(sampling_numbers)
            sampling_unitsN = sampling_numbers(kk);
            sampleind_pret = datasample(1:length(NSind_pretrained), sampling_unitsN, 'Replace', false);
            sampleind_perm = datasample(1:length(NSind_permuted), sampling_unitsN, 'Replace', false);
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
            ind_corr = (label==YVal_pm);
            
            tmp = units_PN_permuted(units_PN_permuted>0);
            PNlabel = tmp(sampleind_perm);
            tmp = label_val_perm{1};
            numlabel_sample = squeeze(tmp(1,:,:)); numlabel_sample = numlabel_sample(:);
            numlabel_test = squeeze(tmp(2,:,:)); numlabel_test = numlabel_test(:);
            correctness_inputnumerosity_dat{iterind, ii, kk,1} = ind_corr;
            correctness_inputnumerosity_dat{iterind, ii, kk,2} = numlabel_sample;
            correctness_inputnumerosity_dat{iterind, ii, kk,3} = numlabel_test;
                        
            tuning_curves_correct = zeros(sampling_unitsN, length(ND));
            tuning_curves_wrong = zeros(sampling_unitsN, length(ND));
            for ij = 1:sampling_unitsN % # of units
                PNtmp = number_sets(PNlabel(ij));
                tmp = resp_permuted_val_part{1};
                resp_sampleunit = squeeze(tmp(:,:,ij))';
                resp_sampleunit = resp_sampleunit(:);
                tmp = resp_permuted_val_part{2};
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
    
    %% 2J drawing
    aveTNCV_correct = zeros(length(sampling_numbers),averageiter, length(ND));
    aveTNCV_wrong = zeros(length(sampling_numbers),averageiter, length(ND));
    for kk = 1:length(sampling_numbers)
        %     subplot(1,length(sampling_numbers), kk)
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
        %     plot(NDs(1:2:61), correct(1:2:61));hold on
        %     plot(NDs(1:2:61), wrong(1:2:61));
        %     errorbar(NDs(1:2:61), correct(1:2:61), std1(1:2:61));hold on
        %     errorbar(NDs(1:2:61), wrong(1:2:61),std2(1:2:61));
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
    
    correct_sets(iterind, :) = correct;
    wrong_sets(iterind, :) = wrong;
    performance_pret_perm_sets(iterind, 1) = mean(performance_pret_permtmp(1,:));
    performance_pret_perm_sets(iterind, 2) = mean(performance_pret_permtmp(2,:));
    performance_pret_perm_sets(iterind, 3) = mean(performance_pret_permtmp(3,:));
    performance_pret_perm_sets(iterind, 4) = mean(performance_pret_permtmp(4,:));
    
end

save([savedir '\2h_performancematrix'], 'performance_pret_perm_sets');
save([savedir '\S3d_correctness_inputnumerosity_data_set123'], 'correctness_inputnumerosity_dat');
end
end