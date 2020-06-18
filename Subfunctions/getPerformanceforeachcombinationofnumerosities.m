function [correctratiomat, correcttotdist, correctratiodist] = getPerformanceforeachcombinationofnumerosities(Ntmp, sampling_numbers, ...
    correctness_inputnumerosity_dat, iter, averageiter, iterforeachN_val, number_sets)


indcorrecttot = []; numsampletot = []; numtesttot = [];
for iterind = 1:iter
    for ii = 1:averageiter
        for kk = 1:length(sampling_numbers)
            indcorrecttot = [indcorrecttot; correctness_inputnumerosity_dat{iterind, ii, kk, 1}];
            numsampletot = [numsampletot; correctness_inputnumerosity_dat{iterind, ii, kk, 2}];
            numtesttot = [numtesttot; correctness_inputnumerosity_dat{iterind, ii, kk, 3}];        
        end
    end
end

correctratiomat = zeros(16,16); % Heat map of correct ratio
for indsample = 1:16
    ind1 = numsampletot == indsample;
    for indtest = 1:16
        ind2 = numtesttot == indtest;
        indtmp = (ind1 & ind2);
        correcttmp = indcorrecttot(indtmp);
        correctratiotmp = sum(correcttmp)/length(correcttmp);
        correctratiomat(indtest, indsample) = correctratiotmp;
    end
end

correctratiodist = zeros(iter, 16, 16);
correcttotdist  = zeros(iter,30);

for iterind = 1:iter
    indtmp = Ntmp*(iterind-1)+1:Ntmp*iterind;
    indcorrecttmp = indcorrecttot(indtmp);
    numsampletmp = numsampletot(indtmp);
    numtesttmp = numtesttot(indtmp);
    
    for indsample = 1:16
    ind1 = numsampletmp == indsample;
    for indtest = 1:16
        ind2 = numtesttmp == indtest;
        indtmp = (ind1 & ind2);
        correcttmp = indcorrecttmp(indtmp);
        correctratiotmp = sum(correcttmp)/length(correcttmp);
        correctratiodist(iterind, indtest, indsample) = correctratiotmp;    
    end
    end
    tmp1 = 2*(numsampletmp-1); tmp1(tmp1==0) = 1;
    tmp2 = 2*(numtesttmp-1); tmp2(tmp2==0) = 1;
    difftesttmp = (tmp1-tmp2);
    for ii = 1:size(correcttotdist,2)
    indtmp = difftesttmp ==ii;
    correcttmp = indcorrecttmp(indtmp);
    correctratiotmp = sum(correcttmp)/length(correcttmp);
    correcttotdist(iterind, ii) = correctratiotmp;
    end    
end