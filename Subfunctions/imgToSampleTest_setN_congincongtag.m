function [image_sets_sample, image_sets_test, labels,  training_areas,  training_sets, training_Nums] = ...
    imgToSampleTest_setN_congincongtag(image_sets_standard, image_sets_control1, image_sets_control2, setNo, iterforeachN, number_sets)



image_tot = cat(5,image_sets_standard, image_sets_control1, image_sets_control2);

image_sets = [];
for ss = 1:length(setNo)
    setind = setNo(ss); % set used in the comparison task
    image_sets = cat(3, image_sets, squeeze(image_tot(:,:,:,:,setind))+1i*setind  );
end

image_sets_sample = zeros(size(image_sets_standard,1), size(image_sets_standard,2), ...
    iterforeachN, length(number_sets));
image_sets_test = zeros(size(image_sets_standard,1), size(image_sets_standard,2), ...
    iterforeachN, length(number_sets));


training_Nums = zeros(2,iterforeachN, length(number_sets));
training_labels = zeros(iterforeachN, length(number_sets));
training_areas = zeros(2,iterforeachN,length(number_sets));
training_sets = zeros(2,iterforeachN,length(number_sets));
for ii = 1:length(number_sets)
    
    sampleN = ii;
    
    number_test_cand = 1:length(number_sets);
    number_test_cand(number_test_cand == ii) = [];
    for jj = 1:iterforeachN
        testN = datasample(number_test_cand,1);
        
        sampleind = randi(size(image_sets, 3));
        
        testind = randi(size(image_sets, 3));
        
        image_sets_sample(:,:,jj,ii) = real(image_sets(:,:,sampleind, sampleN));
        image_sets_test(:,:,jj,ii) = real(image_sets(:,:,testind, testN));
        training_Nums(1,jj,ii) = sampleN;
        training_Nums(2,jj,ii) = testN;
        training_labels(jj,ii) = (training_Nums(2,jj,ii)-training_Nums(1,jj,ii))>0;
        
        training_areas(1,jj,ii) = sum(sum(squeeze(image_sets_sample(:,:,jj,ii))));
        training_areas(2,jj,ii) = sum(sum(squeeze(image_sets_test(:,:,jj,ii))));
        
        
        tmp1 = imag(image_sets(:,:,sampleind, sampleN));
        tmp2 = imag(image_sets(:,:,testind, testN));
        training_sets(1,jj,ii) = tmp1(1);
        training_sets(2,jj,ii) = tmp2(1);
        
        
    end
    %disp(kk);
end

labels = cell(1,2);
labels{1} = training_Nums;
labels{2} = training_labels;


end