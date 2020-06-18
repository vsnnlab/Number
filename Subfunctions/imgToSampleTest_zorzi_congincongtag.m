function [image_sets_sample, image_sets_test, labels,  training_areas,  ...
    training_circum, training_dotsz, training_sets, training_Nums] = ...
    imgToSampleTest_zorzi_congincongtag(image_sets, image_sets_stat, iterforeachN, number_sets)


% for ii = 1:8
%     image_sets(:,:,50*(ii-1)+(1:50), :) = image_sets(:,:,50*(ii-1)+(1:50), :)+1i*ii;
% end

image_sets_sample = zeros(size(image_sets,1), size(image_sets,2), ...
    iterforeachN, length(number_sets));
image_sets_test = zeros(size(image_sets,1), size(image_sets,2), ...
    iterforeachN, length(number_sets));


training_Nums = zeros(2,iterforeachN, length(number_sets));
training_labels = zeros(iterforeachN, length(number_sets));
training_areas = zeros(2,iterforeachN,length(number_sets));
training_circum = zeros(2,iterforeachN,length(number_sets));
training_dotsz = zeros(2,iterforeachN,length(number_sets));
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
        
        training_areas(1,jj,ii) = image_sets_stat(1,sampleind,  sampleN);
        training_areas(2,jj,ii) = image_sets_stat(1,testind,  testN);
        
        
%         tmp1 = imag(image_sets(:,:,sampleind, sampleN));
%         tmp2 = imag(image_sets(:,:,testind, testN));
        training_sets(1,jj,ii) = image_sets_stat(4,sampleind,  sampleN);
        training_sets(2,jj,ii) = image_sets_stat(4,testind,  testN);
        
        training_circum(1,jj,ii) = image_sets_stat(2,sampleind,  sampleN);
        training_circum(2,jj,ii) = image_sets_stat(2,testind, testN);
        training_dotsz(1,jj,ii) = image_sets_stat(3,sampleind,  sampleN);
        training_dotsz(2,jj,ii) = image_sets_stat(3,testind, testN);
        
    end
    %disp(kk);
end

labels = cell(1,2);
labels{1} = training_Nums;
labels{2} = training_labels;


end