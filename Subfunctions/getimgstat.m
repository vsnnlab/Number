function [training_areas,  training_sets, training_Nums] = ...
    getimgstat(image_sets_standard, image_sets_control1, image_sets_control2, number_sets)



image_sets = cat(3,image_sets_standard, image_sets_control1, image_sets_control2);


image_sets_sample = zeros(size(image_sets_standard,1), size(image_sets_standard,2), ...
    size(image_sets, 3), length(number_sets));


training_Nums = zeros(1,size(image_sets, 3), length(number_sets));
training_labels = zeros(size(image_sets, 3), length(number_sets));
training_areas = zeros(1,size(image_sets, 3),length(number_sets));
training_sets = zeros(1,size(image_sets, 3),length(number_sets));
for ii = 1:length(number_sets)
    
    sampleN = ii;
    
    for jj = 1:size(image_sets, 3)
       
        sampleind = jj;
        
        
        image_sets_sample(:,:,jj,ii) = real(image_sets(:,:,sampleind, sampleN));
        
        training_Nums(1,jj,ii) = sampleN;
        
        
        training_areas(1,jj,ii) = sum(sum(squeeze(image_sets_sample(:,:,jj,ii))));
        
        
        tmp1 = imag(image_sets(:,:,sampleind, sampleN));
        training_sets(1,jj,ii) = tmp1(1);
        
        
    end
    %disp(kk);
end



end