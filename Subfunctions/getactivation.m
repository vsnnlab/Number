function [response_tot] = getactivation(net, LOI, image_sets)


%% get size of the image
imtmp = squeeze(image_sets(:,:, 1,1));
imtmpp = imtmp*255;
im = cat(3, imtmpp , imtmpp , imtmpp );
% imgSize = size(im);
% imgSize = imgSize(1:2);
act = activations(net,im,LOI);
acttmp = act(:);
N_neurons = length(acttmp);



number_N = size(image_sets,4);
image_iter = size(image_sets,3);


image_sets2 = reshape(image_sets, [size(image_sets,1), size(image_sets,2), 1,size(image_sets,3)*size(image_sets,4)]);
image_sets3 = 255*cat(3, image_sets2, image_sets2, image_sets2);


acttmp = activations(net, image_sets3, LOI);
%acttmp = activations(net, image_sets3, LOI, 'Executionenvironment', 'gpu');

acttmp2 = reshape(acttmp, [size(acttmp, 1)*size(acttmp, 2)*size(acttmp,3), size(image_sets, 3), size(image_sets, 4)]);

response_tot = zeros(number_N, image_iter, N_neurons);
for ii = 1:number_N
for jj = 1:image_iter

acttmp3 = acttmp2(:,jj,ii);
response_tot(ii,jj,:) = acttmp3;
end
end


end