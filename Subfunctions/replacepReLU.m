function net_test = replacepReLU(net, change_layers_ind, scalepReLU)
net_test = net;
net_tmp = net_test.saveobj;
for ind_tl = 1:length(change_layers_ind)
    % rand_layers_ind = [2, 6, 10, 12 14];
    % ind_tl = 1;
    % LOI = layers_set{ind_tl};
    targetlayer_ind = change_layers_ind(ind_tl);
    
    layertmp = leakyReluLayer(scalepReLU, 'Name', ['leaky' num2str(ind_tl)]);
    net_tmp.Layers(targetlayer_ind) = layertmp;
end
net_test = net_test.loadobj(net_tmp);
% figure
% imagesc(net.Layers(2).Weights(:,:,1,1))
%
% figure
% imagesc(net_test.Layers(2).Weights(:,:,1,1))

end