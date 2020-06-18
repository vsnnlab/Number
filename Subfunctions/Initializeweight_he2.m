function [net_test, lim, Weights, Biases] = Initializeweight_he2(net, rand_layers_ind, ver, stdfac)
net_test = net;
net_tmp = net_test.saveobj;
Weights = cell(1,length(rand_layers_ind));
Biases = cell(1,length(rand_layers_ind));

for ind_tl = 1:length(rand_layers_ind)
    % rand_layers_ind = [2, 6, 10, 12 14];
    % ind_tl = 1;
    % LOI = layers_set{ind_tl};
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    
    fan_in = size(weight_conv,1)*size(weight_conv,2)*size(weight_conv,3);
    
    if ver ==1
        lim(ind_tl) = sqrt(2/fan_in);
        Wtmp = stdfac*randn(size(weight_conv))*sqrt(2/fan_in); % he initializaation
        Btmp = randn(size(bias_conv));
    elseif ver ==2
        lim(ind_tl) = sqrt(1/fan_in);
        Wtmp = stdfac*randn(size(weight_conv))*sqrt(1/fan_in); % LeCun initializaation
        Btmp = randn(size(bias_conv));
    elseif ver ==3
        lim(ind_tl) = sqrt(6/fan_in);
        Wtmp = stdfac*(rand(size(weight_conv))-0.5)*2*sqrt(6/fan_in); % he uniform initializaation
        Btmp = randn(size(bias_conv));
    elseif ver ==4
        lim(ind_tl) = sqrt(3/fan_in);
        Wtmp = stdfac*(rand(size(weight_conv))-0.5)*2*sqrt(3/fan_in); % Lecun uniform initializaation
        Btmp = randn(size(bias_conv));
    end

    wstd = std(weight_conv(:));
    bstd = std(bias_conv(:));
    wmean = mean(weight_conv(:));
    bmean = mean(bias_conv(:));
    %% change network parameters
    % num_NS = zeros(1,10);
    % for iii = 1:10
    
%      weight_conv_randomize = (1*Wtmp);
%      bias_conv_randomize = (0*Btmp);
    weight_conv_randomize = wmean-wmean+Wtmp*wstd/wstd;
    bias_conv_randomize = bmean-bmean+Btmp*0*bstd/bstd;
    
    Weights{ind_tl} = weight_conv_randomize;
    Biases{ind_tl} = bias_conv_randomize;
    
    net_tmp.Layers(targetlayer_ind).Weights = weight_conv_randomize;
    net_tmp.Layers(targetlayer_ind).Bias = bias_conv_randomize;
end
net_test = net_test.loadobj(net_tmp);
% figure
% imagesc(net.Layers(2).Weights(:,:,1,1))
%
% figure
% imagesc(net_test.Layers(2).Weights(:,:,1,1))

end