function [meantmp, stdtmp] = getstatisticsofweightdist(net, targetlayer_ind)

    %% get original weight distribution for L5
    
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    stdtmp = std(weight_conv(:));
    meantmp = mean(weight_conv(:));

end