
function [response_tot_blank, net_changed,response_NS_mean_RP, ...
    ind_NS, NS_ind, PNind_RP] = ...
    getIncdecunits(rand_layers_ind, LOI, net, p_th1, p_th2, p_th3, ...
    image_sets_standard, image_sets_control1, image_sets_control2)

net_changed = Randomizeweight_permute(net, rand_layers_ind);
blank = zeros(227, 227,1,1);

response_tot_blank = getactivation(net_changed, LOI, blank);
%% Step1. get NS neurons (index: NS_ind)
%% define layer of interest and calculate response
response_tot_standard_RP = getactivation(net_changed, LOI, image_sets_standard);
response_tot_control1_RP = getactivation(net_changed, LOI, image_sets_control1);
response_tot_control2_RP = getactivation(net_changed, LOI, image_sets_control2);
response_tot_RP = cat(2,response_tot_standard_RP, response_tot_control1_RP, response_tot_control2_RP);
%% get p-values from response
pvalues_RP = getpv(response_tot_RP);

pv1 = pvalues_RP(1,:); pv2 = pvalues_RP(2,:);pv3 = pvalues_RP(3,:);
ind1 = (pv1<p_th1);ind2 = (pv2>p_th2);ind3 = (pv3>p_th3);
ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
NS_ind = logical(ind1.*ind2.*ind3);
%% get preferred number
response_NS_tot_RP = response_tot_RP(:,:,ind_NS);
response_NS_mean_RP = squeeze(mean(response_NS_tot_RP, 2));
[M,PNind_RP] = max(response_NS_mean_RP);
%         xtmp = log2(number_sets);
%         [sigmas_L4, R2s_L4] = getlogfit_individual(ind_NS, response_NS_mean_RP, xtmp);

end





