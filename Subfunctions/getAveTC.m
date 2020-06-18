function [Numerical_distance, resp_sets] = getAveTC(response_NS_mean, PNind, number_sets)

% PNind = PNind_perm
% response_NS_mean = resp_NS_mean_perm;
Numerical_distance = -30:30;
resp_sets = zeros(size(response_NS_mean, 2),length(Numerical_distance))/0;
for ii = 1:size(response_NS_mean, 2)
    
    PNtmp = number_sets(PNind(ii));
    resptmp = response_NS_mean(:, ii);
    disttmp = number_sets-PNtmp;
    disttmpind = 31+ disttmp;
    resp_sets(ii, disttmpind) = resptmp/max(resptmp);
end


end