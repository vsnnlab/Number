function pvalues = getpvforeach(response_tot)

imageset_number = 3; %% == how many number of standard + control sets?
N_neurons = size(response_tot, 3);
image_iter = size(response_tot, 2)/imageset_number;
pvalues = zeros(3,N_neurons);
group = {'1','2','4','6','8','10','12','14','16','18','20','22','24','26','28','30'};
for kk = 1:N_neurons
tabletmp_std = response_tot(:,1:image_iter,kk)';
tabletmp_ctr1 = response_tot(:,image_iter+1:2*image_iter,kk)';
tabletmp_ctr2 = response_tot(:,2*image_iter+1:3*image_iter,kk)';

pstd = anova1(tabletmp_std, group, 'off');
pctr1 = anova1(tabletmp_ctr1, group,'off');
pctr2 = anova1(tabletmp_ctr2, group,'off');
pvalues(:, kk) = [pstd; pctr1; pctr2];
end


end