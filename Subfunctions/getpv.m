function pvalues = getpv(response_tot)

imageset_number = 3; %% == how many number of standard + control sets?
N_neurons = size(response_tot, 3);
image_iter = size(response_tot, 2)/imageset_number;
pvalues = zeros(3,N_neurons);
for kk = 1:N_neurons
tabletmp = response_tot(:,:,kk)';
p = anova2(tabletmp, image_iter, 'off');
pvalues(:, kk) = p;
end


end