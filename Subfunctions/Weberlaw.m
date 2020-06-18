function [sigmas, R2] = Weberlaw(TNcurves, number_sets)  % TNcurves = 16*16 matrix

options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
xtmp = number_sets;
sigmas = zeros(1,length(number_sets));
R2 = zeros(1,length(number_sets));
TNtmp = TNcurves;
for ii = 1:length(number_sets)
    ytmp = TNtmp(ii,:);
    if isnan(ytmp)
        sigmas(ii) = nan;
    else
        f = fit(xtmp.', ytmp.', 'gauss1', options);
        sigmas(ii) = f.c1/sqrt(2);
        tmp = f.a1*exp(-((xtmp-f.b1)/f.c1).^2);
    ymean = nanmean(ytmp);
    SStot = sum((ytmp-ymean).^2);
    SSres = sum((ytmp-tmp).^2);
    R2(ii) = 1-SSres./SStot;
       
    end
end


end