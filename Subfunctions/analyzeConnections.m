function [connectedweights, control1s, NS_wmean, NNS_wmean] = ...
    analyzeConnections(meantmp, L5investigatePN, L4investigatePN...
    , weightdist_foreachL5PN_tot, weightdist_control_tot, weightdist_foreachL5Nonselective_tot, iter, isrelativ)

Ntmp = length(L5investigatePN);
Ntmp2 = length(L4investigatePN);
connectedweights = zeros(iter, Ntmp, Ntmp2); control1s = zeros(iter, Ntmp, Ntmp2); 
NNS_wmean = zeros(iter, Ntmp2); NS_wmean = zeros(iter, Ntmp2);

for iterind = 1:iter
    %% Weight distribution
    check1 = zeros(16,2); 
    control1 = zeros(16,2); 
    
    for ii = 1:length(L5investigatePN)
        
        for jj = 1:length(L4investigatePN)
            weightdist_foreachL5PN = weightdist_foreachL5PN_tot{iterind, ii,jj};
            
            weightdisttmp = weightdist_foreachL5PN;
            weightdistcontrol = weightdist_control_tot{iterind, ii,jj};
            if isrelativ
                weightdisttmp = (weightdisttmp-meantmp);
            end
            
            check1(ii,jj) = mean(weightdisttmp);          
            control1(ii,jj) = mean(weightdistcontrol);
            
        end
       
    end
    connectedweights(iterind, :,:) = check1;
    control1s(iterind, :,:) = control1;
        
    %% For sel/nonselective units
    for jj = 1:length(L4investigatePN)
        weightdisttmp = weightdist_foreachL5Nonselective_tot{iterind, jj};
        if isrelativ
            weightdisttmp = (weightdisttmp-meantmp);
        end
        NNS_wmean(iterind, jj) = mean(weightdisttmp);
        
        weightdisttmp = [];
        for ii = 1:length(L5investigatePN)
            weightdisttmpp = weightdist_foreachL5PN_tot{iterind, ii,jj};
            weightdisttmp = [weightdisttmp, weightdisttmpp'];
        end
        if isrelativ
            weightdisttmp = (weightdisttmp-meantmp);
        end
        NS_wmean(iterind, jj) = mean(weightdisttmp);
    end
end



end