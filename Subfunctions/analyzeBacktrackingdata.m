
function [weightdist_foreachL5PN_tot, weightdist_control_tot, incdecportion_foreachL5PN_tot...
    , weightdist_foreachL5selective_tot, incdecportion_foreachL5selective_tot...
    , weightdist_foreachL5Nonselective_tot, incdecportion_foreachL5Nonselective_tot, ...
    incdecportion_foreachL5_tot]...
    = analyzeBacktrackingdata(iter, pathtmp, L4investigatePN, L5investigatePN, nametmp)

weightdist_foreachL5PN_tot = cell(iter, 16,2);
weightdist_control_tot = cell(iter, 16, 2);
incdecportion_foreachL5PN_tot = zeros(iter,16, 2);
weightdist_foreachL5selective_tot = cell(iter, 2);
incdecportion_foreachL5selective_tot = zeros(iter, 2);
weightdist_foreachL5Nonselective_tot = cell(iter, 2);
incdecportion_foreachL5Nonselective_tot = zeros(iter, 2);
incdecportion_foreachL5_tot = zeros(iter, 2);
for iterind = 1:iter
 
    
    load([pathtmp '/Dataset/Data/Data_for_Backtracking_iter_' num2str(iterind) nametmp])
    
    weightdist_foreachL5PN = cell(length(L5investigatePN), length(L4investigatePN));
    
    for ii = 1:length(L5investigatePN)
        PNtmp_L5 = L5investigatePN(ii);
        indtmp = (NS_PN_L5 == PNtmp_L5);
        tmp1 = weights_ForEachL5neuron(indtmp,:);
        tmp1 = tmp1(:);
        randtmp1 = tmp1(randperm(length(tmp1)));
        tmp2 = PNsofL4_connectedtoeachL5Neuron(indtmp,:);
        tmp2 = tmp2(:);
        
%         tmp3 = R2sofL4_connectedtoeachL5Neuron(indtmp,:);
%         tmp3 = tmp3(:);
        
        for jj = 1:length(L4investigatePN)
            PNtmp_L4 = L4investigatePN(jj);
            indtmp1 = tmp2==PNtmp_L4;
%             indtmp2 = tmp3>-10000;
%             indtmp1 = indtmp1 & indtmp2;
            weightdist_foreachL5PN{ii,jj} = tmp1(indtmp1);
            weightdist_foreachL5PN_tot{iterind, ii,jj} = tmp1(indtmp1);
            
            incdecportion_foreachL5PN_tot(iterind, ii,jj) = sum(indtmp1)/length(indtmp1);
            
            weightdist_control_tot{iterind, ii,jj} = randtmp1(indtmp1);
        end
    end
    
    %% compare selective and nonselective
    indtmp = (NS_PN_L5>0);
    tmp_NS = weights_ForEachL5neuron(indtmp, :);
    tmp_NS = tmp_NS(:);
    tmp2 = PNsofL4_connectedtoeachL5Neuron(indtmp,:);
    tmp2 = tmp2(:);
%     tmp3 = R2sofL4_connectedtoeachL5Neuron(indtmp,:);
%     tmp3 = tmp3(:);
    for jj = 1:length(L4investigatePN)
        PNtmp_L4 = L4investigatePN(jj);
        indtmp1 = tmp2==PNtmp_L4;
%         indtmp2 = tmp3>-10000;
%         indtmp1 = indtmp1 & indtmp2;
        weightdist_foreachL5selective_tot{iterind, jj} = tmp_NS(indtmp1);
        incdecportion_foreachL5selective_tot(iterind, jj) = sum(indtmp1)/length(indtmp1);
    end
    
    
    indtmp = NS_PN_L5<0.1;
    tmp_NNS = weights_ForEachL5neuron(indtmp, :);
    tmp_NNS = tmp_NNS(:);
    
    tmp2 = PNsofL4_connectedtoeachL5Neuron(indtmp,:);
    tmp2 = tmp2(:);
%     tmp3 = R2sofL4_connectedtoeachL5Neuron(indtmp,:);
%     tmp3 = tmp3(:);
    for jj = 1:length(L4investigatePN)
        PNtmp_L4 = L4investigatePN(jj);
        indtmp1 = tmp2==PNtmp_L4;
%         indtmp2 = tmp3>-10000;
%         indtmp1 = indtmp1 & indtmp2;
        weightdist_foreachL5Nonselective_tot{iterind, jj} = tmp_NNS(indtmp1);
        incdecportion_foreachL5Nonselective_tot(iterind, jj) = sum(indtmp1)/length(indtmp1);
    end
    
    indtmp = true(1,length(NS_PN_L5));
    tmp_NNS = weights_ForEachL5neuron(indtmp, :);
    tmp_NNS = tmp_NNS(:);
    
    tmp2 = PNsofL4_connectedtoeachL5Neuron(indtmp,:);
    tmp2 = tmp2(:);
%     tmp3 = R2sofL4_connectedtoeachL5Neuron(indtmp,:);
%     tmp3 = tmp3(:);
    for jj = 1:length(L4investigatePN)
        PNtmp_L4 = L4investigatePN(jj);
        indtmp1 = tmp2==PNtmp_L4;
%         indtmp2 = tmp3>-10000;
%         indtmp1 = indtmp1 & indtmp2;
        incdecportion_foreachL5_tot(iterind, jj) = sum(indtmp1)/length(indtmp1);
    end

end
end