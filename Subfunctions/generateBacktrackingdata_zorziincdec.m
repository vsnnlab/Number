function generateBacktrackingdata_zorziincdec(generatedat,rand_layers_ind...
    , number_sets, iter, net, layers_name, pathtmp, layer_sets, ...
    p_th1, p_th2, p_th3, ind_layer, array_sz,...
    image_sets_standard, image_sets_control1, image_sets_control2, image_sets_zorzi,image_sets_zorzi_stat, nametmp)


if generatedat
    
    for iterind = 1:iter
        resp_NS_mean_L45 = cell(1,2);
        %% raw data simulation
        NS_ind_layers = cell(5,1);
        NS_PN_layers = cell(5,1);
        %net_changed = Randomizeweight_permute(net, rand_layers_ind);
        [net_changed, ~, ~, ~] = Initializeweight_he2(net, rand_layers_ind, 1, 1);
        for indl = 1:length(layer_sets)
            
            ind_l = layer_sets(indl);
            LOI = layers_name{ind_l};
            %             blank = zeros(227, 227,1,1);
            %             response_tot_blank = getactivation(net_changed, LOI, blank);
            
            if ind_l ==5
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
                NS_ind1 = PNind_RP==1;
                NS_PN = zeros(1,length(NS_ind));
                NS_PN(NS_ind) = PNind_RP;
                NS_ind_layers{ind_l} = NS_ind;
                NS_PN_layers{ind_l} = NS_PN;
                
                resp_NS_mean_L45{indl} = response_NS_mean_RP;
            else
                response_tot_zorzi = getactivation(net_changed, LOI, image_sets_zorzi);
                
                
                image_sets_zorzi_areas = squeeze(image_sets_zorzi_stat(1,:,:))';
                image_sets_zorzi_nums = [];
                for ii = 1:16
                    tmp = ii*ones(1,size(image_sets_zorzi_areas,2));
                    image_sets_zorzi_nums = cat(1, image_sets_zorzi_nums, number_sets(tmp));
                end
                
                areastmp = rescale(log((image_sets_zorzi_areas(:))));
                numstmp = rescale(log((image_sets_zorzi_nums(:))));
                
                Coeffs = zeros(3,size(response_tot_zorzi, 3));
                R2s = zeros(1,size(response_tot_zorzi, 3));
                for ii = 1:size(response_tot_zorzi, 3)
                    resptmp = squeeze(response_tot_zorzi(:,:,ii));
                    resptmp = (rescale(resptmp(:)));
                    X = [ones(size(areastmp)) areastmp numstmp];
                    [b,~,~,~,stats ] = regress(resptmp, X);
                    
                    R2s(ii) = stats(1);
                    Coeffs(:,ii) = b;
                end
                
                
                
                areacoef = Coeffs(2,:);
                numcoef = Coeffs(3,:);
                
                units_areacoef = areacoef;
                units_numcoef = numcoef;
                
                ind1 = R2s>0.1;
                ind2 = abs(units_areacoef)<0.1;
                ind = ind1 & ind2;
                
                ind_inc = ind & (units_numcoef>0);
                ind_dec = ind & (units_numcoef<0);
                NS_ind = logical(ind_inc | ind_dec);
                
                NS_PN = zeros(1,length(NS_ind));
                NS_PN(ind_inc) = 16;
                NS_PN(ind_dec) = 1;
                
                NS_ind_layers{ind_l} = NS_ind;
                NS_PN_layers{ind_l} = NS_PN;
                
                
            end
            
        end
        
        Cell_idx = cell(length(NS_ind),5,3); %% all neurons
        Cell_idx_ind = cell(length(NS_ind),5,2); %% all neurons, index
        
        %%
        %         ind_layer = 5;
        for iii = 1:length(NS_ind) % NS neuron order
            
            [row, col, chan] = ind2sub(array_sz(ind_layer,:), iii);
            Weight = net_changed.Layers(rand_layers_ind(ind_layer)).Weights;
            Cell_idx{iii, ind_layer, 1} = row; Cell_idx{iii, ind_layer, 2} = col; Cell_idx{iii, ind_layer, 3} = chan;
            Cell_idx_ind{iii, ind_layer, 1} = iii;
            %% relu5 -> conv5 -> relu4
            A = Weight(:,:,:,chan);
            w = size(Weight(:,:,:,chan),1); % filter size
            s = net.Layers(rand_layers_ind(ind_layer)).Stride;
            p = net.Layers(rand_layers_ind(ind_layer)).PaddingSize;
            row_pre = (row-1)*s(1)+w-2*p(1);
            col_pre = (col-1)*s(1)+w-2*p(1);
            [cc,rr] = meshgrid(-(w(1)-1)/2:1:(w(1)-1)/2);
            row_new = rr+row_pre; col_new = cc+col_pre;
            row_new = row_new(:); col_new = col_new(:);
            Cell_idx{iii,ind_layer-1,1} = row_new; Cell_idx{iii,ind_layer-1,2} = col_new;
        end
        %% For each layer 5 neuron, get NS_ind information of connected layer 4 neurons
        NS_ind_L5 = NS_ind_layers{ind_layer};
        NS_ind_L4 = NS_ind_layers{ind_layer-1};
        NS_PN_L5 = NS_PN_layers{ind_layer};
        NS_PN_L4 = NS_PN_layers{ind_layer-1};
        
        weightsum = zeros(2,length(NS_ind_L5));
        weights_ForEachL5neuron = zeros(length(NS_ind_L5), 3456);
        PNsofL4_connectedtoeachL5Neuron = zeros(length(NS_ind_L5), 3456);
        
        for iii = 1:length(NS_ind_L5)
            % disp(iii)
            % test ind
            %     tmp = find(NS_PN_L5==1);
            %     iii = datasample(tmp, 1);
            %
            row_new = Cell_idx{iii, ind_layer-1,1};
            col_new = Cell_idx{iii, ind_layer-1,2};
            NS_PN_L4_matrix = reshape(NS_PN_L4, array_sz(ind_layer-1,:));
            
            NS_PN_L4_matrix_padded = zeros(15,15,size(NS_PN_L4_matrix,3))/0;
            NS_PN_L4_matrix_padded(2:14, 2:14, :) = NS_PN_L4_matrix;
            rc = row_new(5);
            cc = col_new(5);
            PN_connected_L4_neurons = NS_PN_L4_matrix_padded(rc:rc+2, cc:cc+2, :);
            %   histogram(PN_connected_L4_neurons(PN_connected_L4_neurons>0))
            chan = Cell_idx{iii, ind_layer, 3};
            Weight_L5 = net_changed.Layers(rand_layers_ind(ind_layer)).Weights;
            Weight_L5_forthisneuron = Weight_L5(:,:,:,chan);
            Weight_L5_forthisneuron2 = cat(3, Weight_L5_forthisneuron, Weight_L5_forthisneuron);
            %     histogram(Weight_L5_forthisneuron2)
            %% calculate effect of 1 and 30
            
            tmp = PN_connected_L4_neurons(:);
            tmp1 = Weight_L5_forthisneuron2(:);
            weights_ForEachL5neuron(iii,:) = tmp1;
            PNsofL4_connectedtoeachL5Neuron(iii,:) = tmp;
            indtmp1 = tmp==1;
            indtmp30 = tmp==16;
            %     edges = -0.2:0.01:0.2;
            %     figure
            %     subplot(1,2,1)
            %     histogram(tmp1(indtmp1), edges)
            %     subplot(1,2,2)
            %     histogram(tmp1(indtmp30), edges)
            weightsum_1 = sum(tmp1(indtmp1));
            weightsum_30 = sum(tmp1(indtmp30));
            weightsum(1,iii) = weightsum_1;
            weightsum(2,iii) = weightsum_30;
        end
        
        save([pathtmp '/Dataset/Data/Data_for_Backtracking_iter_' num2str(iterind) nametmp], ...
            'resp_NS_mean_L45', 'weights_ForEachL5neuron', ...
            'PNsofL4_connectedtoeachL5Neuron', 'NS_ind_L5', 'NS_ind_L4', 'NS_PN_L5', 'NS_PN_L4')
    end
end


end