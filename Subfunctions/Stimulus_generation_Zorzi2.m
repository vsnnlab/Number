%% A. numerosity stimulus generation

function [image_sets, image_sets_stat] = Stimulus_generation_Zorzi2(number_sets, image_iter, radius_cand)



[xax, yax] = meshgrid(1:227, 1:227);
rax = xax+1i*yax;
%% control image sets 1 : same total area
image_sets_control1 = zeros(size(rax, 1), size(rax,2), image_iter, length(number_sets));

image_sets = [];
% factor = 227*227/900;
% sqrt((32:32:256)*factor/pi)

% radius_cand = 4*sqrt(1:8); % radius for 30 dots
areatot = 30*(pi*radius_cand.^2);
image_sets_stat = [];
image_sets_stattmp = zeros(4, image_iter, length(number_sets));
% dim1 : area, average dist, dotsize, setNo
% Stoianov & Zorzi 2018 : Area - 32:32:256 /900 total area

%% test
% radius sets : 4 8 12 16 20 24
for areaind = 1:length(areatot)
    areaSumtmp = areatot(areaind);
    for ii = 1:length(number_sets)
        numtmp = number_sets(ii);
        for jj = 1:image_iter
            
            epsil = randn(1,numtmp);
            circle_radius = 7+0.7*epsil;
             
            areaSum = sum(pi*circle_radius.^2);
            scalingtmp = sqrt(areaSum/areaSumtmp);
            circle_radius = circle_radius/scalingtmp;
            circumferencetmp = sum(2*pi*circle_radius);
            dotsizetmp = mean(pi*circle_radius.^2);
            areatmp = sum(pi*circle_radius.^2);
            
            % img statistics
            image_sets_stattmp(1,jj,ii) = areatmp;
%             image_sets_stattmp(2,jj,ii) = circumferencetmp;
            image_sets_stattmp(3,jj,ii) = dotsizetmp;
            image_sets_stattmp(4,jj,ii) = areaind;
            
            % generate img
            radtmp = [];
            loctmp = [];
            
            radind = 1;
            while length(radtmp)<numtmp
                rad = circle_radius(radind);
                loc = ceil(rad)+randi(size(rax,2)-2*ceil(rad))...
                    +1i*(ceil(rad)+randi(size(rax,1)-2*ceil(rad)));
                if length(loctmp)>=1
                    distancestmp = abs(loc-loctmp);
                    radistmp = rad+radtmp;
                else
                    distancestmp = 1;
                    radistmp = 0;
                end
                
                okToAdd = all(distancestmp>radistmp);
                if rad>0
                    if okToAdd
                        radtmp = [radtmp rad];
                        loctmp = [loctmp, loc];
                        radind = radind +1;
                    end
                end
                
            end
            
            %% calculate density
            average_dist = 0;
            if ii>1
                for avdind = 1:length(loctmp)
                    tmp = (abs(loctmp(avdind) - loctmp));
                    tmp(tmp==0) = [];
                    distmeantmp = mean(tmp);
                    average_dist = average_dist+distmeantmp;
                end
                average_dist = average_dist/length(loctmp);
                %                 disp(average_dist)
            else
                average_dist = nan;
            end
            image_sets_stattmp(2,jj,ii) = average_dist;
            
            
            %% draw image
            imgtmp = zeros(size(rax,1), size(rax,2));
            for kk = 1:numtmp
                rtmp = abs(rax-loctmp(kk));
                imgtmpp = rtmp<=radtmp(kk);
                imgtmp = imgtmp+imgtmpp;
            end
            image_sets_control1(:,:,jj,ii) = imgtmp;
            
            
        end
    end
    image_sets = cat(3,image_sets, image_sets_control1);
    image_sets_stat = cat(2,image_sets_stat, image_sets_stattmp);
%     areaind
end

% for ii = 1:8
%     subplot(1,8,ii)
%     imagesc(image_sets(:,:,50*(ii-1)+16,12))
% end

%%
% 
% figure
% subplot(2,3,1)
% tmp = squeeze(image_sets_control1(:,:, 1,3));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,2)
% tmp = squeeze(image_sets_control1(:,:, 1,6));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,3)
% tmp = squeeze(image_sets_control1(:,:, 1,10));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,4)
% tmp = squeeze(image_sets_control1(:,:, 2,3));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,5)
% tmp = squeeze(image_sets_control1(:,:, 2,6));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,6)
% tmp = squeeze(image_sets_control1(:,:, 2,15));
% imagesc(tmp);colormap(gray);axis image xy

end