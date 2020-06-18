%% A. numerosity stimulus generation

function [image_sets_standard, image_sets_control1, image_sets_control2, polyxy] = Stimulus_generation_Nasr(number_sets, image_iter)

%% input image size : 227x227
%% A-1. a standard set : all dots had about the same radius

[xax, yax] = meshgrid(1:227, 1:227);
rax = xax+1i*yax;

% number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
% image_iter = 7;
image_sets_standard = zeros(size(rax, 1), size(rax,2), image_iter, length(number_sets));

for ii = 1:length(number_sets)
    numtmp = number_sets(ii);
    for jj = 1:image_iter
        
        epsil = randn;
        circle_radius = 7+0.7*epsil;
        circle_loc = ceil(circle_radius)+randi(size(rax,2)-2*ceil(circle_radius))...
            +1i*(ceil(circle_radius)+randi(size(rax,1)-2*ceil(circle_radius)));
        radtmp = circle_radius;
        loctmp = circle_loc;
        
        while length(radtmp)<numtmp
            epsil = randn;
            circle_radius = 7+0.7*epsil;
            circle_loc = ceil(circle_radius)+randi(size(rax,2)-2*ceil(circle_radius))...
                +1i*(ceil(circle_radius)+randi(size(rax,1)-2*ceil(circle_radius)));
            
            distancestmp = abs(circle_loc-loctmp);
            radistmp = (circle_radius+radtmp);
            okToAdd = all(distancestmp>radistmp);
            if circle_radius>0
                if okToAdd
                    radtmp = [radtmp circle_radius];
                    loctmp = [loctmp, circle_loc];
                end
            end
        end
        
        imgtmp = zeros(size(rax,1), size(rax,2));
        for kk = 1:numtmp
            rtmp = abs(rax-loctmp(kk));
            imgtmpp = rtmp<=radtmp(kk);
            imgtmp = imgtmp+imgtmpp;
        end
        image_sets_standard(:,:,jj,ii) = imgtmp;
        
    end
end
% 
% figure
% subplot(2,3,1)
% tmp = squeeze(image_sets_standard(:,:, 1,3));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,2)
% tmp = squeeze(image_sets_standard(:,:, 1,6));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,3)
% tmp = squeeze(image_sets_standard(:,:, 1,10));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,4)
% tmp = squeeze(image_sets_standard(:,:, 2,3));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,5)
% tmp = squeeze(image_sets_standard(:,:, 2,6));
% imagesc(tmp);colormap(gray);axis image xy
% subplot(2,3,6)
% tmp = squeeze(image_sets_standard(:,:, 2,10));
% imagesc(tmp);colormap(gray);axis image xy

%% control image sets 1 : same total area
image_sets_control1 = zeros(size(rax, 1), size(rax,2), image_iter, length(number_sets));


%% test

for ii = 1:length(number_sets)
    numtmp = number_sets(ii);
    for jj = 1:image_iter
        
        epsil = randn(1,numtmp);
        circle_radius = 7+0.7*epsil;
        areaSum = sum(pi*circle_radius.^2);
        scalingtmp = sqrt(areaSum/1200);
        circle_radius = circle_radius/scalingtmp;
        
        average_dist = 0;
        while ~ (average_dist>90 && average_dist<100)
            radtmp = [];
            loctmp = [];
            average_dist = 0;
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
            
            if numtmp>1
                for avdind = 1:length(loctmp)
                    tmp = (abs(loctmp(avdind) - loctmp));
                    tmp(tmp==0) = [];
                    distmeantmp = mean(tmp);
                    average_dist = average_dist+distmeantmp;
                end
                average_dist = average_dist/length(loctmp);
                %                 disp(average_dist)
            else
                average_dist = 95;
            end
            
        end
        
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

%
% %% control image sets 2 : same total circumference
% image_sets_control2 = zeros(size(rax, 1), size(rax,2), image_iter, length(number_sets));
% rsum = 60;
% for ii = 1:length(number_sets)
%     numtmp = number_sets(ii);
%     for jj = 1:image_iter
%
%         circle_radius = rsum/numtmp;
%         circle_loc = ceil(circle_radius)+randi(size(rax,2)-2*ceil(circle_radius))...
%             +1i*(ceil(circle_radius)+randi(size(rax,1)-2*ceil(circle_radius)));
%         radtmp = circle_radius;
%         loctmp = circle_loc;
%
%         while length(radtmp)<numtmp
%             epsil = randn;
%             circle_loc = ceil(circle_radius)+randi(size(rax,2)-2*ceil(circle_radius))...
%                 +1i*(ceil(circle_radius)+randi(size(rax,1)-2*ceil(circle_radius)));
%
%             distancestmp = abs(circle_loc-loctmp);
%             radistmp = (circle_radius+radtmp);
%             okToAdd = all(distancestmp>sqrt(2)*radistmp);
%             if circle_radius>0
%                 if okToAdd
%                     radtmp = [radtmp circle_radius];
%                     loctmp = [loctmp, circle_loc];
%                 end
%             end
%         end
%
%         imgtmp = zeros(size(rax,1), size(rax,2));
%         for kk = 1:numtmp
%             rtmp = abs(rax-loctmp(kk));
%             imgtmpp = rtmp<=radtmp(kk);
%             imgtmp = imgtmp+imgtmpp;
%         end
%         image_sets_control2(:,:,jj,ii) = imgtmp;
%
%     end
%end

[xax, yax] = meshgrid(1:227, 1:227);
rax = xax+1i*yax;

%% control image sets 2 : different shape + convex hull
image_sets_control2 = zeros(size(rax, 1), size(rax,2), image_iter, length(number_sets));

polyxx = zeros(length(number_sets), image_iter, 5);
polyyy = zeros(length(number_sets), image_iter, 5);
for ii = 1:length(number_sets)
    numtmp = number_sets(ii);
 
    for jj = 1:image_iter
        %% define convex hull
        theta = randi(180)*pi/180;
        xtmpp = zeros(1,5);
        ytmpp = zeros(1,5);
        for thetaind = 1:5
            xtmpp(thetaind) = round(size(rax,1)/2)+110*cos(theta+(thetaind-1)*2*pi/5);
            ytmpp(thetaind) = round(size(rax,1)/2)+110*sin(theta+(thetaind-1)*2*pi/5);
        end
        k = convhull(xtmpp, ytmpp);
        polyxx(ii,jj,:) = xtmpp;
        polyyy(ii,jj,:) = ytmpp;
        %%
        radtmp = [];
        loctmp = [];
        
        
        while length(radtmp)<numtmp
            epsil = randn;
            circle_radius = 7+0.7*epsil;
            circle_loc = ceil(circle_radius)+randi(size(rax,2)-2*ceil(circle_radius))...
                +1i*(ceil(circle_radius)+randi(size(rax,1)-2*ceil(circle_radius)));
            
            distancestmp = abs(circle_loc-loctmp);
            radistmp = (circle_radius+radtmp);
            okToAdd = all(distancestmp>sqrt(2)*radistmp);
            IN = inpolygon(real(circle_loc),imag(circle_loc),xtmpp(k),ytmpp(k));
            if circle_radius>0
                if okToAdd && IN
                    radtmp = [radtmp circle_radius];
                    loctmp = [loctmp, circle_loc];
                end
            end
        end
        
        imgtmp = zeros(size(rax,1), size(rax,2));
        
        for kk = 1:numtmp
            rtmp = (rax-loctmp(kk));
            tmp = rand();

            if tmp>0.75 % rectangle
                imgtmpp = abs(real(rtmp))<=radtmp(kk) & abs(imag(rtmp))<=radtmp(kk);
            elseif tmp>0.5 & tmp<0.75 % circle
                imgtmpp = abs(rtmp)<=radtmp(kk);
            elseif tmp<0.5 & tmp>0.25 % ellipse
                imgtmpp = (real(rtmp).^2/(radtmp(kk)^2)+imag(rtmp).^2/((0.5*radtmp(kk))^2) )<1;
            else % triangle
                theta = randi(180)*pi/180;
                xtmp = zeros(1,3);
                ytmp = zeros(1,3);
                for thetaind = 1:3
                    xtmp(thetaind) = real(loctmp(kk))+radtmp(kk)*cos(theta+(thetaind-1)*2*pi/3);
                    ytmp(thetaind) = imag(loctmp(kk))+radtmp(kk)*sin(theta+(thetaind-1)*2*pi/3);
                end
                k = convhull(xtmp, ytmp);
                xx = real(rax); yy= imag(rax);
                IN = inpolygon(xx, yy,xtmp(k),ytmp(k));
                imgtmpp = IN;
            end
            imgtmp = imgtmp+imgtmpp;
        end
        image_sets_control2(:,:,jj,ii) = imgtmp;
    end
end
polyxy = cat(4, polyxx, polyyy);


end