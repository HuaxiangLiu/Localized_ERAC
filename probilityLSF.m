%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of "Localized edge-region-based active contour model by incorporating
% the probability scores for medical image segmentation" 
% Huaxiang Liu
% East China University of Technology&&Central South University, Changsha, 
% China
% 6th, March, 2020
% Email: felicia_liu@126.com
%im_bg:the label image with background (blue color) and foreground (red)
%Img  :the original segmented image
%type :the local fitting scale is set to 3,4,4,5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [edges, phi,mask,s] = probilityLSF(im_bg,Img,type)

  fg = find(im_bg(:,:,1) == 255&im_bg(:,:,2) == 0);
  bg = find(im_bg(:,:,3) == 255&im_bg(:,:,1) == 0);

  TrainLabel = zeros(length(bg)+length(fg),1);
  [h,w] =size(Img);
  
  if type == 0
      trainset = zeros(length(bg)+length(fg),9);
      testset = [reshape(Img(1:h-2,1:w-2),[],1) reshape(Img(1:h-2,2:w-1),[],1) reshape(Img(1:h-2,3:w),[],1)...
                 reshape(Img(2:h-1,1:w-2),[],1) reshape(Img(2:h-1,2:w-1),[],1) reshape(Img(2:h-1,3:w),[],1)...
                 reshape(Img(3:h  ,1:w-2),[],1) reshape(Img(3:h  ,2:w-1),[],1) reshape(Img(3:h  ,3:w),[],1)];
      for i = 1: length(fg)
          [th,tw] = ind2sub(size(Img),fg(i));
          trainset(i,:) = [Img(th-1,tw-1) Img(th-1,tw) Img(th-1,tw+1)...
                           Img(th,tw-1)   Img(th,tw)   Img(th,tw+1)...
                           Img(th+1,tw-1) Img(th+1,tw) Img(th+1,tw+1)]; 
          TrainLabel(i,:) = 1;
      end
      for i = 1: length(bg)
           [lh,lw] = ind2sub(size(Img),bg(i));
           trainset(length(fg)+i,:) = [Img(lh-1,lw-1) Img(lh-1,lw) Img(lh-1,lw+1)...
                                       Img(lh,lw-1)   Img(lh,lw)   Img(lh,lw+1)...
                                       Img(lh+1,lw-1) Img(lh+1,lw) Img(lh+1,lw+1)]; 
           TrainLabel(length(fg)+i,:) = 0;
      end
  elseif type ==1
      trainset = zeros(length(bg)+length(fg),16);
      testset = [reshape(Img(1:h-3,1:w-3),[],1) reshape(Img(1:h-3,2:w-2),[],1) reshape(Img(1:h-3,3:w-1),[],1) reshape(Img(1:h-3,4:w),[],1)...
                 reshape(Img(2:h-2,1:w-3),[],1) reshape(Img(2:h-2,2:w-2),[],1) reshape(Img(2:h-2,3:w-1),[],1) reshape(Img(2:h-2,4:w),[],1)...
                 reshape(Img(3:h-1,1:w-3),[],1) reshape(Img(3:h-1,2:w-2),[],1) reshape(Img(3:h-1,3:w-1),[],1) reshape(Img(3:h-1,4:w),[],1)...
                 reshape(Img(4:h  ,1:w-3),[],1) reshape(Img(4:h  ,2:w-2),[],1) reshape(Img(4:h  ,3:w-1),[],1) reshape(Img(4:h  ,4:w),[],1)]; 
      for i = 1: length(fg)
           [th,tw] = ind2sub(size(Img),fg(i));
           trainset(i,:) = [Img(th-2,tw-2) Img(th-2,tw-1) Img(th-2,tw) Img(th-2,tw+1)...
                            Img(th-1,tw-2) Img(th-1,tw-1) Img(th-1,tw) Img(th-1,tw+1)...
                            Img(th,  tw-2) Img(th  ,tw-1) Img(th  ,tw) Img(th  ,tw+1)...
                            Img(th+1,tw-2) Img(th+1,tw-1) Img(th+1,tw) Img(th+1,tw+1)]; 
           TrainLabel(i,:) = 1;
       end
       for i = 1: length(bg)
           [lh,lw] = ind2sub(size(Img),bg(i));
           trainset(length(fg)+i,:) = [Img(lh-2,lw-2) Img(lh-2,lw-1) Img(lh-2,lw) Img(lh-2,lw+1)...
                                       Img(lh-1,lw-2) Img(lh-1,lw-1) Img(lh-1,lw) Img(lh-1,lw+1)...
                                       Img(lh,  lw-2) Img(lh  ,lw-1) Img(lh  ,lw) Img(lh  ,lw+1)...
                                       Img(lh+1,lw-2) Img(lh+1,lw-1) Img(lh+1,lw) Img(lh+1,lw+1)]; 
           TrainLabel(length(fg)+i,:) = 0;
       end
  elseif type ==2
            trainset = zeros(length(bg)+length(fg),25);
      testset = [reshape(Img(1:h-4,1:w-4),[],1) reshape(Img(1:h-4,2:w-3),[],1) reshape(Img(1:h-4,3:w-2),[],1) reshape(Img(1:h-4,4:w-1),[],1) reshape(Img(1:h-4,5:w),[],1)...
                 reshape(Img(2:h-3,1:w-4),[],1) reshape(Img(2:h-3,2:w-3),[],1) reshape(Img(2:h-3,3:w-2),[],1) reshape(Img(2:h-3,4:w-1),[],1) reshape(Img(2:h-3,5:w),[],1)...
                 reshape(Img(3:h-2,1:w-4),[],1) reshape(Img(3:h-2,2:w-3),[],1) reshape(Img(3:h-2,3:w-2),[],1) reshape(Img(3:h-2,4:w-1),[],1) reshape(Img(3:h-2,5:w),[],1)...
                 reshape(Img(4:h-1,1:w-4),[],1) reshape(Img(4:h-1,2:w-3),[],1) reshape(Img(4:h-1,3:w-2),[],1) reshape(Img(4:h-1,4:w-1),[],1) reshape(Img(4:h-1,5:w),[],1)...
                 reshape(Img(5:h,  1:w-4),[],1) reshape(Img(5:h  ,2:w-3),[],1) reshape(Img(5:h  ,3:w-2),[],1) reshape(Img(5:h  ,4:w-1),[],1) reshape(Img(5:h  ,5:w),[],1)]; 
      for i = 1: length(fg)
           [th,tw] = ind2sub(size(Img),fg(i));
           trainset(i,:) = [Img(th-2,tw-2) Img(th-2,tw-1) Img(th-2,tw) Img(th-2,tw+1) Img(th-2,tw+2)...
                            Img(th-1,tw-2) Img(th-1,tw-1) Img(th-1,tw) Img(th-1,tw+1) Img(th-1,tw+2)...
                            Img(th,  tw-2) Img(th  ,tw-1) Img(th  ,tw) Img(th  ,tw+1) Img(th  ,tw+2)...
                            Img(th+1,tw-2) Img(th+1,tw-1) Img(th+1,tw) Img(th+1,tw+1) Img(th+1,tw+2)...
                            Img(th+2,tw-2) Img(th+2,tw-1) Img(th+2,tw) Img(th+2,tw+1) Img(th+2,tw+2)]; 
           TrainLabel(i,:) = 1;
       end
       for i = 1: length(bg)
           [lh,lw] = ind2sub(size(Img),bg(i));
           trainset(length(fg)+i,:) = [Img(lh-2,lw-2) Img(lh-2,lw-1) Img(lh-2,lw) Img(lh-2,lw+1) Img(lh-2,lw+2)...
                                       Img(lh-1,lw-2) Img(lh-1,lw-1) Img(lh-1,lw) Img(lh-1,lw+1) Img(lh-1,lw+2)...
                                       Img(lh,  lw-2) Img(lh  ,lw-1) Img(lh  ,lw) Img(lh  ,lw+1) Img(lh  ,lw+2)...
                                       Img(lh+1,lw-2) Img(lh+1,lw-1) Img(lh+1,lw) Img(lh+1,lw+1) Img(lh+1,lw+2)...
                                       Img(lh+2,lw-2) Img(lh+2,lw-1) Img(lh+2,lw) Img(lh+2,lw+1) Img(lh+2,lw+2)]; 
           TrainLabel(length(fg)+i,:) = 0;
       end
    elseif type ==3
      trainset = zeros(length(bg)+length(fg),16);
      testset = [reshape(Img(1:h-3,1:w-3),[],1) reshape(Img(1:h-3,2:w-2),[],1) reshape(Img(1:h-3,3:w-1),[],1) reshape(Img(1:h-3,4:w),[],1)...
                 reshape(Img(2:h-2,1:w-3),[],1) reshape(Img(2:h-2,2:w-2),[],1) reshape(Img(2:h-2,3:w-1),[],1) reshape(Img(2:h-2,4:w),[],1)...
                 reshape(Img(3:h-1,1:w-3),[],1) reshape(Img(3:h-1,2:w-2),[],1) reshape(Img(3:h-1,3:w-1),[],1) reshape(Img(3:h-1,4:w),[],1)...
                 reshape(Img(4:h  ,1:w-3),[],1) reshape(Img(4:h  ,2:w-2),[],1) reshape(Img(4:h  ,3:w-1),[],1) reshape(Img(4:h  ,4:w),[],1)]; 
      for i = 1: length(fg)
           [th,tw] = ind2sub(size(Img),fg(i));
           trainset(i,:) = [Img(th-1,tw-2) Img(th-1,tw-1) Img(th-1,tw) Img(th-1,tw+1)...
                            Img(th  ,tw-2) Img(th,tw-1) Img(th,tw) Img(th,tw+1)...
                            Img(th+1,tw-2) Img(th+1,tw-1) Img(th+1,tw) Img(th+1,tw+1)...
                            Img(th+2,tw-2) Img(th+2,tw-1) Img(th+2,tw) Img(th+2,tw+1)]; 
           TrainLabel(i,:) = 1;
       end
       for i = 1: length(bg)
           [lh,lw] = ind2sub(size(Img),bg(i));
           trainset(length(fg)+i,:) = [Img(lh-1,lw-2) Img(lh-1,lw-1) Img(lh-1,lw) Img(lh-1,lw+1)...
                                       Img(lh,  lw-2) Img(lh  ,lw-1) Img(lh  ,lw) Img(lh  ,lw+1)...
                                       Img(lh+1,lw-2) Img(lh+1,lw-1) Img(lh+1,lw) Img(lh+1,lw+1)...
                                       Img(lh+2,lw-2) Img(lh+2,lw-1) Img(lh+2,lw) Img(lh+2,lw+1)]; 
           TrainLabel(length(fg)+i,:) = 0;
       end
  end  
  
 %Factor = TreeBagger(100, trainset, TrainLabel);
 %[label,possiblity] = predict(Factor, double(testset));
 model = ClassificationKNN.fit(trainset,TrainLabel,'NumNeighbors',99);
 [label,possiblity] = predict(model,double(testset));
  score = possiblity(:,2);
  stride = type+1;  
  
  s =  ones(h,w);	mask = s;
  if type ==3
      stride = stride -2;
  end
  s(2:h-stride,2:w-stride) = reshape(score,[h-1-stride,w-1-stride]);
  %mask(2:h-stride,2:w-stride) = reshape(str2double(label),[h-1-stride,w-1-stride]);
  mask(2:h-stride,2:w-stride) = reshape(label,[h-1-stride,w-1-stride]);
  edges = sin(pi*s/2);
  pro = zeros(size(Img));phi = pro;phi(:,:) = -2;
  phi(im_bg(:,:,1) == 255&im_bg(:,:,2) == 0) = 2;

