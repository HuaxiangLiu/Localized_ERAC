%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of "Localized edge-region-based active contour model by incorporating
% the probability scores for medical image segmentation" 
% Huaxiang Liu
% East China University of Technology&&Central South University, Changsha, 
% China
% 6th, March, 2020
% Email: felicia_liu@126.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;
close all;
addpath 'image'

Img    = imread('118.bmp');                                       % original image
Img_bg = imread('118_mrk.bmp');                                   % images with marks (blue : background, red : foreground)
Img = Img(:,:,1);

iternum     = 500;
timestep    = 4;
alfa        = 1.5;
belta1      = 0.2/timestep;
belta2      = 1.5;
epsilon     = 1.5;

T=mean(Img(:));

[edge, phi,mask,s] = probilityLSF(Img_bg,Img,1);

figure, subplot(2,2,1); title('Initialization');
imshow(Img); hold on;
contour(phi, [0 0], 'r','LineWidth',1);
hold off;  drawnow;

subplot(2,2,2); title('Segmentation');
tic;
% start level set evolution
for n=1:iternum                              
     phi = edge_region_ac(double(Img), phi, edge, alfa, belta1, belta2, epsilon, timestep,n,edge,T);
     if mod(n,5)==0
        pause(0.1);
        imshow(Img); hold on;
        contour(phi, [0 0], 'r','LineWidth',1);        
     end
end
toc;
subplot(2,2,3); title('Segmentation');
phibw = im2bw(phi);
imshow(phibw);

