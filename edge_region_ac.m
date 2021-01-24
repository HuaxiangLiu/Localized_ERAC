%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of "Localized edge-region-based active contour model by incorporating
% the probability scores for medical image segmentation" 
% Huaxiang Liu
% East China University of Technology&&Central South University, Changsha, 
% China
% 6th, March, 2020
% Email: felicia_liu@126.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function phi = edge_region_ac(Img, phi_0, g, alfa, belta1,belta2, epsilon, timestep, iter,edge,T)

  phi=phi_0;
  phi=NeumannBoundCond(phi);
  %% The reguarization term
  diracPhi=Dirac(phi,epsilon);
  k = curvature_central(phi);
  reguarizationterm = diracPhi.*k;
  
  %% The penerty term
  penertyTerm=penertyterm(phi);
  
  %% The data term  
  [phi_x,phi_y]=gradient(phi); 
  s=sqrt(phi_x.^2 + phi_y.^2);

  idx = find(phi <= 1.2 & phi >= -1.2);  %get the curve's narrow band
    
  %-- find interior and exterior mean
  upts_local = find(phi <= 1.2 &phi>=0);                 % interior points
  vpts_local = find(phi >= -1.2&phi<0);                  % exterior points
  upts = find(phi<=0);                 % interior points
  vpts = find(phi>0);                  % exterior points
  c1 = sum(Img(upts))/(length(upts)+eps); % interior mean
  c2 = sum(Img(vpts))/(length(vpts)+eps); % exterior mean
  u_local = sum(Img(upts_local))/(length(upts_local)+eps); % interior mean
  v_local = sum(Img(vpts_local))/(length(vpts_local)+eps); % exterior mean
    
    
  F = (Img(idx)-0.5*(c1+u_local)).^2-(Img(idx)-0.5*(c2+v_local)).^2;   % region-based force from image information
  curvature = reguarizationterm(idx);  % force from curvature penalty
    
  % note that d phi/dt= - d E/d phi
  dphidt = g(idx).*F./max(abs(F)) + belta1*penertyTerm(idx)+ belta2*curvature-signed(c1,c2,T)*alfa*diracPhi(idx).*g(idx);  % gradient descent to minimize energy
    
  %-- maintain the CFL condition
  dt = .12/(max(dphidt)+eps);
    
  %-- evolve the curve
  phi(idx) = phi(idx) + dt*dphidt; 
    
  phi = sussman(phi, .5);
    
% else
%     dataterm = alfa*diracPhi.*g;
%     phi = phi + timestep*(dataterm+ belta1*penertyTerm + belta2*reguarizationterm);
% end
% phi = sussman(phi, .5);




function D = sussman(D, dt)
  % forward/backward differences
  a = D - shiftR(D); % backward
  b = shiftL(D) - D; % forward
  c = D - shiftD(D); % backward
  d = shiftU(D) - D; % forward
  
  a_p = a;  a_n = a; % a+ and a-
  b_p = b;  b_n = b;
  c_p = c;  c_n = c;
  d_p = d;  d_n = d;
  
  a_p(a < 0) = 0;
  a_n(a > 0) = 0;
  b_p(b < 0) = 0;
  b_n(b > 0) = 0;
  c_p(c < 0) = 0;
  c_n(c > 0) = 0;
  d_p(d < 0) = 0;
  d_n(d > 0) = 0;
  
  dD = zeros(size(D));
  D_neg_ind = find(D < 0);
  D_pos_ind = find(D > 0);
  dD(D_pos_ind) = sqrt(max(a_p(D_pos_ind).^2, b_n(D_pos_ind).^2) ...
                       + max(c_p(D_pos_ind).^2, d_n(D_pos_ind).^2)) - 1;
  dD(D_neg_ind) = sqrt(max(a_n(D_neg_ind).^2, b_p(D_neg_ind).^2) ...
                       + max(c_n(D_neg_ind).^2, d_p(D_neg_ind).^2)) - 1;
  
  D = D - dt .* sussman_sign(D) .* dD;
  
%-- whole matrix derivatives
function shift = shiftD(M)
  shift = shiftR(M')';

function shift = shiftL(M)
  shift = [ M(:,2:size(M,2)) M(:,size(M,2)) ];

function shift = shiftR(M)
  shift = [ M(:,1) M(:,1:size(M,2)-1) ];

function shift = shiftU(M)
  shift = shiftL(M')';
  
function S = sussman_sign(D)
  S = D ./ sqrt(D.^2 + 1);    
  
  
function f = penertyterm(phi)
% compute the distance regularization term with the double-well potential p2 in eqaution (16)
[phi_x,phi_y]=gradient(phi);
s=sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
% ps=a.*sin(2*pi*s)/(2*pi)+b.*(s-1);  % compute first order derivative of the double-well potential p2 in eqaution (16)
ps = 2*a.*s/3.*(s-2/3)+b.*(s-1);
dps=((ps~=0).*ps+(ps==0))./((s~=0).*s+(s==0));  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
f = div(dps.*phi_x - phi_x, dps.*phi_y - phi_y) + 4*del2(phi);  

function k = curvature_central(u)
% compute curvature for u with central difference scheme
[ux,uy] = gradient(u);
normDu = sqrt(ux.^2+uy.^2+1e-10);
Nx = ux./normDu;
Ny = uy./normDu;
[nxx,junk] = gradient(Nx);
[junk,nyy] = gradient(Ny);
k = nxx+nyy;

function f = Dirac(x, sigma)
% f=(1/2/sigma)*(1+cos(pi*x/sigma));
% b = (x<=sigma) & (x>=-sigma);
% f = f.*b;
f=(1/pi)*sigma./(sigma.^2+x.^2);
b = (x<=sigma) & (x>=-sigma);
f = f.*b;


function f = div(nx,ny)
[nxx,junk]=gradient(nx);  
[junk,nyy]=gradient(ny);
f=nxx+nyy;

function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);

function s = signed(c1,c2,T)
 if abs(c1-c2)>T*0.01
     s =-1;
 elseif c1==c2
     s =0;
 else
     s=1;
 end
