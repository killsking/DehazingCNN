setup() ;



net1=load('net-epoch-1726.mat');
net1.net.layers(end) = [] ;
net = vl_simplenn_tidy(net1.net) ;
%% Part 3.3: learn the model

%
imname='examples/sam_3.bmp';
im =imread(imname);
im=im2double(im);
figure, imshow(im , []);
method = 'he'; 
wsz = 15; % window size
A = Airlight( im, method, wsz); 

im(:,:,1)=im(:,:,1)-A(1);
im(:,:,2)=im(:,:,2)-A(2);
im(:,:,3)=im(:,:,3)-A(3);
im=single(im);
% Evaluate network on an image
res = vl_simplenn(net,im) ;
t=res(end).x;
J=im;
%t=t-(1-t)*0.1;
J(:,:,1)=t(:,:,1)+A(1);
J(:,:,2)=t(:,:,2)+A(2);
J(:,:,3)=t(:,:,3)+A(3);

ts=sqrt( im(:,:,1).^2 + im(:,:,2).^2 +im(:,:,3).^2 )./sqrt( t(:,:,1).^2 + t(:,:,2).^2 +t(:,:,3).^2 );
im =imread(imname);
trans_lower_bound = 1 - min(bsxfun(@rdivide,im2double(im),reshape(A,1,1,3)) ,[],3);
ts = max(ts, trans_lower_bound);
A=A*255;

r0 = 50;
eps = 10^-3; 
gray_I =im2double( rgb2gray(im));

tt= guidedfilter(gray_I, ts ,r0, eps);
ts=max(min(1,tt),0.1);
%lambda = 1;  % regularization parameter, the more this parameter, the closer to the original patch-wise transmission
%ts = CalTransmission(im, ts, lambda, 0.5);
%ts=max(min(1,ts),0.1);

r=Dehazefun(im, ts, A, 1);




adj_percent = [0.005, 0.995];
rImg = adjust(r,adj_percent);
figure, imshow(rImg , []);
