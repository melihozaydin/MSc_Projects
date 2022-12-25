I = imread('C:\Users\ceyda\codes\cv\images\campus\p1070503.jpg');
Id = im2double(I);
Id = mean(Id, 3);
Iblur=imfilter(Id,fspecial('gaussian',3));
sobel_y=[-1 -2 -1; ...
            0 0 0;...
            1 2 1];
sobel_x = sobel_y';
I_x = imfilter(Iblur, sobel_x);
I_y = imfilter(Iblur, sobel_y);
gf = fspecial('gaussian',7);
A = imfilter((I_x.*I_x), gf);
B = imfilter((I_y.*I_y), gf);
C = imfilter((I_x.*I_y), gf);

det_M = A.*B-(C.*C);
trc_M = A+B;
k = 0.05;
Cornerness = det_M-k*(trc_M.^2);
max_c = max(Cornerness(:));
T = max_c*0.05;
Cornerness(Cornerness<T)=0;
W = [7 7];
% Window max cornerness search

Cornerness_dilated = imdilate(Cornerness>0,strel('disk',5));
figure, imshow (Cornerness_dilated)

red = I(:,:,1);
red(Cornerness_dilated) = 255;
Icorners = I;
Icorners(:,:,1) = red;
figure, imshow (Icorners)