I=imread('images/street.png');
Id = im2double(I);
G=fspecial('gaussian',[43 43],7);
xderf=[-1 1];
yderf = xderf';
Gx = imfilter(G,xderf);
Gy = imfilter(G,yderf);
figure, imshow([Gx Gy],[min(Gx(:)) max(Gx(:))])
sobelx = [-1 0 1;
          -2 0 2;
          -1 0 1];
sobely = sobelx';
Id_Gx = imfilter(Id,Gx);
Id_Gy = imfilter(Id,Gy);
Id_Sx = imfilter(Id,sobelx);
Id_Sy = imfilter(Id,sobely);

I_Gradient = sqrt(Id_Gx.^2+Id_Gy.^2);
I_Angle = atan2d(Id_Gy,Id_Gx);

%% LOG Filter
Gxx = imfilter(Gx,xderf);
Gyy = imfilter(Gy,yderf); 
LoGfilter = Gxx+Gyy;
figure, imshow(LoGfilter,[min(LoGfilter(:)) max(LoGfilter(:))])
