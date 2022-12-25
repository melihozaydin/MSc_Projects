I= imread('..\images\street.png');

T = [   cos(pi/3) -sin(pi/3) 0.0003;
        sin(pi/3) cos(pi/3) 0.001;
        0 0 1 ];
S=[   1 0 0;
      0 1 0;
      0 0 1 ];
P = S*T;
P(end,1:end)=[0.2 0.3 1];

t_proj = projective2d(P);

I_projective = imwarp(I,t_proj);
imshow(I_projective)
title('Projective');