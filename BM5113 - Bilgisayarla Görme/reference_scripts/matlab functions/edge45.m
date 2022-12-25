function edge45(f)
f= im2double(f);
w45=[-1 -1 -1;...
     -1 8 -1;...
      -1 -1 -1];
g45=imfilter(f,w45,'replicate');
T=0.7*max(abs(g45(:)));
g45=abs(g45)>T;
figure,
imshow(g45,[0 1])
end