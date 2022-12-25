function edge45minus(f)
f= im2double(f);
w45=[0 -1 -2;1 0 -1;2 1 0];
g45=imfilter(f,w45,'replicate');
T=0.3*max(abs(g45(:)));
g45=abs(g45)>T;
figure,
imshow(g45,[0 1])
end