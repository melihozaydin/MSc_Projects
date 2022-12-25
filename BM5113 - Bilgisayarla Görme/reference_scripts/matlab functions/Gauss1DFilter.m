function filter=Gauss1DFilter(sigma)
%Boyutun tam sayý ve tek olmasýný garantile
size = ceil(sigma*3);
if mod(size,2)==0
    size=size+1;
end
x=(-size:1:size);
h = exp(-(x.^2)/(2*sigma^2));
%h = (1/(sqrt(2*pi)*sigma))*h;
filter= h./sum(h(:));
%figure,plot(filter);
end