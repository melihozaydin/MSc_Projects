function filter=Gauss2DFilter(sigma,mu)
%Boyutun tam sayý ve tek olmasýný garantile
size = ceil(sigma*3);
if mod(size,2)==0
    size=size+1;
end

x=(-size:1:size);
y=(-size:1:size);

for i=1:length(x)
    for j=1:length(y)
        filter(i,j)= (1/(sqrt(2*pi)*sigma))*exp(-1.*((x(i)^2+y(j)^2))/(2*sigma^2));
    end
end
filter= filter./sum(filter(:));
figure,mesh(filter);