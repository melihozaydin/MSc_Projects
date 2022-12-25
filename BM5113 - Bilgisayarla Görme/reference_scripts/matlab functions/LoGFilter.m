function filter=LoGFilter(sigma)
%Boyutun tam sayý ve tek olmasýný garantile
size = ceil(sigma*3);
if mod(size,2)==0
    size=size+1;
end

x=(-size:1:size);
y=(-size:1:size);

for i=1:length(x)
    for j=1:length(y)
        filter(i,j)= (((x(i)^2+y(j)^2)-2*sigma^2)/(sigma^4))*exp(-((x(i)^2+y(j)^2))/(2*sigma^2));
    end
end
sumFilt = sum(filter(:));
%Filitre deðerleri toplamýný 1'e eþitler.
filter= filter./sumFilt;
%Filitreyi tersine çevir. Ýþlem sýrasýnda bu kullanýlýr.
filter= -filter;
figure,mesh(filter);
end