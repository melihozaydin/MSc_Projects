function filter=LoGFilter(sigma)
%Boyutun tam say� ve tek olmas�n� garantile
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
%Filitre de�erleri toplam�n� 1'e e�itler.
filter= filter./sumFilt;
%Filitreyi tersine �evir. ��lem s�ras�nda bu kullan�l�r.
filter= -filter;
figure,mesh(filter);
end