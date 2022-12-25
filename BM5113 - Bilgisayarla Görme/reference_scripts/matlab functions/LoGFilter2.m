function filter=LoGFilter2(sigma)
%LoG Filitrenin elemanlarý toplamýnýn 0 olmasý için Gauss fonksiyonunun
%Laplacianý alýnýr. Böylece sabit yoðunluklu alanlarda herhangi bir deðiþim
%deðeri dönülmemesi saðlanýr.Kenar tespiti sonucunu fazla etkilemediðinden 
%LoG filitreleme sýrasýnda kullanýlmamýþtýr. 
filter = Gauss2DFilter(sigma,0);
laplacian = [1 1 1;...
            1 -8 1;...
             1 1 1];
filter = imfilter(filter,laplacian,'replicate');
sumFilt = sum(filter(:));
filter= filter./sumFilt;
%Filitreyi tersine çevir. Ýþlem sýrasýnda bu kullanýlýr.
filter= -filter;
figure,mesh(filter);
end