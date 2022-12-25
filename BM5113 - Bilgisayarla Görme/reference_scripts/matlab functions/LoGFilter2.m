function filter=LoGFilter2(sigma)
%LoG Filitrenin elemanlar� toplam�n�n 0 olmas� i�in Gauss fonksiyonunun
%Laplacian� al�n�r. B�ylece sabit yo�unluklu alanlarda herhangi bir de�i�im
%de�eri d�n�lmemesi sa�lan�r.Kenar tespiti sonucunu fazla etkilemedi�inden 
%LoG filitreleme s�ras�nda kullan�lmam��t�r. 
filter = Gauss2DFilter(sigma,0);
laplacian = [1 1 1;...
            1 -8 1;...
             1 1 1];
filter = imfilter(filter,laplacian,'replicate');
sumFilt = sum(filter(:));
filter= filter./sumFilt;
%Filitreyi tersine �evir. ��lem s�ras�nda bu kullan�l�r.
filter= -filter;
figure,mesh(filter);
end