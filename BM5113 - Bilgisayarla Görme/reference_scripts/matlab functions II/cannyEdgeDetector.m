function Icanny=cannyEdgeDetector(I,sigma)
    %connected Component algoritmas�nda kullan�lan global de�i�keni tan�mla
    global labeledI;
    set(0,'RecursionLimit',3000);
    [row, column] = size(I);
    %1.Ad�m: Resmi 2D Gaussian ile bulan�kla�t�rma
    %Filitreyi �ret
    filter = Gauss2DFilter(sigma,0);
    ISmoothed = imfilter(I,filter,'replicate');
    %2.Ad�m: 1. dereceden t�revi Prewitt kullanarak hesapla. Di�er
    %filitreler test i�in olu�turuldu.
    prewittX = [-1 -1 -1;0 0 0;1 1 1];
    prewittY = [-1 0 1;-1 0 1;-1 0 1];
    sobelX = [-1 -2 -1;0 0 0;1 2 1];
    sobelY = [-1 0 1;-2 0 2;-1 0 1];
    gradX =[1;-1];
    gradY=[1 -1];
    IGradientX =  imfilter(ISmoothed,prewittX,'replicate');     
    IGradientY =  imfilter(ISmoothed,prewittY,'replicate');
    
    %3.Ad�m: T�rev de�er matrisi M ve a�� matrisi A'y� hesapla
    M = sqrt(IGradientX.^2+IGradientY.^2);
    M = M./max(M(:));
    figure('Name','Magnitude'), imshow(M./max(M(:)));
    
    %Radyan a�� d�ner
    A = atan(IGradientY./IGradientX);
    
    %4.Ad�m: Maksimum olmayan noktalar�n bast�r�m�
    directions = NaN(row,column);
    %Find directions
    directions((A>=-pi/8 & A <=pi/8)| (A<=-7*pi/8 & A>=7*pi/8)) = 0;        %horizontal
    directions((A>=pi/8 & A <=3*pi/8) | (A<=-5*pi/8 & A>=-7*pi/8)) = 1;     %-45    
    directions((A>=-5*pi/8 & A <=-3*pi/8) | (A<=5*pi/8 & A >=3*pi/8)) = 2;  %vertical 
    directions((A>=-3*pi/8 & A <=-pi/8) | (A<=7*pi/8 & A>=5*pi/8)) = 3;     %+45
%     figure,imshow(directions==0)
%     figure,imshow(directions==1)
%     figure,imshow(directions==2)
%     figure,imshow(directions==3)
    % Remove weak edges
    Gn = supress(M,directions);
    figure('Name','Removed weak edges'), imshow(Gn);
    
    %5.Ad�m: �ifte e�ik de�eri kullanarak hatal� kenar noktalar�n� temizle
    TLow = 0.05;
    THigh = TLow*2.5;    
    Gn(Gn<TLow) = 0;
    %E�ik de�eri y�ksek g��l� kenar noktalar�n�n 8-ba�l�l���ndaki zay�f kenar
    %noktalar�n� sapta. searchComponent fonksiyonunu kullan�r.
    labeledI = -(Gn>0);
    %G��l� kenar noktalar�n�n x ve y koordinatlar�n� oku
    [strongX,strongY] = ind2sub([row,column],find(Gn>=THigh));
    numOfStrongEdges = length(strongX);
    label =1;
    %Herbir g��l� kenar noktas� i�in
    for ind = 1:numOfStrongEdges
        labeledI(strongX(ind),strongY(ind)) = label;        
        searchComponent(label,strongX(ind),strongY(ind),8);        
    end
    %G��l� kenarlara ba�l� olmayan zay�f kenar noktalar�n� temizle.
    labeledI(labeledI==-1) = 0;
    %7.Ad�m: 1 pixelden daha kal�n olabilecek pikseller i�in 1 kez
    %zay�flatma i�lemini uygula   
    Icanny = bwmorph(labeledI,'thin',1);
   
end

function Gn=supress(magnitude,directions)
    [w,h] = size(magnitude);
    Gn = zeros(size(w,h));
    % T�rev de�erlerini kolayca kar��la�t�rabilmek i�in resmin �evresine 0
    % ekle. De�erlerin 0'dan b�y�k e�it oldu�u garanti oldu�undan problem de�il 
    magPadded = [zeros(1,h+2);zeros(w,1) magnitude zeros(w,1);zeros(1,h+2)];
    % Remove weak edges
    for i= 1:w
        for j= 1:h
            if (directions(i,j)==0 && magPadded(i+1,j+1)>= magPadded(i,j+1) && magPadded(i+1,j+1)>= magPadded(i+2,j+1))   ||... %horizontal 
                (directions(i,j)==1 && magPadded(i+1,j+1)>= magPadded(i,j) && magPadded(i+1,j+1)>= magPadded(i+2,j+2))    ||... %-45  
                (directions(i,j)==2 && magPadded(i+1,j+1)>= magPadded(i+1,j) && magPadded(i+1,j+1)>= magPadded(i+1,j+2))  ||... %vertical 
                (directions(i,j)==3 && magPadded(i+1,j+1)>= magPadded(i+2,j) && magPadded(i+1,j+1)>= magPadded(i,j+2))          %+45 
                    Gn(i,j) = magPadded(i+1,j+1);
            
            end
        end
    end
end
