function Icanny=cannyEdgeDetector(I,sigma)
    %connected Component algoritmasýnda kullanýlan global deðiþkeni tanýmla
    global labeledI;
    set(0,'RecursionLimit',3000);
    [row, column] = size(I);
    %1.Adým: Resmi 2D Gaussian ile bulanýklaþtýrma
    %Filitreyi üret
    filter = Gauss2DFilter(sigma,0);
    ISmoothed = imfilter(I,filter,'replicate');
    %2.Adým: 1. dereceden türevi Prewitt kullanarak hesapla. Diðer
    %filitreler test için oluþturuldu.
    prewittX = [-1 -1 -1;0 0 0;1 1 1];
    prewittY = [-1 0 1;-1 0 1;-1 0 1];
    sobelX = [-1 -2 -1;0 0 0;1 2 1];
    sobelY = [-1 0 1;-2 0 2;-1 0 1];
    gradX =[1;-1];
    gradY=[1 -1];
    IGradientX =  imfilter(ISmoothed,prewittX,'replicate');     
    IGradientY =  imfilter(ISmoothed,prewittY,'replicate');
    
    %3.Adým: Türev deðer matrisi M ve açý matrisi A'yý hesapla
    M = sqrt(IGradientX.^2+IGradientY.^2);
    M = M./max(M(:));
    figure('Name','Magnitude'), imshow(M./max(M(:)));
    
    %Radyan açý döner
    A = atan(IGradientY./IGradientX);
    
    %4.Adým: Maksimum olmayan noktalarýn bastýrýmý
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
    
    %5.Adým: Çifte eþik deðeri kullanarak hatalý kenar noktalarýný temizle
    TLow = 0.05;
    THigh = TLow*2.5;    
    Gn(Gn<TLow) = 0;
    %Eþik deðeri yüksek güçlü kenar noktalarýnýn 8-baðlýlýðýndaki zayýf kenar
    %noktalarýný sapta. searchComponent fonksiyonunu kullanýr.
    labeledI = -(Gn>0);
    %Güçlü kenar noktalarýnýn x ve y koordinatlarýný oku
    [strongX,strongY] = ind2sub([row,column],find(Gn>=THigh));
    numOfStrongEdges = length(strongX);
    label =1;
    %Herbir güçlü kenar noktasý için
    for ind = 1:numOfStrongEdges
        labeledI(strongX(ind),strongY(ind)) = label;        
        searchComponent(label,strongX(ind),strongY(ind),8);        
    end
    %Güçlü kenarlara baðlý olmayan zayýf kenar noktalarýný temizle.
    labeledI(labeledI==-1) = 0;
    %7.Adým: 1 pixelden daha kalýn olabilecek pikseller için 1 kez
    %zayýflatma iþlemini uygula   
    Icanny = bwmorph(labeledI,'thin',1);
   
end

function Gn=supress(magnitude,directions)
    [w,h] = size(magnitude);
    Gn = zeros(size(w,h));
    % Türev deðerlerini kolayca karþýlaþtýrabilmek için resmin çevresine 0
    % ekle. Deðerlerin 0'dan büyük eþit olduðu garanti olduðundan problem deðil 
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
