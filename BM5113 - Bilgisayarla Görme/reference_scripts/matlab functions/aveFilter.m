function aveFilter()
    %- filtreleme
    ftest=imread('test.tif');
    figure
    imshow(ftest);
    boyut = 35 ;
    w = ones(boyut);
    w = w/ (boyut^2);
    ffilter =  imfilter(ftest,w);

    figure('Name','filtre boyut = 35 ') 
    imshow(ffilter);

end