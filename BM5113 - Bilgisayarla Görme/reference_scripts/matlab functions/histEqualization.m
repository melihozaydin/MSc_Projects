function histEqualization()
    f=imread('washedPolen.tif');
    figure
    imshow(f);
    hist =imhist(f);
    [m n] = size(f);
    figure
    plot(hist);
    fNew = f;
    hist2 = hist(:)/(m*n) ;
    hist3 = cumsum(hist2);  
    hist4 = round(hist3(:)*256);
    figure
    plot(hist4);
    for k = 1 : 256
    indx = find (f ==k);
    fNew(indx) = hist4(k);
    end
    figure
    imshow(fNew);
end