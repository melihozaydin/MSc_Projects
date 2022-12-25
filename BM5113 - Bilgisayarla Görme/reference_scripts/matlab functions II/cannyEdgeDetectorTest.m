function cannyEdgeDetectorTest(IName)
close all;
I = im2double(imread(IName));

for sigma = 4:4
    Icanny = cannyEdgeDetector(I,sigma);   
    figure('Name',[IName num2str(sigma)]), imshow(Icanny);
    imwrite(Icanny,['canny_' num2str(sigma) '_' IName]);
    
end

end