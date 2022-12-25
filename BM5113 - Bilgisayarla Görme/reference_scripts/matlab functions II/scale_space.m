I=im2double(imread('../images/street.png'));
figure;
close all;
increaseSigma = 5;
maxSigma = 26;
out = zeros(size(I,1),size(I,2),floor(maxSigma/increaseSigma)+1);
for sig = 1:increaseSigma:maxSigma
    indexAxis = floor((sig-1)/increaseSigma)+1;
    out(:,:,indexAxis) = imgaussfilt(I,sig);
    subplot(2,3,indexAxis);
    imshow(out(:,:,indexAxis));
    title(['Gauss blur, sigma=' num2str(sig)])
end
figure;
for i= 1:floor(maxSigma/increaseSigma)
    DoG = out(:,:,i)-out(:,:,i+1);
    subplot(2,3,i),imshow(DoG, [min(DoG(:)) max(DoG(:))]);
end