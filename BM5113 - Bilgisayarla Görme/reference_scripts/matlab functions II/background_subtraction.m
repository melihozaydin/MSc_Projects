close all;
figure;
boxPolygon = [177, 237;243,233;265,262;268,308;236,320;143,326;143,283;177, 237];
Bg = [];
frameNum_Bg = 10
alfa = 0.005;
threshold_fg = 0.5;
figure,
for i=0:frameNum_Bg-1
    %% Resim yükleme ve özellik gösterim
    if i<10
        frame_str = ['000' num2str(i)];
    elseif i<100
        frame_str = ['00' num2str(i)];
    else
        frame_str = ['0' num2str(i)];
    end

    leftImage= imread(['..\images\bridge-l\image' frame_str '_c0.png']);
    leftImage = im2double(histeq(leftImage));
    if i==0
        Bg = leftImage;
    else
        Bg = Bg + leftImage;
    end
end
    Bg = Bg/frameNum_Bg;

for i=frameNum_Bg:250    
    %% Resim yükleme ve özellik gösterim
    if i<10
        frame_str = ['000' num2str(i)];
    elseif i<100
        frame_str = ['00' num2str(i)];
    else
        frame_str = ['0' num2str(i)];
    end
    leftImage= imread(['..\images\bridge-l\image' frame_str '_c0.png']);
    % imhist(leftImage)
    leftImage = im2double(histeq(leftImage));
    Bg = alfa*leftImage+(1-alfa)*Bg;
    Fg = abs(leftImage-Bg);
    threshold_fg = max(Fg(:))*0.6;
    Fg = Fg > threshold_fg;
    subplot(1,3,1);
    imshow(leftImage); title('I_t');
    subplot(1,3,2);
    imshow(Bg); title('Background');
    subplot(1,3,3);
    imshow(Fg); title('Foreground');
    
    uiwait(gcf,0.1)
end



