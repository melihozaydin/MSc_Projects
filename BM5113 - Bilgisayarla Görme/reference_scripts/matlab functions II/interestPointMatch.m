close all;
figure;
boxPolygon = [177, 237;243,233;265,262;268,308;236,320;143,326;143,283;177, 237];

for i=0:250
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
    leftImage = histeq(leftImage);
    %imhist(leftImage)
    %figure;
    %imshow(leftImage);
    %title('yol sol kamera');
    %boxImage = leftImage(237:319,140:270,:);
    %boxImage = mean(boxImage,3);
    %figure(),
    %imshow(boxImage)
    
    %% Karşılaştırılacak sayfa
    % Read the target image containing a cluttered scene.
    rightImage = imread(['..\images\bridge-r\image' frame_str '_c1.png']);
    % imhist(rightImage)
    rightImage = histeq(rightImage);
    %imhist(rightImage)
    %figure; 
    %imshow(rightImage);
    %title('yol sag kamera');
    
    %% Step 2: Detect Feature Points
    % Detect feature points in both images.
    leftPoints = detectSURFFeatures(leftImage);
    rightPoints = detectSURFFeatures(rightImage);
    
    %% 
    % Sol Kamera Odak noktaları.
%     figure; 
%     imshow(leftImage);
%     title('Sol Kamera Odak noktaları ');
%     hold on;
%     plot(selectStrongest(leftPoints, 50));
%     
    %% 
    % Sag Kamera Odak noktaları.
%     figure; 
%     imshow(rightImage);
%     title('Sag Kamera Odak noktaları');
%     hold on;
%     plot(selectStrongest(rightPoints, 50));
    
    %% Step 3: Belirgin noktaların belirlenmesi
    % sag sol kamera odak noktalarının belirlenmesi
    [leftFeatures, leftPoints] = extractFeatures(leftImage, leftPoints);
    [rightFeatures, rightPoints] = extractFeatures(rightImage, rightPoints);
    
    %% Step 4: 
    % Sol kamera ile eşleşilen kısımların çıkarılması 
    matchedPairs = matchFeatures(leftFeatures, rightFeatures);
    
    %% 
    % sag sol kamera eşit yerlerinin çıkarılması. 
    matchedLeftPoints = leftPoints(matchedPairs(:, 1), :);
    matchedRightPoints = rightPoints(matchedPairs(:, 2), :);
    
%     showMatchedFeatures(leftImage, rightImage, matchedLeftPoints, ...
%         matchedRightPoints, 'montage');
%     title('Putatively Matched Points (Including Outliers)');

    %% Step 5: varsayılan odak noktalarının belirlenmes,
    [tform, inlierIdx] = ...
        estimateGeometricTransform2D(matchedLeftPoints, matchedRightPoints, 'affine');
    inlierLeftPoints   = matchedLeftPoints(inlierIdx, :);
    inlierRightPoints = matchedRightPoints(inlierIdx, :);
    
    
    % odak olmayan noktaların çıkarılması
    showMatchedFeatures(leftImage, rightImage, inlierLeftPoints, ...
        inlierRightPoints, 'montage');
    title('Matched Points (Inliers Only)');

    %% 
    % Get the bounding polygon of the reference image.
    % boxPolygon = [1, 1;...                           % top-left
    %         size(boxImage, 2), 1;...                 % top-right
    %         size(boxImage, 2), size(boxImage, 1);... % bottom-right
    %         1, size(boxImage, 1);...                 % bottom-left
    %         1, 1];                   % top-left again to close the polygon
    
    %%
    % Transform the polygon into the coordinate system of the target image.
    % The transformed polygon indicates the location of the object in the
    % scene.
    newBoxPolygon = transformPointsForward(tform, boxPolygon);    
    boxPolygon = newBoxPolygon;
    %%
    % Display the detected object.
    % figure;
    % imshow(rightImage);
    hold on;
    line(size(rightImage,2)+newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y','LineWidth',3);
    %title('Detected Box');
    uiwait(gcf,1)
    % drawnow;
end



