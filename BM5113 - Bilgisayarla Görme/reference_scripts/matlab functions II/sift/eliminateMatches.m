function matchesElim=eliminateMatches(matches,points1,points2)
    matchesElim=[];
    for i = 1:size(matches,2)
        %Distance between correspondences
        d = sum( (points1(1:2,matches(1,i)) - points2(1:2,matches(2,i))).^2 );
        if d < 250
            %Inlier
            matchesElim=[matchesElim matches(:,i)];
        end
    end
end