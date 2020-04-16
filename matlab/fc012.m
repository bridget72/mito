load timed012.mat


new = arr(70, 1:20,1:200,1:200);
subImg = squeeze(new);
subImg = smooth3(subImg,'gaussian');
bwImg = zeros(size(subImg)); 
bwImg(subImg>0) = 1;
iso=isosurface(bwImg, 0.5);
pp = patch(iso, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
axis([ 1 200 1 20 1 200  ])
axis equal
view(3)
