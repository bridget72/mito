load timed016.mat


new = arr(107, 1:100,1:200,1:200);
subImg = squeeze(new);
subImg = smooth3(subImg,'gaussian',11,1);
bwImg = zeros(size(subImg)); 
bwImg(subImg>0) = 1;
iso=isosurface(bwImg, 0.5);
pp = patch(iso, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
axis([1 200 1 100 1 200])
axis equal
view(3)
