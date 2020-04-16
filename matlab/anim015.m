load timed015.mat
% a 77*17*220*232 array
% where there are 77 iterations for the 3d evolving processes
% the orginal data 'ltr.tif' is of size 17*220*232


%{
% ploting the final contour
new = arr(40, 2:17, 50:200, 50:200);
subImg = squeeze(new);
bwImg = zeros(size(subImg)); 
bwImg(subImg>0) = 1; % this way the vertex normals point outwards

iso=isosurface(bwImg, 0.5);
figure
%figure('Renderer', 'painters', 'Position', [200 200 1000 1000])
patch(iso, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
axis([50 200 0 20 50 200])
axis equal
view(3)
%}

%{
figure
for k = 1 : 2 : 63
    new = arr(k, 2:17, 50:200, 50:200);
    subImg = squeeze(new);
    bwImg = zeros(size(subImg)); 
    bwImg(subImg>0) = 1;
    iso=isosurface(bwImg, 0.5);
    pp = patch(iso, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
    axis([50 200 0 20 50 200])
    axis equal
    view(3)
    title(sprintf('iter = %.1f', k));
    
    pause(0.1);
    clf
end
%}

% saving to video

myVideo = VideoWriter('test'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)

for k = 1 : 2 : 109
    new = arr(k, 1:50,1:200,1:200);
    subImg = squeeze(new);
    bwImg = zeros(size(subImg)); 
    bwImg(subImg>0) = 1;
    iso=isosurface(bwImg, 0.5);
    pp = patch(iso, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
    axis([1 100 1 100 1 100])
    axis equal
    view(3)
    title(sprintf('iter = %.1f', k));
    pause(0.2);
    frame = getframe(gcf); %get frame
    writeVideo(myVideo, frame);
    clf
end
close(myVideo)

