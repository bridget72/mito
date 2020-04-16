myVideo = VideoWriter('3dsmooth016'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)

for k = 1 : 2 : 107
    new = arr(k, 1: 100,1:200,1:200);
    subImg = squeeze(new);
    subImg = smooth3(subImg,'gaussian',11,1);
    bwImg = zeros(size(subImg)); 
    bwImg(subImg>0) = 1;
    iso=isosurface(bwImg, 0.5);
    pp = patch(iso, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
    axis([1 200 1 100 1 200])
    axis equal
    view(3)
    title(sprintf('iter = %.1f', k));
    pause(0.2);
    frame = getframe(gcf); %get frame
    writeVideo(myVideo, frame);
    clf
end
close(myVideo)