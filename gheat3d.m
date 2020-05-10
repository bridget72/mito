%load .mat with size(#num_iter, size of stack(:,:,:))
load timed0182.mat
I= arr(50, 1:99,1:320,1:290);
I = squeeze(I);
deltaT = 0.2;
STEPS = 50;
pausefor = 1;

for n = 1:STEPS 
    [Gx,Gy,Gz] = gradient(I);
    Gnorm = sqrt(Gx.^2+Gy.^2+Gz.^2) + eps;
    
    Gxnorm = Gx./Gnorm;
    Gynorm = Gy./Gnorm;
    Gznorm = Gz./Gnorm;
    
    Idiv = divergence(Gxnorm,Gynorm,Gznorm);
    
    I = I+ deltaT * (Idiv.*Gnorm);
end

bwImg = zeros(size(I)); 
bwImg(I>0) = 1;
iso=isosurface(bwImg, 0.5);
pp = patch(iso, 'FaceColor', 'none', 'EdgeColor', 'r', 'EdgeAlpha', 0.1);
% change to size of stack
axis([ 1 99 1 320 1 290 ])
axis equal
view(3)
    