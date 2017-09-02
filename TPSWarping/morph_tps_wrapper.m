function morphed_im = morph_tps_wrapper(I, J, im1_pts, im2_pts, warp_frac, dissolve_frac)
im1_size = size(I);
im2_size = size(J);
% Resizing and Padding
if(im1_size(1)>im2_size(1))
    I = imresize(I,im2_size(1)/im1_size(1));
    im1_size = size(I);
end
if(im1_size(2)>im2_size(2))
    I = imresize(I,im2_size(2)/im1_size(2));
    im1_size = size(I);
end
if(im1_size(1)<im2_size(1) || im1_size(1)<im2_size(1))
    I = padarray(I, [im2_size(1)-im1_size(1), im2_size(2)-im1_size(2)], 'replicate', 'post');
end
% Intermediate points
impoints = (1 - warp_frac) * im1_pts + warp_frac * im2_pts;
% TPS 
[a1_x,ax_x,ay_x,w_x] = est_tps(impoints, im1_pts(:,1));
[a1_y,ax_y,ay_y,w_y] = est_tps(impoints, im1_pts(:,2));
morphed_im1 = morph_tps(I, a1_x, ax_x, ay_x, w_x, a1_y, ax_y, ay_y, w_y, impoints, [im2_size(1), im2_size(2)]);
[a1_x,ax_x,ay_x,w_x] = est_tps(impoints, im2_pts(:,1));
[a1_y,ax_y,ay_y,w_y] = est_tps(impoints, im2_pts(:,2));
morphed_im2 = morph_tps(J, a1_x, ax_x, ay_x, w_x, a1_y, ax_y, ay_y, w_y, impoints, [im2_size(1), im2_size(2)]);
% Cross dissolve
morphed_im = (1-dissolve_frac) * morphed_im1 + dissolve_frac * morphed_im2;
morphed_im = uint8(morphed_im);

end