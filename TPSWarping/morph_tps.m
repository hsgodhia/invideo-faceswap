function morphed_im = morph_tps(im_source, a1_x, ax_x, ay_x, w_x, a1_y, ax_y, ay_y, w_y, ctr_pts, sz)
im1_size = size(im_source);
% Resizing and Padding
if(im1_size(1)>sz(1))
    im_source = imresize(im_source,sz(1)/im1_size(1));
    im1_size = size(im_source);
end
if(im1_size(2)>sz(2))
    im_source = imresize(im_source,sz(2)/im1_size(2));
    im1_size = size(im_source);
end
if(im1_size(1)<sz(1) || im1_size(2)<sz(2))
    im_source = padarray(im_source, [sz(1)-im1_size(1), sz(2)-im1_size(2)], 'replicate', 'post');
end
morphed_im = zeros(sz(1),sz(2),3);
for i=1:sz(1)
    for j=1:sz(2)
        point = [j i];
        inter = subtract(ctr_pts,point);
        inter(inter==0) = 1;
        final = U(inter);
        
        % different for x and y from here
        % for x
        w_final_x = w_x.*final;
        total_x = sum(w_final_x);
        x_value = a1_x + ax_x*j + ay_x*i + total_x;
        % for y
        w_final_y = w_y.*final;
        total_y = sum(w_final_y);
        y_value = a1_y + ax_y*j + ay_y*i + total_y;
        % round values 
        x_value = round(x_value);
        y_value = round(y_value);
        if(x_value<1)
            x_value = 1;
        end
        if(y_value<1)
            y_value = 1;
        end
        if(x_value>sz(2))
            x_value = sz(2);
        end
        if(y_value>sz(1))
            y_value = sz(1);
        end
        try
            morphed_im(i,j,:) = im_source(y_value,x_value,:);
        catch
            disp('y_value');
            disp(y_value);
            disp('x_value');
            disp(x_value);
        end
    end
end
end