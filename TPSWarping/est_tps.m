function [a1,ax,ay,w] = est_tps(ctr_pts, target_value)
[r,c] = size(ctr_pts);
% derive K
K_inter = subtract(ctr_pts,ctr_pts);
%disp('K_inter size');
%disp(size(K_inter));
K = U(K_inter);
%K(isinf(K)) = 0;
K(isnan(K)) = 1;
K(K==0) = 1;
%disp('K size');
%disp(size(K));
% derive P
col = ones(r,1);
P = [ctr_pts col];
% derive I
I = eye(r+3,r+3);
% v vector
z = zeros(3,1);
target_new = [target_value;z];
% choose lambda
lambda = 0.0000001;
% put all together
a = zeros(3,3);
big_matrix_x = [K P];
big_matrix_y = [transpose(P) a];
big_matrix = [big_matrix_x;big_matrix_y];
last = (pinv(big_matrix+(lambda*I)))*target_new;
%disp(last);
%disp(size(last));
w = last(1:r,1);
ax = last(r+1,1);
ay = last(r+2,1);
a1 = last(r+3,1);
end