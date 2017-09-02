function K_inter = subtract(x1,x2)
[r1,c1] = size(x1);
[r2,c2] = size(x2);
K_inter = zeros(r1,r2);
for i=1:r1
    for j=1:r2
        a = x1(i,:)-x2(j,:);
        %disp(a);
        K_inter(i,j) = norm(a);
        %disp(K_inter(i,j));
    end
end
end