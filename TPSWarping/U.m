function K = U(K)
[r,c] = size(K);
for i=1:r
    for j=1:c
        K(i,j) = -((K(i,j)^2)*log(K(i,j)^2));
    end
end
end