function x = z2x(z,N)

% function that maps z to x using x=x0+Nz

lenN = length(N);
x = zeros(sum(N),1);
ind = 1;
k = 0;
for i=1:lenN
    x(ind) = z(ind-k);
    ind = ind+1;
    for j=2:(N(i)-1)
        x(ind) = z(ind-k)-z(ind-k-1);
        ind = ind+1;
    end
    x(ind) = -z(ind-k-1)+1;
    k = k+1;
    ind = ind+1;
end