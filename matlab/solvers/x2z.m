function z = x2z(x,N)

% function that maps x to z using x=x0+Nz

lenN = length(N);
z = zeros(sum(N)-lenN,1);
ind = 1;
k = 0;
for i=1:lenN
    if N(i)>1
        z(ind) = x(ind+k);
        ind = ind+1;
        for j=1:(N(i)-2)
            z(ind) = x(ind+k) + z(ind-1);
            ind = ind+1;
        end 
    end
    %z(ind:ind+N(i)-2) = cumsum(x(ind+k:ind+N(i)-2+k));
    %ind = ind+N(i)-1;
    k = k+1;
end