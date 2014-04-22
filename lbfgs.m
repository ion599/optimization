function d = lbfgs(y, s, g, ym, sm, rhom, m)

alpha = zeros(1,m);

q = g;

for j=2:m
    alpha(m-j+1) = rhom(m-j+1)*sm(:,m-j+1)'*q;
    q = q - alpha(m-j+1)*ym(:,m-j+1);
end

gamma = (s'*y)/(y'*y);
r = gamma*q;

for j=1:(m-1)
    beta = rhom(j)*ym(:,j)'*r;
    r = r + sm(:,j)*(alpha(j)-beta);
end

d = -r;