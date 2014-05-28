function w = xProject(w,N)
k=0;
for i=1:length(N)
    w(k+1:k+N(i)) = SimplexProj(w(k+1:k+N(i))');
    k = k+N(i);
end