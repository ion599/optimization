function w = zProject(w,N)
k=0;
for i=1:length(N)
    w(k+1:k+N(i)-1) = PAValgo(w(k+1:k+N(i)-1),ones(N(i)-1,1),0,1);
    k = k+N(i)-1;
end