function x = PAValgo(y,w,l,u)

%{
Function that solves the IRC problem
    min. sum_i w_i(y_i-x_i)^2
    s.t. l <= x_1 <= ... <= x_n <= u
using PAV algorithm described in
Best and Chakravarti. Active Set Algorithms for isotonic regression; a unifying framework
%}

n = length(y);
x = y;

if n>1
    J = 1:(n+1); % J contains the first index of each block
    indB = 1;
    %n
    %length(w)
    while indB < length(J)-1
        %fprintf('Loop\n')
        %lenJ
        %length(J)
        %indB
        %n
        if avg(y,w,J,indB+1) < avg(y,w,J,indB)
            
            J(indB+1) = [];
            %lenJ = lenJ-1;
            
            while indB > 1 && avg(y,w,J,indB-1) > avg(y,w,J,indB)
                %fprintf('Inner Loop\n')
                if avg(y,w,J,indB) <= avg(y,w,J,indB-1)
                    J(indB) = [];
                    %lenJ = lenJ-1;
                    indB = indB-1;
                end
            end
            
        else
            
            indB = indB+1;
        end
        
    end
    %fprintf('Put x together\n')
    for i=1:length(J)-1
        ind = J(i):(J(i+1)-1);
        x(ind) = avg(y,w,J,i)*ones(length(ind),1);
    end
end

if l <= u
    x(x<l) = l;
    x(x>u) = u;
else
    fprintf('We must have l<=u, projection aborted!')
end

end

%=============================

function av = avg(y,w,J,indB)

%{
Function that computes average
%}
ind = J(indB):(J(indB+1)-1); % indices of B
%max(ind)
%length(w)
wB = w(ind);
av = wB'*y(ind) / sum(wB);

end