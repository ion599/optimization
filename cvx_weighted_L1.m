%% Weighted L_1
function [x, z, a] = cvx_weighted_L1(p)
    Phi = p.Phi; f = p.f; n = p.n; L1 = p.L1; block_sizes = p.block_sizes;
    noise = p.noise; epsilon = p.epsilon; lambda = p.lambda; w = p.w;
    blocks = p.blocks;
    noise = 1;
    lambda = .1;
    
    N = p.block_sizes;
    lenN = length(N);
    lenZ = n-lenN;
    assert(sum(N) == n) % Check that nullspace N accounts for number of routes

    % Generate full N matrix
    N2 = zeros(n, n-lenN);
    x0 = zeros(n,1);

    ind = 1;
    k = 0;
    for i=1:length(N)
        N2(ind, ind-k) = 1;
        ind = ind+1;
        for j=2:(N(i)-1)
            N2(ind, ind-k-1) = -1;
            N2(ind, ind-k) = 1;
            ind = ind +1;
        end
        N2(ind, ind-k-1) = -1;
        k = k+1;
        x0(ind) = 1;
        ind = ind+1;
    end

    V = zeros(lenZ,lenN);
    k = 1;
    for i=1:lenN
        V(k:k+N(i)-2,i) = 1;
        k = k + N(i)-1;
    end

    % cvx program
    cvx_begin quiet
        variable z(lenZ)
        variable a(lenN)
%         if ~noise
%             minimize( norm(z - V * a, 1) )
%         else
%             minimize( square_pos(norm(Phi * (x_init + N2 * z) - f, 2)) + ...
%                 lambda * norm(z - V * a,1) )
%         end
        minimize( sum_square(Phi * (x0 + N2 * z) - f) + ...
                lambda * norm(z - V * a,1) )
        subject to
%         if ~noise
%             sum_square(Phi * (x0 + N2 * z) - f) <= epsilon
%         end
        x0 + N2 * z >= 0
%         k = 1;
%         for i=1:lenN
%             for j=1:N(i)-1
%                 if j == 1
%                     z(k) >= 0
%                     z(k) <= z(k+1)
%                 elseif j == N(i)-1
%                     z(k) <= 1
%                 else
%                     z(k) <= z(k+1)
%                 end
%                 k = k+1;
%             end
%         end
    cvx_end
    
    x = zeros(n,1);
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
    fprintf('norm(A*x-b): %8.5e\nnorm(A*x0-b): %8.5e\n', norm(Phi*x-f), norm(Phi*x0-f))

end