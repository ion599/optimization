function [obj,g] = objectiveSparse(z,A,N_sparse,x0,b,alpha)

%{
Function that returns the the value of the objective and the gradient
N is given as the # of lines of each of its block
N = [n1; n2; ...; np]
%}
u = x0+N_sparse*z;
obj = A*u-b;
temp = A' * obj + alpha.*u;
obj = .5 * ((obj'*obj) + alpha'*(u.*u));
g = N_sparse' * temp;
