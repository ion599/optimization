function [obj,g] = objectiveX(x,A,b,alpha)

obj = A*x-b;
g = A' * obj + alpha.*x;
obj = .5 * ((obj'*obj) + alpha'*(x.*x));