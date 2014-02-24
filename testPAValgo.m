clear all; clc;

y = [4;5;1;6;8;7];
w = ones(6,1);
x = PAValgo(y,w,-Inf,Inf);

x2 = PAValgo(y,w,4,7);

x3 = PAValgo(y,w,8,10);
