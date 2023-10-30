rng(1)
% set A and b
A=[1 1 1 0 0;0 1 1 1 0];
b=[1;-1];
% generate random matrix B
B=-1 + 2 * rand(5, 5);
% get Q
Q=B'*B+eye(5);
% calculate x* and lambda*
K=[Q A';A zeros(2)];
q = 0.1 * ones(5, 1);
% get matrix [x;lambda]
xlambda = K \ [-q; b];
% get x*
xstar_KKT=xlambda(1:5)
Vstar_KKT = 0.5 * xstar_KKT'*Q*xstar_KKT +[0.1,0.1,0.1,0.1,0.1]*xstar_KKT