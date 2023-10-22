% set A and b
A=[1 1 1 0 0;0 1 1 1 0];
b=[1;-1];
% generate random matrix B
B=-1 + 2 * rand(5, 5);
% get Q
Q=B'*B+eye(5);
% calculate x* and lambda*
m=[Q A';A zeros(2)];
minv=inv(m);
% get matrix [x;lambda]
xlambda=minv*[-0.5;b;0;0;0;0];
% get x*
xstar_KKT=xlambda(1:5)