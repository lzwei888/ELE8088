rng(1)
% set A and b
A=[1 1 1 0 0;
     0 1 1 1 0];
b=[1;-1];
% calculate x0 and N
x0=A\b;     
N=null(A);
% generate random matrix B
B=-1 + 2 * rand(5, 5);
% get Q
Q=B'*B+eye(5);
% calculate x*
xstar=x0-N*((N'*Q*N)\(N'*(Q*x0+0.1)))

Vstar = 0.5 * xstar'*Q*xstar +[0.1,0.1,0.1,0.1,0.1]*xstar