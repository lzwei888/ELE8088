x = -3:0.01:3; 


y = exp(-x.^2); 

plot(x, y, 'b', 'LineWidth', 2);
grid on;
xlabel('x');
ylabel('f(x)');
title('f(x) = e^{-x^2}');
axis([-3,3,-3,3])

ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

hold on;
plot(x,-ones(size(x)),'r--','LineWidth', 1)
plot(x,ones(size(x)),'r--','LineWidth', 1)
legend('f(x)','boundaries')