x = -30:0.01:30; 


y = x.*sin(x); 

plot(x, y, 'b', 'LineWidth', 2);
grid on;
xlabel('x');
ylabel('f(x)');
title('f(x) = xsin(x)');
axis([-30,30,-30,30])

ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';