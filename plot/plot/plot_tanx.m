ezplot('tan(x)')
grid on;
xlabel('x');
ylabel('y');
title('f(x) = tan(x)');

ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

hold on;
plot(0,0,'r*','LineWidth',3);
plot(pi,0,'r*','LineWidth',3);
plot(-pi,0,'r*','LineWidth',3);


legend('f(x)=tan(x)','stationary points')


