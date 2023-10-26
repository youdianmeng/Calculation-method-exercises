% 采用隐式欧拉公式求解   y' = y -2x/y, 0 < x < 1,
%                       y(0) = 1.
% 的初值问题

x0 = 0; y0 = 1; b = 1; h = 0.1;
xy = [];
while x0 < b - h

    y = h * (y0 - 2*x0/y0) + y0;
    x = x0 + h;
    y = y0 + h * (y - 2*x/y);
    xy = [xy,[x; y]];

    x0 = x;
    y0 = y;
end
disp(xy)
figure('Name', '自己求解_and_matlab求解');
plot(xy(1, :), xy(2, :), '-or')
hold on;

% matlab自带库数值解
% 定义微分方程
dydt = @(x, y) y - 2*x/y;
yy0 = 1;
xspan = [0 1];

% 使用 ode45 求解
[xx, yy] = ode45(dydt, xspan, yy0);

plot(xx, yy, 'b-');
hold off;


