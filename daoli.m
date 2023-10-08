clear

% 倒立摆模型的状态方程
g = 9.8; % 重力加速度
l = 1; % 摆杆长度
m = 1; % 摆杆质量

A = [0 1; g/l 0];
B = [0; -1/(m*l^2)];
C = [1 0];

% 自适应控制器的代码
k1 = 5;
k2 = 5;
k3 = 0.5;
k4 = 0.5;

x0 = [0.1; 0]; % 初始状态
r = 0; % 参考输入
v = 0; % 自适应增益
w = 0; % 干扰估计器的自适应增益
e = r - C*x0; % 初始控制误差
f = 0; % 初始干扰估计误差

tspan = [0 10]; % 时间范围
[t,x] = ode45(@(t,x) invpend_adapt(t,x,A,B,C,k1,k2,k3,k4,r,e,v,f,w), tspan, x0);

% 画出控制误差和控制输出随时间的变化曲线
figure
subplot(2,1,1)
plot(t, x(:,1))
xlabel('Time (s)')
ylabel('Control Error')
title('Control Error vs. Time')

subplot(2,1,2)
plot(t, x(:,2))
xlabel('Time (s)')
ylabel('Control Output')
title('Control Output vs. Time')

function dx = invpend_adapt(t,x,A,B,C,k1,k2,k3,k4,r,e,v,f,w)
% 自适应控制器的代码
% t: 时间
% x: 状态向量
% A,B,C: 倒立摆模型的状态方程
% k1,k2,k3,k4: 自适应增益
% r: 参考输入
% e: 控制误差
% v: 自适应增益
% f: 干扰估计误差
% w: 干扰估计器的自适应增益

u = -k1*e - k2*x(2) + r + v*C*x; % 控制输入
y = C*x; % 控制输出
f_hat = -w*sign(y - C*x); % 干扰估计器的输出

% 更新控制误差、自适应增益和干扰估计误差
e = r - y;
v_dot = -k3*v + f_hat;
f_dot = -k4*f + f_hat;
w_dot = -k4*y*sign(y - C*x);
v = v + v_dot*0.01; % 数值积分
f = f + f_dot*0.01; % 数值积分
w = w + w_dot*0.01; % 数值积分

% 计算状态向量的导数
dx = zeros(2,1);
dx(1) = x(2);
dx(2) = A(2,:)*x + u*B;
end
