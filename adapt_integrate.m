## 梯形法求积分
epsilon = 10^(-7);
a = 0.000000000001;
b = 1;
i = 2;
x0 = linspace(a, b, i);

T_n = (b - a)/2 * (sum(sin(x0) ./ x0));
T_2n = 0;

tic;
display("计算中……")
while abs(T_n - T_2n) >= epsilon
  T_2n = T_n;

  x = setdiff(linspace(a, b, 2*i-1), linspace(a, b, i));
  T_n = T_2n/2 + (b-a)/(2*i-1) * sum(sin(x)./x);
  i = 2 * i - 1;

##  format long;
##  display(T_n);
##  display(T_2n);
  if toc > 10
    break
  end
end
display("计时10s计算的结果为：")
format long;
display(T_2n);
display(T_n);
