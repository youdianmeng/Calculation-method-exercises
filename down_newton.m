## 定义lambda初值向量
lambda = zeros(1,10);
for i = 1:10
  lambda(i) = (1/2)^(i-1);
end
# display(lambda);

## 初值
xk1 = -0.9900000;
xk = 0.0000000;

## 计时防止死循环
tic;
k = 0;
while abs(xk - xk1) >= 0.00001
  k++;
  fprintf("k=%d\n", k);
  xk = xk1;
  for i = 1:10
    xk1 = xk - lambda(i) * (xk^3/3-xk)/(xk^2-1);

    fprintf("lambda=%d\n",lambda(i));
    fprintf("xk1=%f\n",xk1);

    if abs(xk1^3/3-xk1) < abs(xk^3/3-xk)
      break
    end
  end
# display(xk);
  if toc > 10
    break
  end
end
display("最终结果：")

## 保证输出结果位数
format long;
display(xk);


