function LSapprox(x, y, n)
  % x为横坐标向量， y为纵坐标向量
  % n为拟合阶数

  %检查输入
  if length(x) ~= length(y)
    error("The lengths of x and y must be equal");
  end

  if isempty(x)
    error("The inputs cannot be empty");
  end

  %预处理
  x = x(:)';
  y = y(:)';

  % 构造矩阵x_ij和向量yk
  x_ij = zeros(n+1, n+1);
  yk = zeros(n+1, 1);

  for i = 1:n+1
      for j = 1:n+1
          x_ij(i, j) = sum(x.^(i+j-2));
      end
      yk(i) = sum(y.*x.^(i-1));
  end

  % 求解线性方程组
  coefficients = x_ij \ yk;

  % 打印拟合多项式的系数
  fprintf('拟合多项式的系数为：');
  for i = 1:length(coefficients)
      fprintf('%f ', coefficients(i));
  end
  fprintf('\n');

  % 绘制原始数据和拟合曲线
  xx = linspace(min(x), max(x), 100);
  yy = polyval(flip(coefficients), xx);

  figure;
  plot(x, y, 'ko', xx, yy, 'r-')
  legend('原始数据', '拟合曲线');
  xlabel('x');
  ylabel('y');
end
