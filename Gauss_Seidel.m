function x = Gauss_Seidel(A)
  %% 实现高斯-赛德尔迭代，已设定参数默认值，可直接键入 Guass_Seidel 运行
  %% A：增广矩阵

  %% 设定参数默认值
  if ~exist('A')
    A = [-4 1 1 1 1; 1 -4 1 1 1; 1 1 -4 1 1; 1 1 1 -4 1];
  end

  [M N] = size(A);

    %% 初始化
  x0 = zeros(M, 1);
  xk = ones(M, 1);

  while max(abs(x0 - xk)) >= 0.00001

    xk = x0;

    for i = 1 : M
      if i == 1
        x0(i) = (1/A(i, i)) * (A(i, end) - A(i,i+1:end-1) * xk(i+1:end));
      elseif i == M
        x0(i) = (1/A(i, i)) * (A(i, end) - A(i, 1:i-1) * x0(1:i-1));
      else
        x0(i) = (1/A(i, i)) * (A(i, end) - A(i, 1:i-1) * x0(1:i-1) - A(i,i+1:end-1) * xk(i+1:end));
      end
    end

  end

  x = x0;

end

