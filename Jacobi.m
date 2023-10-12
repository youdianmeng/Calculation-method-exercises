function x = Jacobi(A)
  ## 运用向量化思路实现雅可比迭代，已设定参数默认值，可直接键入 Jacobi 运行
  ## A：增广矩阵

  ## 设定参数默认值
  if ~exist('A')
    A = [-4 1 1 1 1; 1 -4 1 1 1; 1 1 -4 1 1; 1 1 1 -4 1];
  end

  [M N] = size(A);

  ## 对于每一个x_i，同时计算，故提取a_ii和a_ij
  aii = diag(A(1:end, 1:end-1)); # 提取对角线元素
  aij = A(1:end, 1:end-1) - eye(M) .* aii; # 满足i≠j，即去除对角线元素

  ## 初始化
  x0 = zeros(M, 1);
  xk = ones(M, 1);

  ## 开始迭代
  while max(abs(x0 - xk)) >= 0.00001

    xk = x0;
  # disp(xk);
    x0 = (1./aii) .* (A(:, end) - aij * xk) ;
  # disp(x0);

  end
  x = x0;

end

