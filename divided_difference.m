function table = divided_difference(x, y)
  % x, y为样本点坐标

  %检查输入
  if length(x) ~= length(y)
    error("The lengths of x and y must be equal");
  end

  if isempty(x)
    error("The inputs cannot be empty");
  end

  % x, y 转化为列向量形式
  x = x(:);
  y = y(:);

  %构建列表
  table = zeros(length(x), length(x) - 1);
  table = [x, y, table];

  %计算差商
  %输入n阶向量，有n-1阶差商，即计算n-1次
  for k = 1 : length(x) -1
    %每一阶差商的个数与当前阶数和为n
    for i = 1 : length(x) - k
      %计算
      table(i, k + 2) = (table(i + 1, k + 1) - table(i, k + 1)) / (table(i + 1, 1) - table(i, 1));
    end
  end

end
