function retval = Toget_gradient(fun, x0)
  %% 计算function 在x0处的梯度
  % fun: 函数句柄
  % x0: 需要求解梯度的坐标
  delta = 1e-6;
  x0 = x0(:);
  gradi = zeros(size(x0));
  for index = 1 : length(x0)
    x0_moved = x0;
    x0_moved(index) = x0_moved(index) + delta;
    gradi(index) = (fun(x0_moved) - fun(x0)) / delta;
  end
  retval = gradi;
end
