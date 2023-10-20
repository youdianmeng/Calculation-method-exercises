function retval = func (A)
    %% 变量轮转法、进退法、黄金分割法的测试函数func
    % A: [x1, x2,……,xn]
     retval =  60 - 10*A(1) - 4*A(2) + A(1)^2 + A(2)^2 - A(1)*A(2);
     %retval = A*A - 2*A + 1;
     %retval = A(1)*A(1) - 2*A(1) + 1 + A(2);
      
end
