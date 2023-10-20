function xmin = Goldmin(fun, A, B, epsilon)
    %% 黄金分割法求最小值
    % fun: 函数句柄
    % A: 初始搜索区间初始点
    % B: 初始搜索区间终止点
    % epsilon: 迭代精度

    %{运行示例代码：
    % clear all;fun = @(x) x^2-2*x +1;A=0,B=2;epsilon=0.0001;Goldmin(fun,A,B,epsilon) }
    
    a = A;
    b = B;
    % a1一直在左侧
    a1 = b - 0.618 * (b - a);
    a2 = a + 0.618 * (b - a);
    fx1 = fun(a1);
    fx2 = fun(a2);

    while b - a > epsilon

        if fx1 <= fx2
            b = a2;
            a2 = a1;
            fx2 = fx1;
            a1 = b - 0.618 * (b - a);
            fx1 = fun(a1);

        else
            a = a1;
            a1 = a2;
            fx1 = fx2;
            a2 = a + 0.618 * (b - a);
            fx2 = fun(a2);

        end

    end
    xmin = (a + b) / 2;
end
