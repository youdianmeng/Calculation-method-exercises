function xmin = goldmin(fun, D, epsilon, alpha, index)
    %% 黄金分割法求最小值
    % fun: 函数句柄
    % D: 初始搜索区间
    % 迭代精度
    % alpha: 函数初值 向量,可针对多元函数
    % index: 变量轮转法时使用， default = 1

    %{运行示例代码：
    % clear all;fun = @(x) x^2-2*x +1;D=[0,1];epsilon=0.0001;goldmin(fun,D,epsilon) }
    
    n = length(alpha);
    alpha(index) = 0;
    S = zeros(n, 1);
    S(index) = 1;
    a = D(1);
    b = D(2);
    % a1一直在左侧
    a1 = b - 0.618 * (b - a);
    a2 = a + 0.618 * (b - a);
    fx1 = fun(S*a1 + alpha);
    fx2 = fun(S*a2 + alpha);

    while b - a > epsilon

        if fx1 <= fx2
            b = a2;
            a2 = a1;
            fx2 = fx1;
            a1 = b - 0.618 * (b - a);
            fx1 = fun(S*a1 + alpha);

        else
            a = a1;
            a1 = a2;
            fx1 = fx2;
            a2 = a + 0.618 * (b - a);
            fx2 = fun(S*a2 + alpha);

        end

    end
    xmin = (a + b) / 2;
end
