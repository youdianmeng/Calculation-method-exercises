function min = goldmin(fun, D, epsilon)
    %% 黄金分割法求最小值
    % fun: 函数句柄
    % D: 初始搜索区间
    % 迭代精度
    %{运行示例代码：
    % clear all;fun = @(x) x^2-2*x +1;D=[0,1];epsilon=0.0001;goldmin(fun,D,epsilon) }
    a = D(1);
    b = D(2);
    % a1一直在左侧
    a1 = b - 0.618 * (b - a);
    a2 = a + 0.618 * (b - a);
    fx1 = fun(a1);
    fx2 = fun(a2);
%     fprintf('a1=%f ', a1);
%     fprintf('a2=%f ', a2);
%     fprintf('fx1=%f ', fx1);
%     fprintf('fx2=%f\n', fx2);
    while b - a > epsilon

        if fx1 <= fx2
            b = a2;
            a2 = a1;
            fx2 = fx1;
            a1 = b - 0.618 * (b - a); 
            fx1 = fun(a1);
%             fprintf('fx1=%f ', fx1);
%             fprintf('fx2=%f ', fx2);
%             fprintf('a=%f ', a);
%             fprintf('a1=%f ', a1);
%             fprintf('a2=%f ', a2);
%             fprintf('b=%f\n', b);
        else 
            a = a1;
            a1 = a2;
            fx1 = fx2;
            a2 = a + 0.618 * (b - a);
            fx2 = fun(a2);
%             fprintf('fx1=%f ', fx1);
%             fprintf('fx2=%f ', fx2);
%             fprintf('a=%f ', a);
%             fprintf('a1=%f ', a1);
%             fprintf('a2=%f ', a2);
%             fprintf('b=%f\n', b);
        end

    end
    min = (a + b) / 2;
end