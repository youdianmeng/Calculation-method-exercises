function  D = Opt_AdvanceRetreat(fun, alpha, h, index)
    %% 进退法求搜索区间
    % fun: 函数句柄，使用Opt_AdvanceRetreat函数前需要定义函数fun,fun变量应为[x0,x1,……]
    % alpha: 函数初值 向量,可针对多元函数
    % h: 步长,迭代为h,2h,4h,……
    % index: 变量轮转法时使用， default = 1,为迭代方向

    if nargin < 4
        error('检查输入参数个数！')
    end

    n = length(alpha);
    S = zeros(n, 1);
    S(index) = 1;
    % 初始化和初次比较
    xl = 0;
    xr = 0;
    x0 = alpha(index);
    x1 = alpha(index) + h;
    alpha(index) = 0;
    fx0 = fun(S*x0 + alpha);
    fx1 = fun(S*x1 + alpha);
    % 如果fx0和fx1相等，引入一个小的随机扰动，防止出现bug
    if fx0 == fx1
        fx1 = fx1 + rand()*1e-6;
    end
    % 一：高低
    if fx1 < fx0
        i = 1;
        while fx1 < fx0
            % 更新xr
            xl = x0;
            % 更新以满足迭代
            x0 = x1;
            fx0 = fx1;
            % 更新坐标及函数
            h0 = 2 ^ i * h;
            x1 = x0 + h0;
            fx1 = fun(S*x1 + alpha);

            i = i + 1;
        end
        xr = x1;
    % 二：低高
    else
        j = 1;
        while fx1 > fx0
            xr = x1;
            x1 = x0;
            fx1 = fx0;
            h0 = 2 ^ j * (-h);
            x0 = x1 + h0;
            fx0 = fun(S*x0 + alpha);

            j = j + 1;
        end
        xl = x0;
    end
     D = [xl xr];
end
