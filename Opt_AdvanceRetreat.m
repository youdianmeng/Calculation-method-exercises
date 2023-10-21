function  D = Opt_AdvanceRetreat(fun, initial, step, beta, direction)
    %% 进退法求搜索区间
    % fun: 函数句柄，使用Opt_AdvanceRetreat函数前需要定义函数fun,fun变量应为[x0,x1,……]
    % initial: 函数初值 向量,可针对多元函数
    % step: 步长,迭代为h,2h,4h,……
    % beta: 进退法迭代factor,h,3h,9h……
    % direction: 变量轮转法时使用， default = 1,为迭代方向

    % {测试代码：clear all;fun = @(x) x^2-2*x +1;initial=0;step=0.1;beta=2;direction=[1];Opt_AdvanceRetreat(fun,initial,step,beta,direction)}

    if nargin < 5
        error('检查输入参数个数！')
    end

    initial = initial(:);
    direction = direction(:);
    % 初始化和初次比较
    xl = zeros(size(initial));
    xr = zeros(size(initial));

    x0 = initial;
    x1 = initial + direction*step;

    fx0 = fun(x0);
    fx1 = fun(x1);
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
            step0 = beta ^ i * step;
            x1 = x0 + direction*step0;
            fx1 = fun(x1);

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
            step0 = beta ^ j * direction*(-step);
            x0 = x1 + step0;
            fx0 = fun(x0);

            j = j + 1;
        end
        xl = x0;
    end
     D = [xl xr];
end
