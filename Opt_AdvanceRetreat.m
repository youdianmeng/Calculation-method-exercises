function  [xmin, xmax] = Opt_AdvanceRetreat(fun, alpha, h)
    %% 进退法求搜索区间
    % fun: 函数句柄，使用Opt_AdvanceRetreat函数前需要定义函数fun
    % A: 目标多项式函数系数向量,高阶系数在前
    % alpha: 函数初值
    % h: 步长,迭代为h,2h,4h,……
  
    if nargin < 3
        error('检查输入参数个数！')
    end

    % 初始化和初次比较
    xmin = 0;
    xmax = 0;
    x0 = alpha;
    x1 = alpha + h;
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
            % 更新xlim
            xmin = x0;
            % 更新以满足迭代
            x0 = x1;
            fx0 = fx1;
            % 更新坐标及函数
            h0 = 2 ^ i * h;
            x1 = x0 + h0;
            fprintf('xmin=%f',xmin);
            disp(x1);
            fx1 = fun(x1);
               
            i = i + 1;
        end
        xmax = x1;
    % 二：低高
    else
        j = 1;
        while fx1 > fx0
            xmax = x1;
            x1 = x0;
            fx1 = fx0;
            h0 = 2 ^ j * (-h);
            x0 = x1 + h0;
            fx0 = fun(x0);

            j = j + 1;
        end
        xmin = x0;
    end
    
end