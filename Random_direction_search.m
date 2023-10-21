function result = Random_direction_search(fun, restraint, Var_area, step, epsilon, Nmax)
    %% 多约束随机方向搜索优化
    % fun: 目标函数
    % restraint: 约束条件，以函数形式调用
    % Var_area: 初始变量估计上下限，size = n × 2
    % step: 步长
    % epsilon：步长精度
    % Nmax ：某迭代点的最大迭代次数

    % 代码示例：Var_area = [-3, 1; -1, 3];step = 1.1; epsilon = 0.01;Nmax = 50;Random_direction_search(@RDS_fun, @RDS_restraint, Var_area, step, epsilon, Nmax)
    format long;
    
    if nargin < 6
        error('检查参数个数')
    end 
    
    n = size(Var_area, 1);
    x0 = zeros(n, 1);
    %------------------------- 生成初始点 ---------------------------%
    while 1
        % 随机生成初始点
        for i = 1 : n
            xi = Var_area(i,:);
            x0(i) = xi(1) + rand(1) * (xi(2) - xi(1));
        end
        % 可行性判断
        if restraint(x0)
            fprintf('初始点x0 =');disp(x0');
            break
        end
    end
    %---------------------------------------------------------------%
    
    fx0 = fun(x0);
    k = 0; iter = 0;
    while 1

        % 随机搜索方向的产生,即产生单位圆边内(-1, 1)
        S_square = -1 + rand(n,1) * (1 - (-1)); %以原点为中心，边长为2的矩阵中的随机点
        S = S_square / norm(S_square);
        %fprintf('S =');disp(S');
        while 1
            x = x0 + step * S;
            iter = iter + 1;
            fprintf('iter=%d;\n', iter);
            % fprintf('step=%f;\n ', step);

            % 可行性判断
            if restraint(x) % 满足可行性
                % 下降性判断
                fx = fun(x);
                fprintf('fx=%f; ', fx);
                fprintf('fx0=%f; \n', fx0);
                if fx < fx0 % 满足下降性
                    x0 = x;
                    fx0 = fx;
                else
                    k = k + 1; % 计数器
                    break;
                end
            else
                k = k + 1;
                break;
            end
        end

        if step <= epsilon
            result = [fx0 x0'];
            break
        end

        if k >= Nmax
            step = step /2 ;
            k = 0;
        end
    end
end




