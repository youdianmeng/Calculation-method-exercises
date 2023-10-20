function result = univariate_search(fun, X0, epsilon, AR_h, Goldmin_epsilon)
    %% n维变量轮转法
    % fun: 目标函数,引用时记得加 @
    % X0： 迭代起始点
    % epsilon: 迭代精度
    % AR_h： 进退法步长
    % Goldmin_epsilon: 黄金分割法迭代误差
    % 测试代码：univariate_search(@func, [0, 0], 0.035, 0.0125, 0.0125)

    %-----------------------测试图像-------------------------%
    % 测试图像
    % 定义x和y的范围
    % 定义x和y的范围
    x = -10:0.5:20;
    y = -10:0.5:20;
    % 生成网格点矩阵
    [X, Y] = meshgrid(x, y);
    % 计算Z值
    Z = 60 - 10*X - 4*Y + X.^2 + Y.^2 - X.*Y;
    % 绘制三维曲面图
    mesh(X, Y, Z);
    %--------------------------------------------------------%

    if nargin < 5
      error('检查参数个数')
    end

    X0 = X0(:);
    n = length(X0);
    
    x0 = X0;
    iter = 0;
    while 1
        iter = iter + 1;
        fprintf('iter=%d \n', iter);

        x1 = x0;
        for index = 1 : n
            fprintf('方向=%d,', index);
            direction = zeros(n, 1);
            direction(index) = 1;

            % index方向上的，x0为初始点，使用进退法得到的搜索区间
            D = Opt_AdvanceRetreat(fun, x0, 2, AR_h, direction);
            fprintf('搜索区间=\n');disp(D);
            % 得到变量轮换该方向上的最佳目标点坐标 alpha
            alpha = Goldmin(fun, D(:, 1), D(:, 2), Goldmin_epsilon);
            fprintf('该方向目标点坐标=\n');disp(alpha);
            
            x0 = alpha;
%             % 更新S
%             S = zeros(n, 1);
%             S(index) = 1;
%             x0 = x0 + alpha * S;
            fprintf('x1   =');disp(x1');
            fprintf('x0   =');disp(x0');
            fprintf('误差=%f\n', sum((x0 - x1).^2));

            %-----------------------可视化-------------------------%
            text(x0(1),x0(2), '0', 'color', 'r', 'FontSize', 10);
            pause;
            %------------------------------------------------------%
        end
        
        if sum((x0 - x1).^2) < epsilon
            break;
        end

    end

    result = [fun(x0) x0(:)'];
end

