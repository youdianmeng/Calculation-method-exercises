%% 非线性超圆盘二分类器，利用一对一策略推广及多分类问题
% Z—score归一化
% 随机划分数据集

% 初始化
clear; close all; clc;format long;
%%

Num = 3; % 类别数

% 载入数据
data = load('F:\python_learning\Calculation-method-exercises\seeds_dataset.txt');

% 划分训练集和测试集
labels = data(:, end); 
classes = unique(labels); % 获取所有的类别
trainData = []; testData = []; % 初始化训练集和测试集

for i = 1:length(classes)
    classData = data(labels == classes(i), :); 
    numTrain = round(0.10 * size(classData, 1)); % 训练集的数量
    indices = randperm(size(classData, 1)); % 随机排列索引
    trainData = [trainData; classData(indices(1:numTrain), :)]; 
    testData = [testData; classData(indices(numTrain+1:end), :)]; 
    %trainData = [trainData; classData(1:numTrain, :)]; 
    %testData = [testData; classData(numTrain+1:end, :)]; 
end

% 获取数据信息
X_train = trainData(:, 1:end-1);
Y_train = trainData(:, end);
[M_train, N_train] = size(X_train);

X_test = testData(:, 1:end-1);
Y_test = testData(:, end);
[M_test, N_test] = size(X_test);

% 训练集数据归一化
meanX_train = mean(X_train);
stdX_train = std(X_train);
X_train = Normalization(X_train);
% result = (X_train - mean(X_train)) ./ std(X_train);

% 测试集数据归一化
X_test = (X_test - meanX_train) ./ stdX_train;


%%%------------------------------------------------ 训练 -------------------------------------------------%%%
%------------------------------------------- 求解s和r --------------------------------------------%

beta_ij = zeros(Num, M_train);
s_and_r = zeros(Num, N_train + 1); % 每行最后一个元素为r
for i = 1 : Num
    X = X_train(Y_train==i, :);
    Y = Y_train(Y_train==i, :);

    [M, N] = size(X);

%     % 分类保存数据
%     filename = ['Class', num2str(i), 'Data.mat']; % 创建文件名
%     save(filename, 'X'); % 保存数据

    % 设定拉格朗日系数beta初值,一个样本一个beta
    beta_initial = ones(M, 1) / M;

    % 设定线性等式约束
    Aeq = ones(1, M);
    beq = 1;

    % 设定变量下界
    lb = zeros(1, M) + 1e-4;

    % 求解对偶问题拉格朗日系数, 返回最佳beta
    options = optimoptions('fmincon','MaxFunctionEvaluations',1e6);  % 设置最大函数评估次数为 1000000
    % @(beta)Objective_Function1(beta, X),@(beta) 表示输入要优化的值为beta，X不动
    beta = fmincon(@(beta)Objective_Function1(beta, X), beta_initial, [], [], Aeq, beq, lb, [], [], options);  % 使用新的 options 变量
    beta_ij(i, 1:M) = beta'; % 保存 i 类别的beta

    % 求解s and r
    s = sum(beta.*X);
    [r, max_index] = max(vecnorm(X-s, 2, 2)); % https://ww2.mathworks.cn/help/matlab/ref/vecnorm.html
    s_and_r(i, :) = [s, r];

end
% disp(s_and_r);

%% 非线性超圆盘，采用高斯核函数

%------------------------------------------- 求解分类超平面参数 --------------------------------------------%
% one vs. one 策略
Num_i_vs_j = Num * (Num - 1) / 2; % 要训练 Num_i_vs_j 个分类器
w_and_b = zeros(Num_i_vs_j, 2 + N_train + 1); % [classi, classj, [---w---], b]
fx_ij = zeros(M_test, Num_i_vs_j); % 有N_test个测试样本，每个测试样本进行 Num_i_vs_j次分类
iter = 0;
for i = 1 : Num - 1
    for j = i + 1 : Num
        iter = iter + 1;

        X_plus = X_train(Y_train==i, :);
        X_minus = X_train(Y_train==j, :); % 之前已完成归一化

        M_plus = size(X_plus, 1);
        M_minus = size(X_minus, 1);

        % 设定超平面参数初值，都为列
        alpha_plus = ones(M_plus, 1) / M_plus;
        alpha_minus = ones(M_minus, 1) / M_minus;
        alpha_initial = [alpha_plus; alpha_minus];

        % 设定线性等式约束
        Aeq = zeros(2, M_plus+M_minus);
        Aeq(1, 1:M_plus) = 1;
        Aeq(2, M_plus+1:end) = 1;
        beq = [1; 1];
        
        % 设定非线性约束,只用r，不用s
        r_plus_minus = s_and_r([i, j], end);
        beta_plus = beta_ij(i, 1:M_plus)';
        beta_minus = beta_ij(j, 1:M_minus)';
        beta_plus_minus = [beta_plus; beta_minus];
        nonlcon = @(alpha)Hyperplane_nonlinear_restraint2(alpha, beta_plus_minus, X_plus, X_minus, r_plus_minus);

        % 求解分类超平面参数权重向量, 返回最佳alpha
        options = optimoptions('fmincon','MaxFunctionEvaluations',1e6);  % 设置最大函数评估次数为 1000000
        % @(beta)Objective_Function1(beta, X),@(beta) 表示输入要优化的值为beta，X不动
        alpha = fmincon(@(alpha)Objective_Function3(alpha, X_plus, X_minus), alpha_initial, [], [], Aeq, beq, [], [], nonlcon, options);  % 使用新的 options 变量
        

        % 求解分类超平面参数
        w = sum(alpha(1:M_plus).*X_plus, 1) - sum(alpha(M_plus+1, end).*X_minus, 1); % 行向量
        b = -1/2 * w * (sum(alpha(1:M_plus).*X_plus, 1) + sum(alpha(M_plus+1, end).*X_minus, 1))';
        w_and_b(iter, :) = [i, j, w, b];
        
        %%%-------------------------------- 测试 ---------------------------------%%%

        % 分类，返回行向量，长度为测试集样本个数M_test，1表示为 i 类型，-1为 j 类型，0表示在分类器边界上
        fx = sign(sum((X_plus*X_test').*alpha(1:M_plus), 1) - sum((X_minus*X_test').*alpha(M_plus+1:end), 1) - b); % 行向量，长度为测试集样本个数
        
        fx(fx==1) = i;
        fx(fx==-1) = j;
        fx(fx==0) = 0;
        
        fx_ij(:, iter) = fx;

    end
end


%%
Y_test_pred = mode(fx_ij, 2);
%disp([fx_ij, Y_test_pred, Y_test]);
% 返回准确度
accuracy = sum(Y_test_pred==Y_test) / M_test;
%disp(w_and_b);














