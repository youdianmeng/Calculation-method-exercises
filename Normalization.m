function result = Normalization(X)
%% 归一化
% X: 一行为一个样本点
    
    result = (X - mean(X)) ./ std(X);

end