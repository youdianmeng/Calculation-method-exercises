function K = Gaussian_kernel_function(x, z, sigma)
    %% 高斯核函数
    % x, z: 向量或矩阵,一行为一个坐标;如果x、z一个为向量一个为矩阵，那么返回向量与矩阵每一行的进行的核函数
    % sigma  核函数参数，高斯核宽度

    K =  exp((-(vecnorm(x-z, 2, 2)).^2 / (2*sigma^2)));

end