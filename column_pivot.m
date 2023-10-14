%% 增广矩阵A
%% 方程组解x

A = [-3 2 6 4; 10 -7 0 7; 5 -1 5 6];
display(A);

[N M] = size(A);
x = zeros(N,1);

disp('消元中……')
for i = 1:N-1
  %% 更换主元行数
  B = A(i:end,i:end);
  [max_num max_index] = max(abs(B(:,1)));
%  disp(max_index);

  %% 最大主元索引为index + i - 1
  temp = A(i,:);
  A(i,:) = A(max_index + i - 1,:);
  A(max_index + i - 1,:) = temp;
  display(A);

  %% 判断是否为0
  if A(i,i) ~= 0
    for j = i+1:N
      A(j,:) = A(j,:) - A(i,:)*A(j,i)/A(i,i);
    end
  else
    display("待解决……");
    break
  end
end
disp("消元结果：");
disp(A);

%% 计算x
x(N) = A(end,end)/A(end,end-1);
for i = N-1:-1:1
  x(i) = A(i,end);
  for j = i+1:N
    x(i) = x(i) - A(i,j)*x(j);
  end
  x(i) = x(i)/A(i,i);
end
disp("求解得x=");
disp(x);





