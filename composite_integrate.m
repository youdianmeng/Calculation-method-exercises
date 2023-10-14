x = [0 1/8 2/8 3/8 4/8 5/8 6/8 7/8 8/8];
fx = [4.0 3.9384615 3.7647059 3.5068493 3.2 2.8764045 2.56 2.654867 2.0];

h = (x(end)-x(1))/(length(x)-1);
I = h*(sum(fx)-fx(1)/2-fx(end)/2);

fx1 = fx(2:end-1);
fx2 = fx(3:end-2);
S = h/3*(fx(1)+fx(end)+4*sum(fx1(1:2:end))+2*sum(fx2(1:2:end)));
display(I);
display(S);
