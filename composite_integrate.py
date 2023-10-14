import argparse
import numpy as np
## 尝试用命令行接口运行程序
def composite_integrate_args():
    parser = argparse.ArgumentParser(description = 'Composite intergrate')

    parser.add_argument('-o', '--option', type=int, default=0, choices=[0, 1],
                        help='choose Trapezoidal or Simpson to integrate')
    parser.add_argument('-x_fx',type=list, default=[[0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8],
                        [4.0, 3.9384615, 3.7647059, 3.5068493, 3.2, 2.8764045, 2.56, 2.654867, 2.0]], help='x and fx of the object')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = composite_integrate_args()
    x, fx = args.x_fx
    h = (x[-1] - x[0]) / (len(x) - 1)
    if args.option == 0:
        I = h * (sum(fx) - fx[0]/2 - fx[-1]/2)
        print(f'I = {I}')
    else:
        fx1 = np.copy(fx[1:-1])
        fx2 = np.copy(fx[2:-2])
        S = h/3 * (fx[0] + fx[-1] + 4*np.sum(fx1[::2]) + 2*np.sum(fx2[::2]))
        print(f'S = {S}')


    