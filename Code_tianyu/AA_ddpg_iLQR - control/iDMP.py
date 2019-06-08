import numpy as np
from iLSTM import BATCH_MANAGE
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from scipy.optimize import fsolve
import pickle


class DMP(object):
    start_pos = np.array([0.63276377, - 0.20655886, -1.55968])

    def __init__(self, dim_target):

        self.dim_target = dim_target

        self.num_basisFunction = 30
        self.theta = np.zeros([self.dim_target, self.num_basisFunction])

        self.alphat = 3

        self.c = np.zeros(self.num_basisFunction)  # 高斯核 部件：cj
        for j in range(self.num_basisFunction):
            self.c[j] = np.exp(-self.alphat * j / (self.num_basisFunction - 1))

        self.h_gaus = np.zeros(self.num_basisFunction)  # 高斯核 部件：hj
        for j in range(self.num_basisFunction - 1):
            self.h_gaus[j] = 1 / (self.c[j + 1] - self.c[j]) ** 2

        self.h_gaus[-1] = self.h_gaus[-2]

    def path2dmp(self, list_path, time):

        target_path = np.array(list_path)

        Gt = np.zeros([len(time), self.num_basisFunction])

        for tt in range(len(time)):
            ei = 0
            st = self.st(time[tt])
            for k in range(self.num_basisFunction):
                ei += np.exp(-0.5 * self.h_gaus[k] * (st - self.c[k]) ** 2)
            for j in range(self.num_basisFunction):
                Gt[tt, j] = np.exp(-0.5 * self.h_gaus[j] * (st - self.c[j]) ** 2) * st / ei

        for dim in range(self.dim_target):
            filtered_path = self.myFilter(target_path[dim, :], 100)
            self.theta[dim, :] = self.mySolve(Gt, filtered_path)

        return self.theta

    def mySolve(self, A, y):
        x = np.linalg.solve(np.dot(np.array(A).T.copy(), np.array(A)), np.dot(np.array(A).T.copy(), np.array(y)))
        return x

    def myFilter(self, x, frequency):
        y = signal.medfilt(x, 3)

        return y

    def st(self, t):
        return np.exp(-self.alphat * t)

    def fundmp(self, t):

        st = self.st(t)  # 系统时钟

        result = np.zeros(self.dim_target)
        ei = 0  # 分母
        for k in range(self.num_basisFunction):
            ei += np.exp(-0.5 * self.h_gaus[k] * (st - self.c[k]) ** 2)
        for dim in range(self.dim_target):
            for j in range(self.num_basisFunction):
                result[dim] += np.exp(-0.5 * self.h_gaus[j] * (st - self.c[j]) ** 2) * st / ei * self.theta[dim, j]

        return result


def mySolve(func, value):
    def fun_temp(t):
        return abs(func(t) - value)

    result = []
    for i in np.linspace(0, 1, 10):
        try:
            y = fsolve(fun_temp, i)[0]
        except:
            continue
        if fun_temp(y) < 0.0001:
            result.append(y)

    return list(set(result))[0] if len(result) > 0 else None


if __name__ == '__main__':
    # 主程序
    batchm = BATCH_MANAGE(s_dim=4, a_dim=3)
    batchm.restore()

    num_batch = len(batchm.list_action)

    func_batchs = []

    for i in range(num_batch):
        list_time = np.transpose(batchm.list_states[i])[0]
        list_path = np.transpose(batchm.list_states[i])
        list_time = np.linspace(0, 1, len(list_time))

        func_dims = []
        for j in range(4):
            f = interpolate.interp1d(list_time, list_path[j, :], kind='linear')
            func_dims.append(f)

        func_batchs.append(func_dims)

    path_save = 'funcs.pkl'

    fw = open(path_save, 'wb')
    pickle.dump(func_batchs, fw)
    fw.close()

    fr = open(path_save, 'rb')
    rfuncs = pickle.load(fr)
    fr.close()

    time = np.linspace(0, 1, 200)
    ynew = rfuncs[0][0](time)

    query_process = 0.179
    dim_index = 1
    list_result = []
    for i in range(num_batch):
        this_time = mySolve(rfuncs[i][0], query_process)
        # print(this_time)
        if this_time is not None:
            result = float(rfuncs[i][dim_index](this_time))
            print(result)
            list_result.append(result)

    print(list_result)

    plt.plot(time, ynew, '--r')

    plt.show()
