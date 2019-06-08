import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
import scipy.stats as stats
import pickle
from iLSTM import BATCH_MANAGE


class TUBE(object):

    def __init__(self):

        self.path_demo_function_save = 'demo_function_save.pkl'
        self.path_meanstd_function_save = 'meanstd_function_save.pkl'

    def __mySolve(self, func, value):
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

    def myDistribution(self, x, mean, std):
        func = stats.norm(mean, std)
        y = func.pdf(x)

        return y

    def __query_action(self, query_process):
        # 1-2-3 对应 x-y-w轴
        list_across_batches = []

        for i in range(self.num_batch):
            this_time = self.__mySolve(self.demo_funcs[i][0], query_process)
            # print(this_time)
            if this_time is not None:
                list_across_dims = []
                for j in [1, 2, 3]:
                    list_across_dims.append(float(self.demo_funcs[i][j](this_time)))
                list_across_batches.append(list_across_dims)

        if len(list_across_batches) == self.num_batch:
            mean_action = np.mean(list_across_batches, axis=0)  # axis=0表示列，每列的均值
            std_action = np.std(list_across_batches, axis=0)  # 求每列的方差
            for jj in range(len(std_action)):
                # if std_action[jj] < 0.1:
                #     std_action[jj] = 0.1
                std_action[jj] += 0.2
        else:
            mean_action = np.array([0, 0, 0])
            std_action = np.array([1, 1, 1])

        return mean_action, std_action

    def __interpolation_meanstd_fuction(self):

        process_set = np.linspace(-0.1, 1.1, 200)
        list_mean = []
        list_std = []
        for process in process_set:
            mean, std = self.__query_action(process)
            list_mean.append(mean)
            list_std.append(std)

        array_mean = np.transpose(list_mean)
        array_std = np.transpose(list_std)

        func_dims = []
        for dim in range(3):
            f_mean = interpolate.interp1d(process_set, array_mean[dim, :], kind='linear')
            f_std = interpolate.interp1d(process_set, array_std[dim, :], kind='linear')
            func_dims.append([f_mean, f_std])

        fw = open(self.path_meanstd_function_save, 'wb')
        pickle.dump(func_dims, fw)
        fw.close()
        self.meanstd_funcs = func_dims

    def __interpolation_demo_fuction(self):
        # 对示教采集的 process, dx, dy, dw 集进行差值。形成num_batch x 4个拟合函数
        self.num_batch = len(self.list_states)

        func_batchs = []

        for i in range(self.num_batch):
            list_time = np.transpose(self.list_states[i])[0]
            list_path = np.transpose(self.list_states[i])
            list_time = np.linspace(0, 1, len(list_time))

            func_dims = []
            for j in range(4):
                # 示教cycletime为训练cycletime的5倍以上
                f = interpolate.interp1d(list_time, list_path[j, :] * 2, kind='linear')
                func_dims.append(f)

            func_batchs.append(func_dims)

        fw = open(self.path_demo_function_save, 'wb')
        pickle.dump(func_batchs, fw)
        fw.close()

        self.demo_funcs = func_batchs

    def __restore_demo_fuction(self):

        fr = open(self.path_demo_function_save, 'rb')
        self.demo_funcs = pickle.load(fr)
        fr.close()
        self.num_batch = len(self.demo_funcs)

    def __restore_meanstd_fuction(self):

        fr = open(self.path_meanstd_function_save, 'rb')
        funcs = pickle.load(fr)  # shape=[3,2]
        fr.close()

        self.meanstd_funcs = funcs

        return funcs

    def build_tube(self, list_states):

        self.list_states = list_states

        self.__interpolation_demo_fuction()
        self.__interpolation_meanstd_fuction()

        print('【TUBE】新tube已建立。基于 batchsize = ', len(self.list_states))

    def restore_tube(self):
        return self.__restore_meanstd_fuction()

    def get_tube_eval(self, obsv):
        # obsv=[process,dx,dy,dw]
        assert len(obsv) == 4, obsv
        process = obsv[0]
        list_eval = []
        for dim in [0, 1, 2]:
            mean = self.meanstd_funcs[dim][0](process)
            std = self.meanstd_funcs[dim][1](process)
            eval_dim = abs(obsv[dim + 1] - mean) / std
            list_eval.append(eval_dim)

        return list_eval


class APICKLE(object):
    def __init__(self):

        self.path_save = 'batch_list_demo.pkl'
        self.list_states = list()

    def add_seq(self, s_seq):
        self.list_states.append(s_seq)

    def save(self, path=None):
        # self.update_batch()
        if path is not None:
            self.path_save = path
        fw = open(self.path_save, 'wb')
        pickle.dump(self.list_states, fw)
        fw.close()

    def restore(self, path=None):
        if path is not None:
            self.path_save = path
        print(self.path_save)
        fr = open(self.path_save, 'rb')
        self.list_states = pickle.load(fr)
        fr.close()


if __name__ == '__main__':
    # 主程序

    apickle = APICKLE()
    apickle.restore()

    tube = TUBE()
    tube.build_tube(apickle.list_states)

    # tube.restore_tube()
    #
    # obsv = [0.8, 0.1, 2, 0]
    # eval = tube.get_tube_eval(obsv)
    #
    # print(eval)
