import numpy as np
import multiprocessing as mp
import time
import socket
import struct
import kinematicsPack as kp


class multicoreHMI(object):

    def __init__(self, cycle_time=0.01):
        # 定义共享内存
        self.hmi_ctw = mp.Array('i', [0])
        self.plock = mp.Lock()  # 进程锁

        self.cycle_time = cycle_time

        self.process_get_input = mp.Process(target=self.mp_hmi, args=())

        self.process_get_input.start()  # 启动接收进程

    def get_hmi(self):
        # 接站
        # 坐标转换
        self.plock.acquire()
        hmi_ctw = np.array(self.hmi_ctw[:])
        self.plock.release()
        return hmi_ctw[0]

    def mp_hmi(self):  # 读取RTX采集的状态数据
        while True:
            hmi_ctw = input('【HMI】输入命令\n')
            self.plock.acquire()  # 锁住
            self.hmi_ctw[0] = hmi_ctw
            self.plock.release()  # 释放

    def close(self):
        self.process_get_input.terminate()
        print('【HMI】子进程已终止。')


if __name__ == '__main__':
    # 测试程序

    mc = multicoreHMI(0.005)
    i = 0

    # while True:
    #     print('main process = ',i)
    #     print(mc.get_hmi())
    #     time.sleep(0.5)  # 等待稳态实现
    #     i += 1
