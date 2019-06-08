import time, threading


class multicoreHMI(object):

    def __init__(self, cycle_time=0.01):
        # 定义共享内存
        self.hmi_ctw = ''

        self.process_get_input = threading.Thread(target=self.mp_hmi)

        self.process_get_input.start()  # 启动接收进程

    def get_hmi(self):
        result = self.hmi_ctw
        self.hmi_ctw = ''
        return result

    def mp_hmi(self):  # 读取RTX采集的状态数据
        while True:
            self.hmi_ctw = input()
            print('hmi = ', self.hmi_ctw)


if __name__ == '__main__':
    # 测试程序

    mc = multicoreHMI()
    i = 0

    while True:
        print('main process = ', i)
        ctw_hmi = mc.get_hmi()
        if ctw_hmi != '':
            print(ctw_hmi)
        time.sleep(0.5)
        i += 1
