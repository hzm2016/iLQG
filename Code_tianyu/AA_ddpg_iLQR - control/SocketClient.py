import numpy as np
import multiprocessing as mp
import time
import socket
import struct
import kinematicsPack as kp


class multicoreCommunication(object):

    def __init__(self, cycle_time, print_info=True):
        # 定义共享内存
        self.vc_pos = mp.Array('f', [0, 0, 0])  # 接收区
        self.vc_stf = mp.Array('f', [0, 0, 0])
        self.vc_ctw = mp.Array('i', [0, 0, 0])
        self.clock = mp.Lock()  # 进程锁
        self.vr_pos = mp.Array('f', [0, 0, 0])  # 发送区
        self.vr_fce = mp.Array('f', [0, 0, 0])
        self.vr_ctw = mp.Array('i', [0, 0, 0])
        self.rlock = mp.Lock()  # 进程锁

        self.cycle_time = cycle_time
        self.print_info = print_info

        self.process_get = mp.Process(target=self.socket_upstream, args=())
        self.process_send = mp.Process(target=self.socket_downstream, args=())

        self.process_get.start()  # 启动接收进程
        self.process_send.start()  # 启动发送进程

        while not self.get_test():
            time.sleep(1)
        print('    【Socket消息】与机器人通信已建立。')

        self.residual_bias = np.array([0., 0., 0.])
        # self.correct_residual_bias()

    def correct_residual_bias(self):

        count = 0

        while count < 3:
            _, residual, _ = self.get_raw()  # 读取机器人反馈
            error = np.linalg.norm(residual - self.residual_bias)
            if error < 0.5:
                count += 1
            else:
                count = 0
            self.residual_bias = residual.copy()
            time.sleep(0.05)

        print('    【Socket消息】重力补偿校正完成。bias='+str(self.residual_bias))

    def get_r(self):
        # 接站
        # 坐标转换
        self.rlock.acquire()
        pos_j = np.array(self.vr_pos[:])
        force_j = np.array(self.vr_fce[:])
        ctw = np.array(self.vr_ctw[:])
        self.rlock.release()
        pos_tcp = kp.cal_tcp(pos_j)
        force_tcp = kp.cal_Fext(force_j-self.residual_bias, pos_j)
        if self.print_info:
            print('    接站>> ' + str(pos_tcp) + str(force_tcp) + str(ctw))
        return pos_tcp, force_tcp, ctw

    def get_raw(self):
        # 接站
        # 坐标转换
        self.rlock.acquire()
        pos_j = np.array(self.vr_pos[:])
        force_j = np.array(self.vr_fce[:])
        ctw = np.array(self.vr_ctw[:])
        self.rlock.release()
        if self.print_info:
            print('    关节空间>> ' + str(pos_j) + str(force_j) + str(ctw))
        return pos_j, force_j, ctw

    def get_test(self):

        # 接站：用于检测原始数据
        # 无运动学变换
        self.rlock.acquire()
        pos_j = np.array(self.vr_pos[:])
        force_j = np.array(self.vr_fce[:])
        ctw = np.array(self.vr_ctw[:])
        self.rlock.release()
        time.sleep(0.01)
        self.rlock.acquire()
        pos_j2 = np.array(self.vr_pos[:])
        force_j2 = np.array(self.vr_fce[:])
        self.rlock.release()
        print('    【Socket消息】正在尝试与机器人控制器通信')
        if self.print_info:
            print('    接站检验>> ' + str(pos_j) + str(force_j) + str(ctw))
        if 0 == np.sum(np.abs(pos_j)) or 0 == np.sum(np.abs(force_j)):
            return False
        else:
            if np.all(np.logical_and(force_j == force_j2, pos_j == pos_j2)):
                return False
            else:
                return True

    def send_c(self, pos_c, stf_c, c_ctw):
        # 送站
        # 坐标转换
        self.clock.acquire()
        self.vc_pos[:] = kp.cal_J246(pos_c)
        self.vc_stf[:] = stf_c
        self.vc_ctw[:] = c_ctw
        self.clock.release()
        if self.print_info:
            print('    送站<<<< ' + str(pos_c) + str(self.vc_stf[:]) + str(self.vc_ctw[:]))

    def socket_upstream(self):  # 读取RTX采集的状态数据
        sServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sServer.connect(('192.168.100.20', 8020))  # 拔通电话   #改成服务端网卡IP地址和端口
        while True:
            data = sServer.recv(36)
            unpacked_data = struct.unpack('ffffffiii', data)
            # if len(data) > 0:
            #     print('--->来自Server：', unpacked_data)  # 调试用，输出一下读取结果
            self.rlock.acquire()  # 锁住
            self.vr_pos[:] = unpacked_data[0:3]
            self.vr_fce[:] = unpacked_data[3:6]
            self.vr_ctw[:] = unpacked_data[6:9]
            self.rlock.release()  # 释放

    def socket_downstream(self):  # 向WIN32写入计算结果
        sServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sServer.connect(('192.168.100.20', 8010))  # 拔通电话   #改成服务端网卡IP地址和端口
        i = 1
        while True:
            self.clock.acquire()  # 锁住
            c_pos = self.vc_pos[:]
            c_stf = self.vc_stf[:]
            c_ctw = self.vc_ctw[:]
            self.clock.release()  # 释放
            packed_data = struct.pack('ffffffiii', c_pos[0], c_pos[1], c_pos[2], c_stf[0], c_stf[1], c_stf[2],
                                      c_ctw[0], c_ctw[1], c_ctw[2])
            stw = sServer.send(packed_data)
            # print('      <---第%i次发送' % i)
            time.sleep(self.cycle_time)
            i += 1

    def close(self):
        self.process_get.terminate()
        self.process_send.terminate()
        print('    【Socket消息】子进程已终止。')


if __name__ == '__main__':
    # 测试程序

    mc = multicoreCommunication(0.005)

    c_pos = np.array([0.6335, -0.206, -1.55968])
    c_stf = np.array([0, 32, 12])
    c_ctw = np.array([1, 0, 0])  # c_ctw[0] = -1 急停 0 高阻 1 正常

    while True:
        mc.send_c(c_pos, c_stf, c_ctw)  # 发送控制指令
        time.sleep(0.5)  # 等待稳态实现
        pos, force, ctw = mc.get_r()  # 读取机器人反馈
