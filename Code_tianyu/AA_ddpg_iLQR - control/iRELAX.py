import numpy as np
from iLSTM import BATCH_MANAGE
from iENV import Env_PeginHole
import time
import winsound

NUM_DEMON = 10

if __name__ == '__main__':
    # 主程序
    env = Env_PeginHole()  # 初始化机器人环境
    env.connectRobot(False)

    batchm = BATCH_MANAGE(s_dim=4, a_dim=3)

    process = env.reset_demonstrate()
    for i in range(1000):
        process, a_zip, done = env.step_demonstrate()
        time.sleep(0.5)

    env.close()  # 关闭环境（包括多个进程）
    time.sleep(5)
