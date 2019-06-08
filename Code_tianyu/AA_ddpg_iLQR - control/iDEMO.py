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

    hmi = input('进行轴孔装配示教。准备好了吗\n')
    print('好的。就位')


    for i in range(NUM_DEMON):
        print('开始第%d次示教' % (i + 1))
        process = env.reset_demonstrate()

        seq_s_zip = []
        seq_a_zip = []  # 没有用途

        done = False
        while not done:
            process, a_zip, done = env.step_demonstrate()
            print(process)
            print(a_zip)

            seq_s_zip.append([process] + a_zip.tolist())
            seq_a_zip.append(a_zip)

            time.sleep(0.5)

        winsound.PlaySound('sound/feixin.wav', winsound.SND_ASYNC)
        batchm.add_seq(seq_s_zip, seq_a_zip)
        batchm.save()
        time.sleep(0.5)

    env.reset()
    batchm.save()
    winsound.PlaySound('sound/finish.wav', winsound.SND_ASYNC)
    env.close()  # 关闭环境（包括多个进程）
    time.sleep(5)
