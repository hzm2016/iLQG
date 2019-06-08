import winsound
import traceback
import matplotlib.pyplot as plt
import time
from iENV import Env_PeginHole
from iDDPG import DDPG
import os
import pickle
import numpy as np
from HMI import multicoreHMI
from iIMAGINATION import imaginationROLLOUTS
from iLQR_controller import iLQR, fd_Cost, fd_Dynamics, myCost
import numdifftools as nd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略的CPU警告

MAX_TRAIN_EPISODES = 25  # 训练episode个数
MAX_EP_STEPS = 100

MAX_EVA_EPISODES = 10  # 评价episode个数

# 训练？评价？
RESTORE_AGENT = True
ON_TRAIN = True  # True False

ENABLE_IR = False  # 是否使用 imaginationROLLOUTS
NUM_IR = 5

ENABLE_ALERT = False

ENABLE_ILQR = False

PLAY_SOUND = False  # 播放提示音
SHOW_TRAIN_RESULT = False
SHOW_EVAL_RESULT = False
TRAIN_ROUND = 5

USE_ACTION_BIAS = False
NUM_DEMON = 10

TUBE_ENABLE = False

SAVE_INTERVAL = 10

path_train_seqs_record = 'myRecord/train_seqs_record.pkl'
path_eval_seqs_record = 'myRecord/eval_seqs_record.pkl'


def get_hmi():
    ctw_hmi = hmi.get_hmi()

    global play_sound

    if ctw_hmi == 'S':
        play_sound = True
        print('【EPISODE提示音】开启')
    if ctw_hmi == 's':
        play_sound = False
        print('【EPISODE提示音】关闭')


def train(env, agent_ddpg):  # start training

    # agent_ddpg = DDPG(a_dim, s_dim)

    global play_sound
    global enable_ir
    global enable_ilqr

    env.seed(200)
    file_log = open("Log_train.txt", 'w')
    train_seqs_record = []

    if RESTORE_AGENT:
        agent_ddpg.restore()  # 读取之前的训练成果
        fr = open(path_train_seqs_record, 'rb')
        train_seqs_record = pickle.load(fr)
        fr.close()

    if agent_ddpg.iid > 0:
        fr = open(path_train_seqs_record, 'rb')
        train_seqs_record = pickle.load(fr)
        fr.close()

    action_std = 0.1  # 动作噪声标准差。对应action范围[-1,1]
    action_bias_weight = 0.1

    ir = imaginationROLLOUTS()

    for i in range(MAX_TRAIN_EPISODES):

        get_hmi()

        s, _ = env.reset(agent_ddpg)

        ir.reset_localFitting()

        ep_reward = 0
        ep_step = 0

        seq_record = []

        s_terminal = 0
        flag_emergency = False

        if i >= 50:
            enable_ir = False
        if agent_ddpg.flag_train_start:
            enable_ilqr = False

        ilqr_a_init = np.zeros(a_dim)

        for j in range(MAX_EP_STEPS):

            a_raw = agent_ddpg.choose_action(s)

            # iLQR计算控制策略
            if enable_ilqr and ir.flag_ready:
                def f(x, u):
                    assert len(x) == s_dim, x.shape
                    assert len(u) == a_dim, u.shape
                    x_, _, _, _ = ir.pred_Kalmanfilter(x, u)
                    return x_

                def l(x, u):
                    reward = env.get_running_cost(u, x)
                    return reward

                def l_terminal(x):
                    reward = env.get_reward_terminal(x)
                    return reward

                dynamics = fd_Dynamics(f, s_dim, a_dim)
                cost = fd_Cost(l, l_terminal, s_dim, a_dim)

                # print('x = ', s)
                # print('u = ', a_raw)
                # print('l = ', l(s, a_raw))
                # l_x = cost.l_x(s, a_raw)
                # l_u = cost.l_u(s, a_raw)
                # f_u = dynamics.f_u(s, a_raw)
                #
                # print('l_u = ', l_u)
                # print('l_x = ', l_x)
                # print('f_u = ', f_u)

                N = 1  # Number of time-steps in trajectory.
                x_init = s  # Initial state.
                u_init = np.array([ilqr_a_init])  # Random initial action path.

                ilqr = iLQR(dynamics, cost, N)
                xs, us = ilqr.fit(x_init, u_init)

                a_raw = us[0]
                a_raw[1] = -abs(a_raw[1])
                a_raw = np.tanh(a_raw)

            if enable_ilqr and ir.flag_jamming:
                gain_std = 3
            else:
                gain_std = 1

            # 添加探索噪声
            act_ddpg = agent_ddpg.action_add_noise(a_raw, gain_std * action_std)
            act_safe = agent_ddpg.choose_action_safe(s)
            ilqr_a_init = act_ddpg.copy()

            # 【IR】安全预警
            if ENABLE_ALERT and ENABLE_IR and ir.flag_ready and np.min(act_safe) == 1:
                ir_s = s
                ir_a = act_ddpg
                ir_s_, _, _, _ = ir.pred_Kalmanfilter(ir_s, ir_a)
                ir_f_ = ir_s_[0:3]
                if np.any(ir_f_ > 1):
                    print('IR警告：可能出现接触力超限')
                    num_try = 10
                    while np.any(ir_f_ > 1) and num_try > 0:
                        print('  --正在重新选择action')
                        act_ddpg = agent_ddpg.action_add_noise(a_raw, action_std)
                        ir_s = s
                        ir_a = act_ddpg
                        ir_s_, _, _, _ = ir.pred_Kalmanfilter(ir_s, ir_a)
                        ir_f_ = ir_s_[0:3]
                        num_try = num_try - 1

            if flag_emergency:
                flag_emergency = False

            print('    <<<< 动作 = [%6.3f,' % act_ddpg[0], ' %6.3f,' % act_ddpg[1], ' %6.3f],' % act_ddpg[2],
                  ' [%6.3f,' % act_ddpg[3], ' %6.3f,' % act_ddpg[4], ' %6.3f] ' % act_ddpg[5],
                  ' [%6.3f,' % act_safe[0], ' %6.3f,' % act_safe[1], ' %6.3f] ' % act_safe[2])
            try:
                s_, r, s_terminal = env.step(act_ddpg, act_safe, agent_ddpg)
            except Exception as e:
                print('【训练组】运行时出现异常。' + str(e))
                traceback.print_exc()
                s_ = s.copy()
                r = -1
                s_terminal = -40

            print('    >>>> 接触力 = [%6.3f,' % s_[0], ' %6.3f,' % s_[1], ' %6.3f],' % s_[2],
                  ' 进程 = %5.3f,' % s_[12],
                  ' 奖惩 = %6.3f' % r)

            if s_terminal == 0 and j == MAX_EP_STEPS - 1:
                # 到达episode最大数目
                s_terminal = -10

            if np.min(act_safe) == 1:  # act_safe未被激活
                agent_ddpg.store_transition(s, act_ddpg, r, s_, s_terminal)
                ir.store_and_fitting(s, act_ddpg, r, s_, s_terminal)
                # 【IR】添加扩增训练数组
                if enable_ir and ir.flag_ready:
                    for nn in range(NUM_IR):
                        ir_s = s
                        ir_a = agent_ddpg.choose_action(ir_s)
                        ir_a = agent_ddpg.action_add_noise(ir_a, action_std)
                        ir_s_, _, _, _ = ir.pred_Kalmanfilter(ir_s, ir_a)
                        ir_r, ir_s_terminal = env.get_reward(ir_a, ir_s_)
                        agent_ddpg.store_transition(ir_s, ir_a, ir_r, ir_s_, ir_s_terminal)

            seq_record.append([s, act_ddpg, r, s_, s_terminal])

            if agent_ddpg.flag_train_start:
                action_std *= .999995  # decay the action randomness
                action_bias_weight *= .996
            # agent_ddpg.train(TRAIN_ROUND)  # 在env.step中训练以复用时间

            s = s_
            ep_reward += r
            ep_step += 1

            if s_terminal == -2:
                flag_emergency = True

            if s_terminal == 1 or s_terminal == -1 or s_terminal == -4 or s_terminal == -40:
                break

        if 1 == s_terminal and play_sound:
            winsound.PlaySound('sound/feixin.wav', winsound.SND_ASYNC)
        if s_terminal < 0 and play_sound:
            winsound.PlaySound('sound/YY.wav', winsound.SND_ASYNC)

        rps = float(ep_reward) / float(ep_step)
        stats_list = ['Episode: %i ' % i, ' Reward: %.2f ' % ep_reward, ' Rps: %.3f ' % rps,
                      ' Explore: %.2f ' % action_std,
                      ' processY: %.3f ' % s[12], ' Step: %i ' % ep_step,
                      ' done ' if 1 == s_terminal else '', ' coercion ' if -1 == s_terminal else '']
        stats = ''
        stats = stats.join(stats_list)
        print(stats)
        file_log.write(stats + '\n')
        file_log.flush()
        train_seqs_record.append(seq_record)  # 保存过程数据
        if i % 5 == 0 and i != 0:
            # 保存网络
            agent_ddpg.save()
            fw = open(path_train_seqs_record, 'wb')
            pickle.dump(train_seqs_record, fw)
            fw.close()

    _, _ = env.reset()
    file_log.close()
    # 保存网络
    agent_ddpg.save()
    fw = open(path_train_seqs_record, 'wb')
    pickle.dump(train_seqs_record, fw)
    fw.close()

    del agent_ddpg


def eval():
    agent_ddpg = DDPG(a_dim, s_dim)

    global play_sound
    global enable_ir
    global enable_ilqr

    env.seed(200)
    file_log = open("Log_train.txt", 'w')
    train_seqs_record = []

    if RESTORE_AGENT:
        agent_ddpg.restore()  # 读取之前的训练成果
        fr = open(path_train_seqs_record, 'rb')
        train_seqs_record = pickle.load(fr)
        fr.close()

    global play_sound

    env.seed(200)
    file_log = open("Log_train.txt", 'w')
    eval_seqs_record = []

    agent_ddpg.restore()  # 读取之前的训练成果

    for i in range(MAX_TRAIN_EPISODES):

        get_hmi()

        s, _ = env.reset()
        ep_reward = 0
        ep_step = 0

        seq_record = []

        s_terminal = 0
        flag_emergency = False
        for j in range(MAX_EP_STEPS):

            act_ddpg = agent_ddpg.choose_action(s)
            act_safe = agent_ddpg.choose_action_safe(s)

            if flag_emergency:
                flag_emergency = False

            print('    <<<< 动作 = [%6.3f,' % act_ddpg[0], ' %6.3f,' % act_ddpg[1], ' %6.3f],' % act_ddpg[2],
                  ' [%6.3f,' % act_ddpg[3], ' %6.3f,' % act_ddpg[4], ' %6.3f] ' % act_ddpg[5],
                  ' [%6.3f,' % act_safe[0], ' %6.3f,' % act_safe[1], ' %6.3f] ' % act_safe[2])
            s_, r, s_terminal = env.step(act_ddpg, act_safe)
            print('    >>>> 接触力 = [%6.3f,' % s_[0], ' %6.3f,' % s_[1], ' %6.3f],' % s_[2],
                  ' 进程 = %5.3f,' % s_[12],
                  ' 奖惩 = %6.3f' % r)

            seq_record.append([s, act_ddpg, r, s_, s_terminal])

            s = s_
            ep_reward += r
            ep_step += 1

            if s_terminal == -2:
                flag_emergency = True

            if s_terminal == 1 or s_terminal == -1 or s_terminal == -4 or s_terminal == -5:
                break

        if 1 == s_terminal and play_sound:
            winsound.PlaySound('sound/feixin.wav', winsound.SND_ASYNC)
        if s_terminal < 0 and play_sound:
            winsound.PlaySound('sound/YY.wav', winsound.SND_ASYNC)

        rps = float(ep_reward) / float(ep_step)
        stats_list = ['Episode: %i ' % i, ' Reward: %.2f ' % ep_reward, ' Rps: %.3f ' % rps,
                      ' processY: %.3f ' % (s[-1] * 0.5 + 0.5), ' Step: %i ' % ep_step,
                      ' done ' if 1 == s_terminal else '', ' coercion ' if -1 == s_terminal else '']
        stats = ''
        stats = stats.join(stats_list)
        print(stats)
        file_log.write(stats + '\n')
        file_log.flush()
        eval_seqs_record.append(seq_record)  # 保存过程数据

    _, _ = env.reset()
    file_log.close()
    # 保存数据
    fw = open(path_eval_seqs_record, 'wb')
    pickle.dump(eval_seqs_record, fw)
    fw.close()


if __name__ == '__main__':
    # 主程序

    env = Env_PeginHole()  # 初始化机器人环境
    env.connectRobot(False)
    env.robot_go_terminal()
    env.correct_residual_bias()

    s_dim = env.state_space.shape[0]
    a_dim = env.action_space.shape[0]

    if ON_TRAIN:
        print('【训练组】')
        try:
            for k in range(1):
                print('  第%i组' % k)

                agent_ddpg = DDPG(a_dim, s_dim, iid=4)

                hmi = multicoreHMI()
                global play_sound
                play_sound = PLAY_SOUND

                global enable_ir
                enable_ir = ENABLE_IR

                global enable_ilqr
                enable_ilqr = ENABLE_ILQR

                train(env, agent_ddpg)

        except Exception as e:
            print('【训练组】运行时出现异常。' + str(e))
            traceback.print_exc()
    else:
        print('【评价组】')
        eval()

    winsound.PlaySound('sound/finish.wav', winsound.SND_ASYNC)
    env.close()  # 关闭环境（包括多个进程）
    print('【Main消息】主进程已结束。')
    time.sleep(5)
