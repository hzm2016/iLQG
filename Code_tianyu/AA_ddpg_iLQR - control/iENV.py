import kinematicsPack as kp
import numpy as np
from gym import spaces
from gym.utils import seeding
import time
from SocketClient import multicoreCommunication
import winsound
from iEMAIL import send_email


class Env_PeginHole(object):
    # start_pos_x = 0.63276377  # 0.633461
    # start_pos_y = - 0.20655886  # -0.2028786 -0.206
    # start_pos_w = -1.55968

    start_pos = np.array([0.63276377, - 0.20655886, -1.55968])

    final_pos = np.array([0.63276377, - 0.20655886 + 0.005, -1.55968])

    def __init__(self):

        self.y_hole_bottom = -0.2420
        self.depth_hole = 0.036

        # 状态变量
        self.pos = np.array([0., 0., 0.])
        self.force = np.array([0., 0., 0.])
        self.ctw = np.array([0, 0, 0])
        # 控制变量
        self.c_pos = np.array([0., 0., 0.])
        self.c_stf = np.array([200, 200, 20])
        self.c_ctw = np.array([0, 0, 0])  # c_ctw[0] = -1 急停 0 高阻 1 正常

        self.istep = 0

        # 接触力限制
        self.force_limit = np.array([40., 40., 5.])
        # action限制 0.002
        self.action_bound_high = np.array([0.001, 0.001, 0.2 * np.pi / 180,
                                           4000, 4000, 200])
        self.action_bound_low = np.array([-0.001, -0.001, -0.2 * np.pi / 180,
                                          0, 0, 0])
        # state参考范围
        self.state_bound_high = np.array([40, 40, 5,  # 接触力 0 1 2
                                          0.01, 0.01, 2 * np.pi / 180,  # 偏移量 3 4 5
                                          0.01, 0.01, 2 * np.pi / 180,  # 位移增量 6 7 8
                                          40, 40, 5,  # 力增量 9 10 11
                                          1])  # 12 进度
        self.state_bound_low = np.array([-40, -40, -5,
                                         -0.01, -0.01, -2 * np.pi / 180,
                                         -0.01, -0.01, -2 * np.pi / 180,
                                         -40, -40, -5,
                                         0])

        assert list(self.action_bound_high) > list(self.action_bound_low)
        self.action_scale = (self.action_bound_high - self.action_bound_low) / 2
        self.action_bias = (self.action_bound_high + self.action_bound_low) / 2
        assert list(self.state_bound_high) > list(self.state_bound_low)
        self.state_scale = (self.state_bound_high - self.state_bound_low) / 2
        self.state_bias = (self.state_bound_high + self.state_bound_low) / 2

        self.action_space = spaces.Box(low=self.action_bound_low, high=self.action_bound_high)
        self.state_space = spaces.Box(low=self.state_bound_low, high=self.state_bound_high)

        self.seed()

    def connectRobot(self, print_cmnctn_info=True):
        self.mc = multicoreCommunication(0.005, print_cmnctn_info)
        winsound.PlaySound('sound/request.wav', winsound.SND_ASYNC)
        time.sleep(1)

    def correct_residual_bias(self):
        self.mc.correct_residual_bias()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act_agent, act_safe, agent_ddpg=None):

        a = self.decode_action(act_agent)
        act_apos = a[0:3]
        act_stf = a[3:6]

        # 跟随因子。1-虚拟位移完全不跟随实际位移，0-虚拟位移完全跟随实际位移
        self.c_pos = self.pos + act_safe * (self.c_pos - self.pos)
        self.c_pos += act_apos

        self.c_stf = act_stf
        for i in range(3):
            if self.c_stf[i] < 0:
                self.c_stf[i] = 0

        self.istep += 1
        last_pos = self.pos
        last_force = self.force

        self.c_ctw[0] = 1
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 发送控制指令
        # 趁机训练Agent
        start_timer = time.time()
        if agent_ddpg is not None:
            agent_ddpg.train(5)
        end_timer = time.time()
        interval_timer = end_timer - start_timer
        time.sleep(max(0, 0.5 - interval_timer))
        # 等待稳态实现 0.5秒~10次迭代
        self.pos, self.force, self.ctw = self.mc.get_r()  # 读取机器人反馈

        deviation_pos = self.c_pos - self.pos

        inc_pos = self.pos - last_pos  # 实际的机器人位移。用以判断是否卡死
        inc_force = self.force - last_force

        s_ = np.concatenate((self.force, deviation_pos, inc_pos, inc_force, [self.insert_process]))
        s_ = self.code_state(s_)

        r, s_terminal = self.get_reward(act_agent, s_)  # 计算reward

        if s_terminal == -2:
            print('  【ENV警告】接触力超限。')
        if s_terminal == -4:
            print('  【ENV警告】接触力超限超过2倍！')
        if s_terminal == -1:
            self.c_ctw[0] = 0  # 机器人高阻
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
            print('  【ENV警告】机器人已跑飞。')
            time.sleep(0.5)  # 等待稳态实现
        if s_terminal == -5:
            self.c_ctw[0] = 0
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
            print('  【ENV警告】机器人严重偏离平衡位置。')
            time.sleep(0.5)
        if s_terminal == 1:
            self.c_ctw[0] = 0  # 机器人高阻
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)

        return s_, r, s_terminal

    def get_reward(self, a, s_):
        # 计算Reward
        r = self.get_running_cost(a, s_)

        on_goal = False
        apos_limit = self.action_bound_high[0:3]
        stf_limit = self.action_bound_high[3:6]
        devi_limit = self.state_bound_high[3:6]

        a = self.decode_action(a)
        s_ = self.decode_state(s_)

        act_apos = a[0:3]
        act_stf = a[3:6]
        y_move = s_[7]
        force = s_[0:3]
        pos_devi = s_[3:6]
        process = s_[12]

        # reward_move = 0.5 * np.clip(-y_move / self.action_bound_high[1], -1, 1)
        # punish_force = 0.1 * np.clip(max(abs(force / self.force_limit)) ** 2, 0, 1)
        # punish_stf = 0.01 * np.clip(max(abs(act_stf / stf_limit)) ** 2, 0, 1)
        # punish_apos = 0.01 * np.clip(max(abs(act_apos / apos_limit)) ** 2, 0, 1)
        # punish_devi = 0.1 * np.clip(max(abs(pos_devi / devi_limit)) ** 2, 0, 1)
        # r = reward_move - punish_force - punish_stf - punish_apos - punish_devi

        s_terminal = 0

        safty_force = abs(force) <= self.force_limit
        # 【异常】接触力超限
        if False in safty_force:
            r = r - 0.5
            s_terminal = -2
        safty_force = abs(force) <= 2 * self.force_limit
        if False in safty_force:
            r = r - 0.5
            s_terminal = -4

        # 【异常】平衡位置严重偏离
        if np.any(np.abs(pos_devi) > np.array([2, 2, 2])):
            r = r - 1
            s_terminal = -5

        # 【异常】运动范围超限
        if process < 0:
            r = r - 1
            s_terminal = -1  # 任务强制终止

        # 【成功】检查是否完成任务
        if process > 0.975:
            on_goal = True
        if on_goal and s_terminal == 0:
            s_terminal = 1  # 任务已完成
            r = r + 1

        return r, s_terminal

    def get_running_cost(self, a, s_):
        # 计算Reward

        a = self.decode_action(a)
        s_ = self.decode_state(s_)

        apos_limit = self.action_bound_high[0:3]
        stf_limit = self.action_bound_high[3:6]
        devi_limit = self.state_bound_high[3:6]

        act_apos = a[0:3]
        act_stf = a[3:6]
        y_move = s_[7]
        force = s_[0:3]
        pos_devi = s_[3:6]

        reward_move = 0.5 * np.tanh(-y_move / self.action_bound_high[1])
        punish_force = 0.5 * np.tanh(np.linalg.norm(force / self.force_limit / 3) ** 2)
        punish_stf = 0.1 * np.tanh(np.linalg.norm(act_stf / stf_limit / 3) ** 2)
        punish_apos = 0.1 * np.tanh(np.linalg.norm(act_apos / apos_limit / 3) ** 2)
        punish_devi = 0.2 * np.tanh(np.linalg.norm(pos_devi / devi_limit / 3) ** 2)

        r = reward_move - punish_force - punish_stf - punish_apos - punish_devi

        return r

    def get_reward_terminal(self, s_):
        # 计算Reward

        devi_limit = self.state_bound_high[3:6]

        s_ = self.decode_state(s_)

        y_move = s_[7]
        force = s_[0:3]
        pos_devi = s_[3:6]

        reward_move = 0.5 * (-y_move / self.action_bound_high[1])
        punish_force = 0.1 * (force / self.force_limit).dot(force / self.force_limit)
        punish_devi = 0.1 * (pos_devi / devi_limit).dot(pos_devi / devi_limit)

        r = reward_move - punish_force - punish_devi

        return r

    def step_demonstrate(self):
        on_goal = False

        process_demon = 1 - (self.last_pos_demon[1] - self.y_hole_bottom) / self.depth_hole

        self.pos, _, _ = self.mc.get_r()  # 读取机器人反馈

        # 检查是否完成任务
        if abs(self.pos[1] - self.y_hole_bottom) < 0.001:
            on_goal = True
            self.c_ctw[0] = 0  # 机器人高阻
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)

        action_pos = self.pos - self.last_pos_demon
        action_pos = (action_pos - self.action_bias[0:3]) / self.action_scale[0:3]
        # action_pos = np.clip(action_pos, -1, 1)

        self.last_pos_demon = self.pos.copy()

        return process_demon, action_pos, on_goal

    def reset_demonstrate(self):

        low = np.array([0, 0, -3 * np.pi / 180])
        high = np.array([0, 0, 3 * np.pi / 180])

        pos_noise = self.np_random.uniform(low=low, high=high)

        # pos_noise = np.array([0, 0, 0 * np.pi / 180])

        self.c_stf[0] = 0
        self.c_stf[1] = 7000
        self.c_stf[2] = 5

        self.pos, self.force, self.ctw = self.mc.get_r()

        self.c_pos = (self.start_pos + self.pos) / 2

        print('  【Env消息】机器人开始置位。')
        self.c_ctw[0] = 1
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 插补一次
        time.sleep(0.7)
        self.c_pos = self.start_pos
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 轴孔拔出/回初始位置
        time.sleep(0.5)

        self.pos, self.force, self.ctw = self.mc.get_r()

        while self.pos[1] < (self.y_hole_bottom + self.depth_hole) - 0.002:
            self.c_stf[1] = 9000
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
            print('  【Env提示】机器人需要辅助置位。')
            winsound.PlaySound('sound/problem.wav', winsound.SND_ASYNC)
            time.sleep(0.5)
            self.c_stf[1] = 0
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
            time.sleep(0.5)
            self.pos, self.force, self.ctw = self.mc.get_r()

        # 示教用
        self.last_pos_demon = self.pos.copy()

        self.c_stf[0] = 6000
        self.c_stf[1] = 6000
        self.c_stf[2] = 200

        self.c_pos = self.start_pos + pos_noise  # 加入初始位置偏差

        self.c_ctw[0] = 1
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
        time.sleep(1)

        self.c_ctw[0] = 0  # 机器人高阻
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
        time.sleep(2)

        print('  【Env消息】机器人已置位。')

        winsound.Beep(800, 1000)
        self.c_ctw[0] = 1
        self.c_stf[0] = 0
        self.c_stf[1] = 0
        self.c_stf[2] = 0
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 机器人进入柔顺状态

        self.pos, self.force, self.ctw = self.mc.get_r()
        process_demon = 1 - (self.last_pos_demon[1] - self.y_hole_bottom) / self.depth_hole

        return process_demon

    def reset(self, agent_ddpg=None):

        low = np.array([-0.001, -0.001, -1 * np.pi / 180])
        high = np.array([0.001, 0.000000, 1 * np.pi / 180])

        pos_noise = self.np_random.uniform(low=low, high=high)

        # pos_noise = np.array([0, 0, 0 * np.pi / 180])

        self.c_stf[0] = 100
        self.c_stf[1] = 9000
        self.c_stf[2] = 100

        self.pos, self.force, self.ctw = self.mc.get_r()

        base_pos = self.pos.copy()
        vec_cur2start = self.start_pos - base_pos
        reset_distance_pos = np.linalg.norm(vec_cur2start[0:2])
        reset_distance_angle = np.linalg.norm(vec_cur2start[2:3])
        max_inc_pos = 0.003
        max_inc_angle = 0.6 * np.pi / 180
        num_steps = int(max(np.ceil(reset_distance_pos / max_inc_pos), np.ceil(reset_distance_angle / max_inc_angle)))
        vec_inc = vec_cur2start / num_steps

        print('  【Env消息】机器人开始置位。回零距离 %.5f m，角度 %.2f °，分为 %d 步' % (
            reset_distance_pos, reset_distance_angle * 180 / np.pi, num_steps))
        for i in range(num_steps):
            self.c_pos = base_pos + vec_inc * (i + 1)
            self.c_ctw[0] = 1
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 插补运动
            # 趁机训练Agent
            start_timer = time.time()
            if agent_ddpg is not None:
                agent_ddpg.train(5)
            end_timer = time.time()
            interval_timer = end_timer - start_timer
            time.sleep(max(0, 0.4 - interval_timer))
            # 等待稳态实现 0.5
            self.c_ctw[0] = 0
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 暂停一会儿
            time.sleep(0.1)

        self.pos, self.force, self.ctw = self.mc.get_r()

        count = 0
        while self.pos[1] < (self.y_hole_bottom + self.depth_hole) - 0.002:
            self.c_ctw[0] = 1
            self.c_stf[0] = 10
            self.c_stf[1] = 7000
            self.c_stf[2] = 10
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
            print('  【Env提示】机器人需要辅助置位。', count)
            if count <= 10:
                winsound.PlaySound('sound/problem.wav', winsound.SND_ASYNC)
            time.sleep(0.5)
            self.c_stf[1] = 10
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
            time.sleep(0.2)
            self.pos, self.force, self.ctw = self.mc.get_r()
            count += 1
            if count == 40:
                send_email('又被卡住了。。(￣^￣) ')
            if count == 80:
                self.close()
                raise Exception("【异常】无法回零")

        self.c_stf[0] = 6000
        self.c_stf[1] = 6000
        self.c_stf[2] = 200

        self.c_pos = self.start_pos + pos_noise  # 加入初始位置偏差

        self.c_ctw[0] = 1
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
        time.sleep(1)

        self.c_ctw[0] = 0  # 机器人高阻
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
        time.sleep(0.5)

        print('  【Env消息】机器人已置位。')

        # for i in range(3):
        #     self.c_pos[i] = self.pos[i]  # 初始化位置控制增量

        inc_pos = np.array([0., 0., 0.])
        inc_force = np.array([0., 0., 0.])

        deviation_pos = self.c_pos - self.pos

        s = np.concatenate((self.force, deviation_pos, inc_pos, inc_force, [self.insert_process]))
        return self.code_state(s), self.pos

    def code_state(self, state):
        process = state[12]
        state_coded = (state - self.state_bias) / self.state_scale
        state_coded[12] = process  # process范围为 0-1
        return state_coded

    def decode_state(self, state_coded):
        process = state_coded[12]
        state = self.state_scale * state_coded + self.state_bias
        state[12] = process
        return state

    def decode_action(self, action_raw):
        action = self.action_scale * action_raw + self.action_bias
        return action

    def render(self):
        # 并不显示
        pass

    def close(self):

        self.robot_go_terminal()

        print('  【Env消息】机器人即将去使能。')
        for k in range(30, 1, -1):
            time.sleep(1)
            print('  【Env消息】倒计时 ', k)

        # 机器人停机
        self.c_ctw[0] = -1
        self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)
        time.sleep(0.5)
        self.mc.close()

    def robot_go_terminal(self):
        # 机器人回到安全位置

        self.c_stf[0] = 200
        self.c_stf[1] = 9000
        self.c_stf[2] = 200

        self.pos, self.force, self.ctw = self.mc.get_r()

        base_pos = self.pos.copy()
        vec_cur2start = self.final_pos - base_pos
        reset_distance_pos = np.linalg.norm(vec_cur2start[0:2])
        reset_distance_angle = np.linalg.norm(vec_cur2start[2:3])
        max_inc_pos = 0.003
        max_inc_angle = 0.6 * np.pi / 180
        num_steps = int(max(np.ceil(reset_distance_pos / max_inc_pos), np.ceil(reset_distance_angle / max_inc_angle)))
        vec_inc = vec_cur2start / num_steps

        print('  【Env消息】机器人最终位置。回零距离 %.5f m，角度 %.2f °，分为 %d 步' % (
            reset_distance_pos, reset_distance_angle * 180 / np.pi, num_steps))
        for i in range(num_steps):
            self.c_pos = base_pos + vec_inc * (i + 1)
            self.c_ctw[0] = 1
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 插补运动
            time.sleep(0.6)
            self.c_ctw[0] = 0
            self.mc.send_c(self.c_pos, self.c_stf, self.c_ctw)  # 暂停一会儿
            time.sleep(0.1)
        time.sleep(0.5)

    def get_pos(self):
        return self.pos

    # 作为属性调用
    @property
    def insert_process(self):
        # 返回插孔进程 0~1
        pos, _, _ = self.mc.get_r()  # 读取机器人反馈
        return 1 - (pos[1] - self.y_hole_bottom) / self.depth_hole


if __name__ == '__main__':
    # 主程序

    env = Env_PeginHole()  # 初始化机器人环境
    env.connectRobot(False)
    # env.reset()

    for i in range(1000):
        pos, force, ctw = env.mc.get_r()
        print('>>>> 接触力 = [%6.3f,' % force[0], ' %6.3f,' % force[1], ' %6.3f],' % force[2],
              ' [%6.3f,' % pos[0], ' %6.3f,' % pos[1], ' %6.3f]' % pos[2])
        time.sleep(0.5)

    env.close()

    # while True:
    #     env.mc.get_r()
    #     time.sleep(1)
