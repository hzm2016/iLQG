import tensorflow as tf
import tensorflow.contrib as tc  # 常用功能封装成的高级API
import numpy as np
import pickle
from ops import *

#####################  hyper parameters  ####################

LR_A = 0.0003  # learning rate for actor 0.001 0.0003
LR_C = 0.0001  # learning rate for critic 0.002 0.0003 0.0001
GAMMA = 0.99  # reward discount
TAU = 0.001  # soft replacement
MEMORY_CAPACITY = 9000
PATH_MEMORY_BACKUP = 'myRecord/memory_backup_ddpg.pkl'
PATH_NET_BACKUP = 'myRecord/DDPG_Neronet'
BATCH_SIZE = 100  # 128 500


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


###############################  DDPG  ####################################

class DDPG(object):

    # def __new__(self, a_dim, s_dim):
    #     return self.__init__(self, a_dim, s_dim)

    def __init__(self, a_dim, s_dim, iid=-1):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.pointer_memory = 0
        self.sess = tf.Session()

        self.training_started = 0
        self.static_2 = 0
        self.static_3 = 0

        self.a_dim = a_dim
        self.s_dim = s_dim

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.Terminal_S_ = tf.placeholder(tf.float32, [None, 1], 'terminals1')  # 1为完成，-1为终止

        self.normalize_layers = True  # 层归一化 True False

        self.iid = iid
        if iid == -1:
            name_agent = 'DDPG_PARA'
        else:
            name_agent = 'DDPG_PARA' + str(iid)

        with tf.variable_scope(name_agent):
            with tf.variable_scope('Actor'):
                self.actor = self._build_a(self.S, scope='eval', is_training=True)
                actor_target = self._build_a(self.S_, scope='target', is_training=True)
            with tf.variable_scope('Critic'):
                critic = self._build_c(self.S, self.actor, scope='eval', trainable=True)
                critic_target = self._build_c(self.S_, actor_target, scope='target', trainable=False)

        # 所有网络参数
        self.para_actor_eval = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_agent + '/Actor/eval')
        self.para_actor_trgt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_agent + '/Actor/target')
        self.para_critic_eval = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_agent + '/Critic/eval')
        self.para_critic_trgt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_agent + '/Critic/target')

        self.para_network = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_agent)

        # soft更新target网络
        self.soft_replace = [[tf.assign(a_t, (1 - TAU) * a_t + TAU * a_e), tf.assign(c_t, (1 - TAU) * c_t + TAU * c_e)]
                             for a_t, a_e, c_t, c_e in
                             zip(self.para_actor_trgt, self.para_actor_eval, self.para_critic_trgt,
                                 self.para_critic_eval)]
        # actor loss定义
        loss_actor = - tf.reduce_mean(critic)
        self.actor_train = tf.train.AdamOptimizer(LR_A).minimize(loss_actor, var_list=self.para_actor_eval)
        # critic loss定义
        q_target = self.R + (1. - abs(self.Terminal_S_)) * GAMMA * critic_target
        loss_critic = tf.losses.mean_squared_error(labels=q_target, predictions=critic)
        self.critic_train = tf.train.AdamOptimizer(LR_C).minimize(loss_critic, var_list=self.para_critic_eval)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.actor, {self.S: s[np.newaxis, :]})[0]

    def choose_action_safe(self, s):

        act_safe = np.array([1., 1., 1.])

        force = s[0:3]
        force_abs = np.abs(force)
        deviation = s[3:6] * 10

        dims_unsafe = np.argwhere(force_abs > 1).flatten()
        for dim in dims_unsafe:  # 对所有力超限的自由度进行操作
            dv = deviation[dim]
            f = force[dim]
            if dv * f > 0:
                act_safe[dim] = 0.9 / force_abs[dim] ** 4

        return act_safe

    def train(self, train_round):

        if self.flag_train_start:
            for k in range(train_round):
                self.__train()

    def __train(self):

        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.pointer_memory if self.pointer_memory < MEMORY_CAPACITY else MEMORY_CAPACITY,
                                   size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1:2 * self.s_dim + self.a_dim + 1]
        b_s_terminal = bt[:, -1:].astype('float32')

        self.sess.run(self.actor_train,
                      {self.S: bs})
        self.sess.run(self.critic_train,
                      {self.S: bs, self.actor: ba, self.R: br, self.S_: bs_,
                       self.Terminal_S_: b_s_terminal})

    def store_transition(self, s, a, r, s_, s_terminal):
        if s_terminal != 0:
            s_terminal = 1  # s_terminal非零即一
        transition = np.hstack((s, a, [r], s_, [s_terminal]))
        index = self.pointer_memory % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer_memory += 1

    def _build_a(self, s, scope, is_training):
        with tf.variable_scope(scope):
            x = s
            # 第一层
            x = tf.layers.dense(x, 64)
            if self.normalize_layers:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            # 第二层
            x = tf.layers.dense(x, 64)
            if self.normalize_layers:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            # 输出层
            x = tf.layers.dense(x, self.a_dim,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)  # [-1,1]
        return x

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            x = s
            x = tf.layers.dense(x, 64)  # 第一层
            if self.normalize_layers:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, a], axis=-1)  # 矩阵连接
            x = tf.layers.dense(x, 64)  # 第二层
            if self.normalize_layers:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # 输出层
            x = tf.layers.dense(x, 1,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def flag_train_start(self):
        if 1 == self.training_started:
            return True
        if self.pointer_memory > BATCH_SIZE:
            self.training_started = 1
            print('  【DDPG】 Ready to learning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return True
        else:
            return False

    def action_add_noise(self, action_clean, action_std):

        action_noisy = np.clip(np.random.normal(action_clean, action_std), -1, 1)  # 添加探索噪声
        return action_noisy

    def save(self):
        # net
        saver = tf.train.Saver(self.para_network)
        saver.save(self.sess, PATH_NET_BACKUP, write_meta_graph=False)
        # memory
        fw = open(PATH_MEMORY_BACKUP, 'wb')
        pickle.dump((self.memory, self.pointer_memory), fw)
        fw.close()

    def restore(self):
        # net
        saver = tf.train.Saver(self.para_network)
        saver.restore(self.sess, PATH_NET_BACKUP)
        # memory
        try:
            fr = open(PATH_MEMORY_BACKUP, 'rb')
            (self.memory, self.pointer_memory) = pickle.load(fr)
            fr.close()
            self.training_started = 1
            print('  【DDPG】memory已恢复。Length = ', len(self.memory))
        except:
            print('  【DDPG】恢复memory失败。已略过。')
            self.training_started = 0
