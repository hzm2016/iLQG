import tensorflow as tf
import numpy as np
import pickle
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

PRINT_INFO = True  # True False

CELL_UNITS = 128  # RNN隐层单元数目 64
CELL_dropout = 0.3
LR = 0.006  # 学习率 0.006
MIN_TRAIN_NUM = 1000  # 每次训练的迭代次数 最少
MAX_TRAIN_NUM = 10000  # 每次训练的迭代次数 最大
PATH_NET_SAVE = './RNN_Neronet'


class LSTM_RNN(object):
    def __init__(self, s_dim, a_dim):
        self.sess = tf.Session()

        self.cell_size = CELL_UNITS

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.latest_loss = 0
        self.latest_training_num = 0

        self.build_network()

    def build_network(self):
        with tf.name_scope('inputs'):
            # 输入：observation
            self.obsv_input = tf.placeholder(tf.float32, [None, None, self.s_dim])
            # 输入：target
            self.target_input = tf.placeholder(tf.float32, [None, None, self.a_dim])
            # 输入：序列长度，[50,52,...,48]
            self.seq_lengths = tf.placeholder(tf.int32, [None])

            shape = tf.shape(self.obsv_input)
            self.batch_size, self.max_seq_length = shape[0], shape[1]

        # RNN及输出层
        with tf.variable_scope('LSTM_PARA', reuse=tf.AUTO_REUSE):
            x = self.input_layer(self.obsv_input)
            x, _ = self.cell_layers(x)
            self.pred_seq = self.output_layer(x)

        self.para_network = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM_PARA')

        # 计算loss
        with tf.name_scope('loss'):
            self.loss = self.compute_loss()
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def cell_layers(self, inputs):
        rnn_cells = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        y, cell_final_state = tf.nn.dynamic_rnn(cell=rnn_cells,
                                                inputs=inputs,
                                                dtype=tf.float32,
                                                sequence_length=self.seq_lengths)
        return y, cell_final_state

    def output_layer(self, inputs):
        # 输出层
        pred = tf.layers.dense(inputs, self.a_dim, name='output_layer')
        return pred

    def input_layer(self, inputs):
        # 输入层
        y = tf.layers.dense(inputs, self.cell_size, name='input_layer')
        return y

    def compute_loss(self):
        mse = self.ms_error(self.pred_seq, self.target_input)
        loss = tf.reduce_mean(mse)

        with tf.name_scope('average_cost'):
            return loss

    def train(self, padding_obsv, padding_target, seq_lengths):
        self.feed_dict_train = {self.obsv_input: padding_obsv,
                                self.target_input: padding_target,
                                self.seq_lengths: seq_lengths}

        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict=self.feed_dict_train)

        return loss

    def give_seq_pred(self, obsv_seq):
        seq_len = np.array([len(obsv_seq)])
        format_obsv = np.array(obsv_seq)[np.newaxis, :]

        feed_dict = {self.obsv_input: format_obsv,
                     self.seq_lengths: seq_len}
        action_seq = self.sess.run(self.pred_seq,
                                   feed_dict=feed_dict)
        return np.array(action_seq)[0].tolist()

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, PATH_NET_SAVE, write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, PATH_NET_SAVE)

    def assign_para(self, para_ref):
        assign = [tf.assign(target, ref)
                  for target, ref in
                  zip(self.para_network, para_ref)]
        self.sess.run(assign)

    def give_para(self):
        return self.sess.run(self.para_network)

    def __del__(self):
        pass

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))


class BATCH_MANAGE(object):
    def __init__(self, s_dim, a_dim):
        self.path_save = 'batch_list.pkl'

        self.max_length = 0
        self.__s_dim = s_dim
        self.__a_dim = a_dim
        self.list_states = list()
        self.list_action = list()

        self.max_batch_size = 100
        self.index_overwrite = 0

    def add_seq(self, s_seq, a_seq):
        assert len(s_seq) == len(a_seq)
        if len(self.list_action) < self.max_batch_size:
            self.list_states.append(s_seq)
            self.list_action.append(a_seq)
        else:
            # batchsize大于指定长度则覆盖
            self.list_states[self.index_overwrite] = s_seq
            self.list_action[self.index_overwrite] = a_seq
            self.index_overwrite += 1
            self.index_overwrite = self.index_overwrite % self.max_batch_size

    @property
    def batch_size(self):
        return len(self.list_states)

    def update_batch(self):
        self.seq_lengths = [len(s) for s in self.list_states]
        self.max_length = max(self.seq_lengths)

        # 生成一个全零array来存放padding后的数据集
        self.padding_states = np.zeros([self.batch_size, self.max_length, self.__s_dim])
        self.padding_action = np.zeros([self.batch_size, self.max_length, self.__a_dim])
        self.mask_dataset = np.zeros([self.batch_size, self.max_length])

        for idx, seq in enumerate(self.list_states):
            self.padding_states[idx, :len(seq), :] = seq
            self.mask_dataset[idx, :len(seq)] = 1

        for idx, seq in enumerate(self.list_action):
            self.padding_action[idx, :len(seq), :] = seq

    def save(self, path=None):
        # self.update_batch()
        if path is not None:
            self.path_save = path
        fw = open(self.path_save, 'wb')
        pickle.dump((self.list_states, self.list_action), fw)
        fw.close()

    def restore(self, path=None):
        if path is not None:
            self.path_save = path
        print(self.path_save)
        fr = open(self.path_save, 'rb')
        self.list_states, self.list_action = pickle.load(fr)
        fr.close()
        self.update_batch()


def train():
    # 主程序

    padding_obsv, padding_target, seq_lengths = \
        batchm.padding_states, batchm.padding_action, batchm.seq_lengths

    print(seq_lengths)

    agentRNN = LSTM_RNN(s_dim_rnn, a_dim_rnn)

    for i in range(num_train):
        loss = agentRNN.train(padding_obsv, padding_target, seq_lengths)
        if i % 20 == 0:
            stats_list = ['num: %d ' % i,
                          ' loss: %.8f ' % loss]
            stats = ''
            stats = stats.join(stats_list)
            print(stats)

    agentRNN.save()


def eval():
    padding_states, padding_action, seq_lengths = \
        batchm.padding_states, batchm.padding_action, batchm.seq_lengths
    print('训练集加载完毕。batchsize=%d' % batchm.batch_size)

    agent_test = LSTM_RNN(s_dim_rnn, a_dim_rnn)
    agent_test.restore()

    for i in range(batchm.batch_size):
        seq_length = seq_lengths[i]
        obsv = padding_states[i][:seq_length]
        target = padding_action[i][:seq_length]
        time = np.array(range(seq_length))

        prediction = np.array(agent_test.give_seq_pred(obsv))

        plt.plot(time, target[:, 1], 'r', time, prediction[:, 1], 'b--')
        plt.pause(2)
        plt.clf()

    # 图像窗口不关闭
    plt.show()


if __name__ == '__main__':
    s_dim_rnn = 4
    a_dim_rnn = 3

    num_train = 8000

    batchm = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)
    batchm.restore('batch_list_process.pkl')

    # train()
    eval()
