import numpy as np
import matplotlib.pyplot as plt
from iLSTM import BATCH_MANAGE


def show_action():
    batchm = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)
    batchm.restore()

    padding_states, padding_action, max_length, seq_lengths = \
        batchm.padding_states, batchm.padding_action, batchm.max_length, batchm.seq_lengths
    print('训练集加载完毕')

    for i in range(batchm.batch_size):
        seq_length = seq_lengths[i]
        action_target = padding_action[i][:seq_length]
        time = np.array(range(seq_length))

        plt.plot(time, action_target[:, 1], 'r')
        plt.pause(2)
        plt.clf()


def show_comparision():
    batchm = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)
    batchm.restore()

    batchm2 = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)
    batchm2.restore('batch_list_process.pkl')

    padding_states, padding_action, max_length, seq_lengths = \
        batchm.padding_states, batchm.padding_action, batchm.max_length, batchm.seq_lengths

    padding_states2, padding_action2, max_length2, seq_lengths2 = \
        batchm2.padding_states, batchm2.padding_action, batchm2.max_length, batchm2.seq_lengths

    print('训练集加载完毕')

    for i in range(batchm.batch_size):
        seq_length = seq_lengths[i]
        action_target = padding_action[i][:seq_length]
        time = np.array(range(seq_length))

        plt.plot(time, action_target[:, 0], 'r')

        seq_length = seq_lengths2[i]
        action_target = padding_action2[i][:seq_length]
        time = np.array(range(seq_length))

        plt.plot(time, action_target[:, 0], 'b--')

        plt.show()
        plt.clf()


def show_state():
    batchm = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)
    # batchm.restore()
    batchm.restore('batch_list_demo.pkl')

    padding_states, padding_action, max_length, seq_lengths = \
        batchm.padding_states, batchm.padding_action, batchm.max_length, batchm.seq_lengths
    print('训练集加载完毕')

    for i in range(batchm.batch_size):
        seq_length = seq_lengths[i]
        state = padding_states[i][:seq_length]
        time = np.array(range(seq_length))

        plt.plot(time, state[:, 2], 'r')
        plt.pause(2)
        plt.clf()


def process():
    batchm = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)
    batchm.restore()

    batchm_new = BATCH_MANAGE(s_dim=s_dim_rnn, a_dim=a_dim_rnn)

    upper = np.array([1, 1, 1])
    lower = np.array([-1, -1, -1])

    for i in range(batchm.batch_size):

        old_state_seq = batchm.list_states[i]
        new_state_seq = []

        old_action_seq = batchm.list_action[i]
        new_action_seq = []
        diff_action = np.array([0, 0, 0])

        for j in range(len(old_action_seq))[::-1]:
            old_action = np.array(old_action_seq[j])  # 逆序添加
            old_action = old_action + diff_action
            if (lower < old_action).all() and (old_action < upper).all():
                # print(old_action.tolist())
                new_action_seq.append(old_action.tolist())
                diff_action = np.array([0, 0, 0])
            else:
                clipped_action = np.clip(old_action, -1, 1)
                print(clipped_action.tolist())
                new_action_seq.append(clipped_action.tolist())
                diff_action = old_action - clipped_action
            new_state_seq.append(old_state_seq[j])

        while diff_action.tolist() != [0, 0, 0]:
            if (lower < old_action).all() and (old_action < upper).all():
                # print(old_action.tolist())
                new_action_seq.append(diff_action.tolist())
                diff_action = np.array([0, 0, 0])
            else:
                clipped_action = np.clip(diff_action, -1, 1)
                # print(clipped_action.tolist())
                new_action_seq.append(clipped_action.tolist())
                diff_action = diff_action - clipped_action
            new_state_seq.append(old_state_seq[0])

        new_state_seq = new_state_seq[::-1]
        new_action_seq = new_action_seq[::-1]
        batchm_new.add_seq(new_state_seq, new_action_seq)

    # print(batchm_new.list_action[0])
    # print(batchm_new.list_action[1])
    # print(batchm_new.list_action[2])

    batchm_new.save('batch_list_process.pkl')


if __name__ == '__main__':
    s_dim_rnn = 4
    a_dim_rnn = 3

    num_train = 8000

    # show_action()
    show_state()
    # process()
    # show_comparision()
