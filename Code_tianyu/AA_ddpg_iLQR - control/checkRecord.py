import matplotlib.pyplot as plt
import pickle
import numpy as np

path_train_seqs_record = 'myRecord/train_seqs_record.pkl'
path_eval_seqs_record = 'myRecord/eval_seqs_record.pkl'

if __name__ == '__main__':
    # 主程序
    fr = open(path_train_seqs_record, 'rb')
    all_seqs_record = pickle.load(fr)
    fr.close()

    num_episodes = len(all_seqs_record)

    for i in range(num_episodes):
        seq = all_seqs_record[i]
        len_seq = len(seq)
        final_step = seq[len_seq - 1]
        [s, a_noisy, r, s_, s_terminal] = final_step
        [_, _, r, _, _] = np.sum(seq, 0)
        process = s[12]
        print('episode %3d' % i,
              ' len %3d' % len_seq,
              ' total reward %6.3f' % r,
              ' process %6.3f' % process,
              ' final_state %2d' % s_terminal)
