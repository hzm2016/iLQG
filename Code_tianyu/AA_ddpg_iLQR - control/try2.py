import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pickle
from iENV import Env_PeginHole


def model_pred(input_s, input_a):
    env = Env_PeginHole()
    s = env.decode_state(input_s)
    a = env.decode_action(input_a)

    s_ = np.zeros(13)
    s_[0] = s[0] + a[0] * a[3]
    s_[1] = s[1] + a[1] * a[4]
    s_[2] = s[2] + a[2] * a[5]
    s_[3] = s[3] + a[0]
    s_[4] = s[4] + a[1]
    s_[5] = s[5] + a[2]
    s_[6] = 0
    s_[7] = 0
    s_[8] = 0
    s_[9] = a[0] * a[3]
    s_[10] = a[1] * a[4]
    s_[11] = a[2] * a[5]
    s_[12] = s[12]
    s_ = env.code_state(s_)

    return s_


if __name__ == '__main__':
    # 主程序

    path_train_seqs_record = 'train_seqs_record.pkl'

    fr = open(path_train_seqs_record, 'rb')
    all_seqs_record = pickle.load(fr)
    fr.close()

    num_episode=len(all_seqs_record)

    for en in range(num_episode):

        seq = all_seqs_record[en]
        len_seq = len(seq)

        seq_sas_ = []
        seq_err = []
        for sn in range(len_seq):
            step_tuple = seq[sn]
            [s, a_noisy, _, s_, _] = step_tuple
            vec_sas_ = s.tolist() + a_noisy.tolist() + s_.tolist()
            dim_s = len(s.tolist())
            dim_a = len(a_noisy.tolist())
            seq_sas_.append(vec_sas_)

            pred_err = s_ - model_pred(s, a_noisy)
            seq_err.append(pred_err)

        mean_sas_ = np.mean(seq_sas_, axis=0)
        cov_sas_ = np.cov(np.array(seq_sas_).T)

        mean_1 = mean_sas_[0:dim_s + dim_a]
        mean_2 = mean_sas_[dim_s + dim_a:]

        cov_11 = cov_sas_[0:dim_s + dim_a, 0:dim_s + dim_a]
        cov_22 = cov_sas_[dim_s + dim_a:, dim_s + dim_a:]
        cov_12 = cov_sas_[0:dim_s + dim_a, dim_s + dim_a:]
        cov_21 = cov_sas_[dim_s + dim_a:, 0:dim_s + dim_a]

        # s = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # a = np.array([0, 0, 0, 0, 0, 0])

        index = 19

        s = np.array(seq_sas_[index][0:dim_s])
        a = np.array(seq_sas_[index][dim_s:dim_s + dim_a])

        s_ = np.array(seq_sas_[index + 1][0:dim_s])

        x1 = s.tolist() + a.tolist()
        x1 = np.array(x1)

        mean_est = mean_2 + np.matmul(np.matmul(cov_21, np.linalg.inv(cov_11)), (x1 - mean_1))
        cov_est = cov_22 - np.matmul(np.matmul(cov_21, np.linalg.inv(cov_11)), cov_12)

        list_err_mat = []
        for sn in range(len(seq_err)):
            list_err_mat.append(np.matmul(np.mat(seq_err[sn]).T, np.mat(seq_err[sn])))
        cov_err = np.mean(list_err_mat, axis=0)

        # print(list_mat)

        # print(list_mat)

        # print(cov_est)
    # k=range(0, -1, -1)
    #
    # print(k)

    for m in range(0, -1, -1):
        print(m)



