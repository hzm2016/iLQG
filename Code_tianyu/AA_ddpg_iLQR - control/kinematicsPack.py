import numpy as np

L1 = 0.42
L2 = 0.38
L3 = 0.16 + 0.088

y_hole_bottom = -0.0365


def cal_tcp(q):
    # 正运动学。计算TCP轨迹
    # 输入单位 deg 输出单位 m rad

    q2 = q[0] * np.pi / 180
    q4 = q[1] * np.pi / 180
    q6 = q[2] * np.pi / 180

    cx = L1 * np.sin(q2) + L2 * np.sin(q2 + q4) + L3 * np.sin(q2 + q4 + q6)
    cy = L1 * np.cos(q2) + L2 * np.cos(q2 + q4) + L3 * np.cos(q2 + q4 + q6)
    cw = np.pi / 2 - (q2 + q4 + q6)

    pos_cartesian = np.array([cx, cy, cw])

    return pos_cartesian


def cal_J246(c_pos):
    # 由笛卡尔坐标计算关节角

    cx = c_pos[0]
    cy = c_pos[1]
    cw = c_pos[2]

    sum246 = np.pi / 2 - cw
    wx = cx - L3 * np.sin(sum246)
    wy = cy - L3 * np.cos(sum246)
    Lh_2 = wx * wx + wy * wy

    Lh = np.sqrt(Lh_2)

    assert Lh <= L1 + L2, str(c_pos) + str(sum246) + str(wx) + str(wy) + str(Lh_2) + str(Lh)

    L1_2 = L1 * L1
    L2_2 = L2 * L2

    cc = (L1_2 + L2_2 - Lh_2) / (2 * L1 * L2)

    if cc > 1:
        cc = 1

    if cc < -1:
        cc = -1

    aLh = np.arccos(cc)
    ss = L2 / Lh * np.sin(aLh)

    if ss > 1:
        ss = 1

    if ss < -1:
        ss = -1

    aL2 = np.arcsin(ss)

    q4 = np.pi - aLh

    azw = np.arctan(wx / wy)

    if azw < 0:
        azw = np.pi + azw

    q2 = azw - aL2
    q6 = sum246 - q2 - q4

    q2 = q2 * 180 / np.pi
    q4 = q4 * 180 / np.pi
    q6 = q6 * 180 / np.pi

    assert 0 < q2 < 90, str(c_pos) + ', ' + str(sum246) + ', ' + str(wx) + ', ' + str(wy) + ', ' + str(
        Lh_2) + ', ' + str(Lh) + ', ' + str(cc) + ', ' + str(aLh) + ', ' \
                        + str(ss) + ', ' + str(aL2) + ', ' + str(q4) + ', ' + str(azw)

    q = np.array([q2, q4, q6])

    return q


def cal_Fext(rsd, q):
    # 根据外力矩计算外力

    q2 = q[0] * np.pi / 180
    q4 = q[1] * np.pi / 180
    q6 = q[2] * np.pi / 180

    rsd_2 = rsd[0]
    rsd_4 = rsd[1]
    rsd_6 = rsd[2]

    itj = np.ones([3, 3], dtype=np.float64)

    itj[0, 0] = np.sin(q2 + q4) / (np.cos(q2) * np.sin(q2 + q4) * L1 - np.sin(q2) * np.cos(q2 + q4) * L1)
    itj[0, 1] = -(np.sin(q2) * L1 + np.sin(q2 + q4) * L2) / (
            np.cos(q2) * np.sin(q2 + q4) * L1 * L2 - np.sin(q2) * np.cos(q2 + q4) * L1 * L2)
    itj[0, 2] = np.sin(q2) / (np.cos(q2) * np.sin(q2 + q4) * L2 - np.sin(q2) * np.cos(q2 + q4) * L2)
    itj[1, 0] = np.cos(q2 + q4) / (np.cos(q2) * np.sin(q2 + q4) * L1 - np.sin(q2) * np.cos(q2 + q4) * L1)
    itj[1, 1] = -(np.cos(q2) * L1 + np.cos(q2 + q4) * L2) / (
            np.cos(q2) * np.sin(q2 + q4) * L1 * L2 - np.sin(q2) * np.cos(q2 + q4) * L1 * L2)
    itj[1, 2] = np.cos(q2) / (np.cos(q2) * np.sin(q2 + q4) * L2 - np.sin(q2) * np.cos(q2 + q4) * L2)
    itj[2, 0] = (np.cos(q2 + q4 + q6) * np.sin(q2 + q4) * L3 - np.sin(q2 + q4 + q6) * np.cos(q2 + q4) * L3) / (
            np.cos(q2) * np.sin(q2 + q4) * L1 - np.sin(q2) * np.cos(q2 + q4) * L1)
    itj[2, 1] = (np.cos(q2) * np.sin(q2 + q4 + q6) * L1 * L3 - np.cos(q2 + q4 + q6) * np.sin(q2) * L1 * L3 - np.cos(
        q2 + q4 + q6) * np.sin(q2 + q4) * L2 * L3 + np.sin(q2 + q4 + q6) * np.cos(q2 + q4) * L2 * L3) / (
                        np.cos(q2) * np.sin(q2 + q4) * L1 * L2 - np.sin(q2) * np.cos(q2 + q4) * L1 * L2)
    itj[2, 2] = -(np.cos(q2) * np.sin(q2 + q4 + q6) * L3 - np.cos(q2 + q4 + q6) * np.sin(q2) * L3 + np.cos(q2) * np.sin(
        q2 + q4) * L2 - np.sin(q2) * np.cos(q2 + q4) * L2) / (
                        np.cos(q2) * np.sin(q2 + q4) * L2 - np.sin(q2) * np.cos(q2 + q4) * L2)

    fext = itj * np.mat([[rsd_2], [rsd_4], [rsd_6]])

    f_x = fext[0, 0]
    f_y = fext[1, 0]
    f_w = fext[2, 0]

    force_cartesian = np.array([f_x, f_y, f_w])

    return force_cartesian
