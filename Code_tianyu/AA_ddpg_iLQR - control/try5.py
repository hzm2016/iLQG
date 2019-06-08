import numpy as np
from iLQR_controller import iLQR, fd_Cost, fd_Dynamics

state_size = 2  # [position, velocity]
action_size = 1  # [force]

dt = 0.01  # Discrete time-step in seconds.
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.

Q = 100 * np.eye(state_size)
R = 0.01 * np.eye(action_size)

# This is optional if you want your cost to be computed differently at a
# terminal state.
Q_terminal = np.array([[100.0, 0.0], [0.0, 0.1]])

# State goal is set to a position of 1 m with no velocity.
x_goal = np.array([1.0, 0.0])


def f(x, u):
    """Dynamics model function.

    Args:
        x: State vector [state_size].
        u: Control vector [action_size].
        i: Current time step.

    Returns:
        Next state vector [state_size].
    """
    [x, x_dot] = x
    [F] = u

    # Acceleration.
    x_dot_dot = np.sin(x_dot) * (1 - alpha * dt / m) + F * dt / m

    return np.array([
        x + x_dot * dt,
        x_dot + x_dot_dot * dt,
    ])


def l(x, u):
    """Instantaneous cost function.

    Args:
        x: State vector [state_size].
        u: Control vector [action_size].
        i: Current time step.

    Returns:
        Instantaneous cost [scalar].
    """
    x_diff = x - x_goal
    return x_diff.T.dot(Q).dot(x_diff) + u.T.dot(R).dot(u)


def l_terminal(x):
    """Terminal cost function.

    Args:
        x: State vector [state_size].
        i: Current time step.

    Returns:
        Terminal cost [scalar].
    """
    x_diff = x - x_goal
    return x_diff.T.dot(Q_terminal).dot(x_diff)


# NOTE: Unlike with AutoDiffDynamics, this is instantaneous, but will not be
# as accurate.
if __name__ == '__main__':
    dynamics = fd_Dynamics(f, state_size, action_size)
    cost = fd_Cost(l, l_terminal, state_size, action_size)

    N = 1  # Number of time-steps in trajectory.
    x0 = np.array([0.0, -0.1])  # Initial state.
    us_init = np.random.uniform(-1, 1, (N, 1))  # Random initial action path.

    print(x0)
    print(us_init)

    ilqr = iLQR(dynamics, cost, N)


    xs, us = ilqr.fit(x0, us_init)

    # print(xs)
    print(us)
