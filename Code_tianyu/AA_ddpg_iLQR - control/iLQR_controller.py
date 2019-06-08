import warnings
import numpy as np
from scipy.optimize import approx_fprime
import numdifftools as nd


class iLQR(object):
    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = np.zeros((N, dynamics.action_size))
        self._K = np.zeros((N, dynamics.action_size, dynamics.state_size))

        super(iLQR, self).__init__()

    def fit(self, x_init, u_init, n_iterations=1, tol=1e-6, on_iteration=None):
        """Computes the optimal controls.

        Args:
            x_init: Initial state [state_size].
            u_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1 ** (-np.arange(10) ** 2)

        us = u_init.copy()
        xs = self._forward_rollout(x_init, us)
        k = self._k
        K = self._K

        J_opt = self._trajectory_cost(xs, us)

        converged = False
        for iteration in range(n_iterations):
            accepted = False

            try:
                k, K = self._backward_pass(xs, us)

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("  【iLQR】exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us

    def _control(self, xs, us, k, K, alpha=0.000001):
        """Applies the controls for a given trajectory. 1.0

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        for i in range(self.N):
            # Eq (12).
            # Applying alpha only on k[i] as in the paper for some reason
            # doesn't converge.
            us_new[i] = us[i] + alpha * (k[i] + K[i].dot(xs_new[i] - xs[i]))

            # Eq (8c).
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i])

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us, range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, terminal=True)

    def _forward_rollout(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            State path [N+1, state_size].
        """

        xs = np.array([x0])
        for i in range(self.N):
            x_new = self.dynamics.f(xs[-1], us[i])
            xs = np.append(xs, [x_new], axis=0)

        return xs

    def _backward_pass(self, xs, us):
        """Computes the feedforward and feedback gains k and K.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = self.cost.l_x(xs[-1], None, terminal=True)
        V_xx = self.cost.l_xx(xs[-1], None, terminal=True)

        k = np.zeros_like(self._k)
        K = np.zeros_like(self._K)

        # 当N=0时，迭代一次
        for i in range(self.N - 1, -1, -1):
            x = xs[i]
            u = us[i]

            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(x, u, V_x, V_xx)
            Q_uu_inv = np.linalg.pinv(Q_uu)

            # Eq (6).
            k[i] = -Q_uu_inv.dot(Q_u)
            K[i] = -Q_uu_inv.dot(Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self, x, u, V_x, V_xx):
        """Computes second order expansion.

        Args:
            x: State [state_size].
            u: Control [action_size].
            V_x: d/dx of the value function at the next time step [state_size].
            V_xx: d^2/dx^2 of the value function at the next time step
                [state_size, state_size].
            i: Current time step.

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        f_x = self.dynamics.f_x(x, u)
        f_u = self.dynamics.f_u(x, u)

        l_x = self.cost.l_x(x, u)
        l_u = self.cost.l_u(x, u)
        l_xx = self.cost.l_xx(x, u)
        l_ux = self.cost.l_ux(x, u)
        l_uu = self.cost.l_uu(x, u)

        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        if self._use_hessians:
            f_xx = self.dynamics.f_xx(x, u)
            f_ux = self.dynamics.f_ux(x, u)
            f_uu = self.dynamics.f_uu(x, u)

            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu


class fd_Cost(object):
    """Finite difference approximated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self,
                 l,
                 l_terminal,
                 state_size,
                 action_size,
                 x_eps=None,
                 u_eps=None):
        """Constructs an FiniteDiffCost.

        Args:
            l: Instantaneous cost function to approximate.
                Signature: (x, u, i) -> scalar.
            l_terminal: Terminal cost function to approximate.
                Signature: (x, i) -> scalar.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self._l = l
        self._l_terminal = l_terminal
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        super(fd_Cost, self).__init__()

    def l(self, x, u, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x)

        return self._l(x, u)

    def l_x(self, x, u, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return approx_fprime(x, lambda x: self._l_terminal(x),
                                 self._x_eps)

        return approx_fprime(x, lambda x: self._l(x, u), self._x_eps)

    def l_u(self, x, u, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return approx_fprime(u, lambda u: self._l(x, u), self._u_eps)

    def l_xx(self, x, u, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_x(x, u, terminal)[m], eps)
            for m in range(self._state_size)
        ])
        return Q

    def l_ux(self, x, u, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_u(x, u)[m], eps)
            for m in range(self._action_size)
        ])
        return Q

    def l_uu(self, x, u, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        eps = self._u_eps_hess
        Q = np.vstack([
            approx_fprime(u, lambda u: self.l_u(x, u)[m], eps)
            for m in range(self._action_size)
        ])
        return Q


class myCost(object):
    """Finite difference approximated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self,
                 l,
                 l_terminal,
                 state_size,
                 action_size,
                 x_eps=None,
                 u_eps=None):
        """Constructs an FiniteDiffCost.

        Args:
            l: Instantaneous cost function to approximate.
                Signature: (x, u, i) -> scalar.
            l_terminal: Terminal cost function to approximate.
                Signature: (x, i) -> scalar.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self._l = l
        self._l_terminal = l_terminal
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

    def l(self, x, u, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x)

        return self._l(x, u)

    def l_x(self, x, u, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return approx_fprime(x, lambda x: self._l_terminal(x),
                                 self._x_eps)

        return approx_fprime(x, lambda x: self._l(x, u), self._x_eps)

    def l_u(self, x, u, terminal=False):

        def fun_u(u, x):
            return self._l(x, u)

        Jfun = nd.Jacobian(fun_u)

        return Jfun(u, x)

    def l_xx(self, x, u, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_x(x, u, terminal)[m], eps)
            for m in range(self._state_size)
        ])
        return Q

    def l_ux(self, x, u, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_u(x, u)[m], eps)
            for m in range(self._action_size)
        ])
        return Q

    def l_uu(self, x, u, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        eps = self._u_eps_hess
        Q = np.vstack([
            approx_fprime(u, lambda u: self.l_u(x, u)[m], eps)
            for m in range(self._action_size)
        ])
        return Q


class fd_Dynamics(object):
    """Finite difference approximated Dynamics Model."""

    def __init__(self, f, state_size, action_size, x_eps=None, u_eps=None):
        """Constructs an FiniteDiffDynamics model.

        Args:
            f: Function to approximate. Signature: (x, u, i) -> x.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self._f = f
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        super(fd_Dynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return True

    def f(self, x, u):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x, u)

    def f_x(self, x, u):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        J = np.vstack([
            approx_fprime(x, lambda x: self._f(x, u)[m], self._x_eps)
            for m in range(self._state_size)
        ])
        return J

    def f_u(self, x, u):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        J = np.vstack([
            approx_fprime(u, lambda u: self._f(x, u)[m], self._u_eps)
            for m in range(self._state_size)
        ])
        return J

    def f_xx(self, x, u):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_x(x, u)[m, n], eps)
                for n in range(self._state_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q

    def f_ux(self, x, u):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_u(x, u)[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q

    def f_uu(self, x, u):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        eps = self._u_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(u, lambda u: self.f_u(x, u)[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q
