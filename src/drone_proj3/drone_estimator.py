import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14
import time

class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][1] is the thrust of the quadrotor
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is translational position in x (m),
            x[i][1] is translational position in z (m),
            x[i][2] is the bearing (rad) of the quadrotor
            x[i][3] is translational velocity in x (m/s),
            x[i][4] is translational velocity in z (m/s),
            x[i][5] is angular velocity (rad/s),
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][1] is distance to the landmark (m)
            y[i][2] is relative bearing (rad) w.r.t. the landmark
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.

    Notes
    ----------
        The landmark is positioned at (0, 5, 5).
    """
    # noinspection PyTypeChecker
    def __init__(self, is_noisy=False):
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.errrors = []
        self.comptimes = []
        self.t = []
        self.fig, self.axd = plt.subplot_mosaic(
            [['xz', 'phi'],
             ['xz', 'x'],
             ['xz', 'z']], figsize=(20.0, 10.0))
        self.ln_xz, = self.axd['xz'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xz_hat, = self.axd['xz'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_z, = self.axd['z'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_z_hat, = self.axd['z'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'

        # Defined in dynamics.py for the dynamics model
        # m is the mass and J is the moment of inertia of the quadrotor 
        self.gr = 9.81 
        self.m = 0.92
        self.J = 0.0023
        # These are the X, Y, Z coordinates of the landmark
        self.landmark = (0, 5, 5)

        # This is a (N,12) where it's time, x, u, then y_obs 
        if is_noisy:
            with open('noisy_data.npy', 'rb') as f:
                self.data = np.load(f)
        else:
            with open('data.npy', 'rb') as f:
                self.data = np.load(f)

        self.dt = self.data[-1][0]/self.data.shape[0]


    def run(self):
        for i, data in enumerate(self.data):
            self.t.append(np.array(data[0]))
            self.x.append(np.array(data[1:7]))
            self.u.append(np.array(data[7:9]))
            self.y.append(np.array(data[9:12]))
            if i == 0:
                self.x_hat.append(self.x[-1])
            else:
                self.update(i)
        return self.x_hat

    def update(self, _):
        raise NotImplementedError
    
    def get_estimation_error(self):
        return np.sum(self.errrors)
    
    def print_comp_times(self):
        for timestamp in self.comptimes:
            print(f'timestep: {timestamp}')
        print(f'Average computation time: {np.mean(self.comptimes) * 1000} ms')

        # Plot computation times in milliseconds
        plt.figure(figsize=(10, 5))
        plt.plot([t * 1000 for t in self.comptimes][1:], marker='o', linestyle='-', color='b', label='Computational time (ms)')
        plt.axhline(np.mean(self.comptimes) * 1000, color='r', linestyle='--', label=f'Average time: {(np.mean(self.comptimes) * 1000):.2f} ms')
        plt.title(f'Per-step computational running time using {self.canvas_title} on the Quadrotor')
        plt.xlabel('Update step')
        plt.ylabel('Computational time (ms)')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_init(self):
        self.axd['xz'].set_title(self.canvas_title)
        self.axd['xz'].set_xlabel('x (m)')
        self.axd['xz'].set_ylabel('z (m)')
        self.axd['xz'].set_aspect('equal', adjustable='box')
        self.axd['xz'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].set_xlabel('t (s)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].set_xlabel('t (s)')
        self.axd['x'].legend()
        self.axd['z'].set_ylabel('z (m)')
        self.axd['z'].set_xlabel('t (s)')
        self.axd['z'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xzline(self.ln_xz, self.x)
        self.plot_xzline(self.ln_xz_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_zline(self.ln_z, self.x)
        self.plot_zline(self.ln_z_hat, self.x_hat)

    def plot_xzline(self, ln, data):
        if len(data):
            x = [d[0] for d in data]
            z = [d[1] for d in data]
            ln.set_data(x, z)
            self.resize_lim(self.axd['xz'], x, z)

    def plot_philine(self, ln, data):
        if len(data):
            t = self.t
            phi = [d[2] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = self.t
            x = [d[0] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_zline(self, ln, data):
        if len(data):
            t = self.t
            z = [d[1] for d in data]
            ln.set_data(t, z)
            self.resize_lim(self.axd['z'], t, z)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])

class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ python drone_estimator_node.py --estimator oracle_observer
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ python drone_estimator_node.py --estimator dead_reckoning
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Dead Reckoning'

    def update(self, i):
        if len(self.x_hat) > 0:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation
            start_time = time.time()

            state_current = self.x_hat[-1]
            u_current = self.u[i]

            x = state_current[0]
            z = state_current[1]
            phi = state_current[2]
            x_dot = state_current[3]
            z_dot = state_current[4]
            phi_dot = state_current[5]        

            u1 = u_current[0]
            u2 = u_current[1]

            x_next   = x + x_dot * self.dt
            z_next   = z + z_dot * self.dt
            phi_next = phi + phi_dot * self.dt

            x_dot_dot      = -u1 * np.sin(phi) * (1 / self.m)
            z_dot_dot      = (u1 / self.m) * np.cos(phi) - self.gr
            z_dot_dot      = -self.gr + u1 * np.cos(phi) * (1 / self.m)
            phi_dot_dot    = u2 * (1 / self.J)

            x_dot_next   = x_dot + x_dot_dot * self.dt
            z_dot_next   = z_dot + z_dot_dot * self.dt
            phi_dot_next = phi_dot + phi_dot_dot * self.dt

            new_state = np.array([x_next, z_next, phi_next, x_dot_next, z_dot_next, phi_dot_next])
            self.x_hat.append(new_state)

            # Measure the difference between the estimated and true position
            error = np.linalg.norm(np.array(self.x[-1][0:2]) - np.array([x_next, z_next]))
            self.errrors.append(error)

            end_time = time.time()
            self.comptimes.append(end_time - start_time)

# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.
            landmark[2] is the z coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ python drone_estimator_node.py --estimator extended_kalman_filter
    """
    def __init__(self, is_noisy=True):
        super().__init__(is_noisy)
        self.canvas_title = 'Extended Kalman Filter'
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.
        self.A = None
        # self.B = None
        self.C = None
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(2) * 10
        self.P = np.eye(6) * 0.01

    # noinspection DuplicatedCode
    def update(self, i):
        if len(self.x_hat) > 0: #and self.x_hat[-1][0] < self.x[-1][0]:
            
            state = self.g(i)

            A_next = self.approx_A(state, self.u, i)

            self.P = A_next @ self.P @ A_next.T + self.Q

            C_next = self.approx_C(state)

            K = self.P @ C_next.T @ np.linalg.inv(C_next @ self.P @ C_next.T + self.R)

            state_next = state + K @ (self.y[i] - self.h(state))

            self.P = (np.eye(6) - K @ C_next) @ self.P

            self.x_hat.append(state_next)



    def g(self, i):
        state_current = self.x_hat[-1]
        u_current = self.u[i]

        x = state_current[0]
        z = state_current[1]
        phi = state_current[2]
        x_dot = state_current[3]
        z_dot = state_current[4]
        phi_dot = state_current[5]        

        u1 = u_current[0]
        u2 = u_current[1]

        x_next   = x + x_dot * self.dt
        z_next   = z + z_dot * self.dt
        phi_next = phi + phi_dot * self.dt

        x_dot_dot      = -u1 * np.sin(phi) * (1 / self.m)
        z_dot_dot      = -self.gr + u1 * np.cos(phi) * (1 / self.m)
        phi_dot_dot    = u2 * (1 / self.J)

        x_dot_next   = x_dot + x_dot_dot * self.dt
        z_dot_next   = z_dot + z_dot_dot * self.dt
        phi_dot_next = phi_dot + phi_dot_dot * self.dt

        return np.array([x_next, z_next, phi_next, x_dot_next, z_dot_next, phi_dot_next])

    def h(self, state):
        x = state[0]
        z = state[1]
        phi = state[2]
        
        lx = self.landmark[0]
        lz = self.landmark[2]
    
        r = np.sqrt((lx - x)**2 + (lz - z)**2)
        
        return np.array([r, phi])

    def approx_A(self, state, u, i):
        u1 = u[i][0]
        phi = state[2]

        a34 = -u1 * (1 / self.m) * np.sin(phi) * self.dt
        a35 = -u1 * (1 / self.m) * np.cos(phi) * self.dt

        A = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, a34, 1, 0, 0],
            [0, 0, a35, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        return A
    
    def approx_C(self, state):
        x = state[0]
        z = state[1]

        lx = self.landmark[0]
        lz = self.landmark[2]

        r = np.sqrt((x - lx)**2 + (z - lz)**2)

        c11 = (x - lx) * (1 / r)
        c12 = (z - lz) * (1 / r)

        C = np.array([
            [c11, c12, 0, 0, 0, 0],
            [0, 0, 1, 0 , 0 , 0],
        ])

        return C