import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
import time
plt.rcParams['font.family'] = ['FreeSans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        d : float
            Half of the track width (m) of TurtleBot3 Burger.
        r : float
            Wheel radius (m) of the TurtleBot3 Burger.
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][0] is timestamp (s),
            u[i][1] is left wheel rotational speed (rad/s), and
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is timestamp (s),
            x[i][1] is bearing (rad),
            x[i][2] is translational position in x (m),
            x[i][3] is translational position in y (m),
            x[i][4] is left wheel rotational position (rad), and
            x[i][5] is right wheel rotational position (rad).
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][0] is timestamp (s),
            y[i][1] is translational position in x (m) when freeze_bearing:=true,
            y[i][1] is distance to the landmark (m) when freeze_bearing:=false,
            y[i][2] is translational position in y (m) when freeze_bearing:=true, and
            y[i][2] is relative bearing (rad) w.r.t. the landmark when
            freeze_bearing:=false.
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
        sub_u : rospy.Subscriber
            ROS subscriber for system inputs.
        sub_x : rospy.Subscriber
            ROS subscriber for system states.
        sub_y : rospy.Subscriber
            ROS subscriber for system outputs.
        tmr_update : rospy.Timer
            ROS Timer for periodically invoking the estimator's update method.

    Notes
    ----------
        The frozen bearing is pi/4 and the landmark is positioned at (0.5, 0.5).
    """
    # noinspection PyTypeChecker
    def __init__(self):
        self.d = 0.08
        self.r = 0.033
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.errrors = []
        self.comptimes = []
        self.dt = 0.1
        self.fig, self.axd = plt.subplot_mosaic(
            [['xy', 'phi'],
             ['xy', 'x'],
             ['xy', 'y'],
             ['xy', 'thl'],
             ['xy', 'thr']], figsize=(20.0, 10.0))
        self.ln_xy, = self.axd['xy'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xy_hat, = self.axd['xy'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_y, = self.axd['y'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_y_hat, = self.axd['y'].plot([], 'o-c', label='Estimated')
        self.ln_thl, = self.axd['thl'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thl_hat, = self.axd['thl'].plot([], 'o-c', label='Estimated')
        self.ln_thr, = self.axd['thr'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thr_hat, = self.axd['thr'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'
        self.sub_u = rospy.Subscriber('u', Float32MultiArray, self.callback_u)
        self.sub_x = rospy.Subscriber('x', Float32MultiArray, self.callback_x)
        self.sub_y = rospy.Subscriber('y', Float32MultiArray, self.callback_y)
        self.tmr_update = rospy.Timer(rospy.Duration(self.dt), self.update)

    def callback_u(self, msg):
        self.u.append(msg.data)

    def callback_x(self, msg):
        self.x.append(msg.data)
        if len(self.x_hat) == 0:
            self.x_hat.append(msg.data)

    def callback_y(self, msg):
        self.y.append(msg.data)

    def update(self, _):
        raise NotImplementedError
    
    def get_estimation_error(self):
        return np.sum(self.errrors)
    
    def print_comp_times(self):
        print(f'Average computation time: {np.mean(self.comptimes) * 1000} ms')

        # Plot computation times in milliseconds
        plt.figure(figsize=(10, 5))
        plt.plot([t * 1000 for t in self.comptimes], marker='o', linestyle='-', color='b', label='Computational time (ms)')
        plt.axhline(np.mean(self.comptimes) * 1000, color='r', linestyle='--', label=f'Average time: {(np.mean(self.comptimes) * 1000):.2f} ms')
        plt.title(f'Per-step computational running time using {self.canvas_title} on the TurtleBot')
        plt.xlabel('Update step')
        plt.ylabel('Computational time (ms)')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_init(self):
        self.axd['xy'].set_title(self.canvas_title)
        self.axd['xy'].set_xlabel('x (m)')
        self.axd['xy'].set_ylabel('y (m)')
        self.axd['xy'].set_aspect('equal', adjustable='box')
        self.axd['xy'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].legend()
        self.axd['y'].set_ylabel('y (m)')
        self.axd['y'].legend()
        self.axd['thl'].set_ylabel('theta L (rad)')
        self.axd['thl'].legend()
        self.axd['thr'].set_ylabel('theta R (rad)')
        self.axd['thr'].set_xlabel('Time (s)')
        self.axd['thr'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xyline(self.ln_xy, self.x)
        self.plot_xyline(self.ln_xy_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_yline(self.ln_y, self.x)
        self.plot_yline(self.ln_y_hat, self.x_hat)
        self.plot_thlline(self.ln_thl, self.x)
        self.plot_thlline(self.ln_thl_hat, self.x_hat)
        self.plot_thrline(self.ln_thr, self.x)
        self.plot_thrline(self.ln_thr_hat, self.x_hat)

    def plot_xyline(self, ln, data):
        if len(data):
            x = [d[2] for d in data]
            y = [d[3] for d in data]
            ln.set_data(x, y)
            self.resize_lim(self.axd['xy'], x, y)

    def plot_philine(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            phi = [d[1] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            x = [d[2] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_yline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            y = [d[3] for d in data]
            ln.set_data(t, y)
            self.resize_lim(self.axd['y'], t, y)

    def plot_thlline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thl = [d[4] for d in data]
            ln.set_data(t, thl)
            self.resize_lim(self.axd['thl'], t, thl)

    def plot_thrline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thr = [d[5] for d in data]
            ln.set_data(t, thr)
            self.resize_lim(self.axd['thr'], t, thr)

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
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=oracle_observer \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
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
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=dead_reckoning \
            noise_injection:=true \
            freeze_bearing:=false
    For debugging, you can simulate a noise-free unicycle model by setting
    noise_injection:=false.
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Dead Reckoning'

    def update(self, i):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation
            start_time = time.time()
            t_current = self.x[-1][0]
            x_hat_prev = self.x_hat[-1]

            u_k = None
            for input_data in self.u:
                if input_data[0] <= t_current:
                    u_k = input_data
                else:
                    break

            if u_k is not None:
                w_l = u_k[1]
                w_r = u_k[2]

                phi_prev = x_hat_prev[1]
                x_prev   = x_hat_prev[2]
                y_prev   = x_hat_prev[3]
                thl_prev = x_hat_prev[4]
                thr_prev = x_hat_prev[5]

                phi_new = phi_prev + (self.r / (2 * self.d)) * (w_r - w_l) * self.dt
                x_new   = x_prev + (self.r / 2) * (w_r + w_l) * np.cos(phi_prev) * self.dt
                y_new   = y_prev + (self.r / 2) * (w_r + w_l) * np.sin(phi_prev) * self.dt
                thl_new = thl_prev + w_l * self.dt
                thr_new = thr_prev + w_r * self.dt

                # Measure the difference between the estimated and true position
                error = np.linalg.norm(np.array(self.x[-1][2:4]) - np.array([x_new, y_new]))
                self.errrors.append(error)

                self.x_hat.append([t_current, phi_new, x_new, y_new, thl_new, thr_new])

                end_time = time.time()
                self.comptimes.append(end_time - start_time)

class KalmanFilter(Estimator):
    """Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    linear unicycle model at the default bearing of pi/4. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive Kalman filter update rule.

    Attributes:
    ----------
        phid : float
            Default bearing of the turtlebot fixed at pi / 4.

    Example
    ----------
    To run the Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=kalman_filter \
            noise_injection:=true \
            freeze_bearing:=true
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Kalman Filter'
        self.phid = np.pi / 4

        self.A = np.eye(3)
        alpha = self.r / (2 * self.d)
        beta = self.r / 2
        self.B = np.array([
            [-alpha * self.dt,  alpha * self.dt],
            [ beta * np.cos(self.phid) * self.dt, beta * np.cos(self.phid) * self.dt],
            [ beta * np.sin(self.phid) * self.dt, beta * np.sin(self.phid) * self.dt]
        ])
        self.C = np.array([
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        self.Q = np.diag([1e-2, 1e-2, 1e-2])
        self.R = np.diag([1e-2, 1e-2])
        self.P = np.eye(3)

    # noinspection DuplicatedCode
    # noinspection PyPep8Naming
    def update(self, _):
        start_time = time.time()

        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            t_current = self.x[-1][0]
            x_hat_prev = self.x_hat[-1]
            X_prev = np.array([x_hat_prev[1], x_hat_prev[2], x_hat_prev[3]])

            u_k = None
            for input_data in self.u:
                if input_data[0] <= t_current:
                    u_k = input_data
                else:
                    break

            if u_k is not None:
                u_vec = np.array([u_k[1], u_k[2]])
                
                X_pred = np.dot(self.A, X_prev) + np.dot(self.B, u_vec)
                P_pred = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
                y_meas = self.y[-1]
                z = np.array([y_meas[1], y_meas[2]])
                
                y_tilde = z - np.dot(self.C, X_pred)
                S = np.dot(np.dot(self.C, P_pred), self.C.T) + self.R
                K = np.dot(np.dot(P_pred, self.C.T), np.linalg.inv(S))
                X_new = X_pred + np.dot(K, y_tilde)
                P_new = np.dot((np.eye(3) - np.dot(K, self.C)), P_pred)
                self.P = P_new

                thl_new = x_hat_prev[4] + u_k[1] * self.dt
                thr_new = x_hat_prev[5] + u_k[2] * self.dt

                # Measure the difference between the estimated and true position
                error = np.linalg.norm(np.array(self.x[-1][2:4]) - np.array([X_new[1], X_new[2]]))
                self.errrors.append(error)

                new_state = [t_current, X_new[0], X_new[1], X_new[2], thl_new, thr_new]
                self.x_hat.append(new_state)

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

    Example
    ----------
    To run the extended Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=extended_kalman_filter \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Extended Kalman Filter'
        self.landmark = (0.5, 0.5)
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.

    # noinspection DuplicatedCode
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            raise NotImplementedError

