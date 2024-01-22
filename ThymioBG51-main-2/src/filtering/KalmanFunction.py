import numpy as np
import matplotlib.pyplot as plt
import math
from tdmclient import aw
import time
import datetime

def motors(l_speed=500, r_speed=500):
    """
    Return the right syntax to set the values for the motor
    """

    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }


def get_wheel_speeds(node):
    """
    Return the values of the wheel speed sensors
    """

    aw(node.wait_for_variables())

    left = node['motor.left.speed']
    right = node['motor.right.speed']

    return [left, right]


def get_ground(node):
    """
    Return the values of the two ground sensors to check if the robot is being kidnapped
    """

    aw(node.wait_for_variables())

    return list(node['prox.ground.reflected'])

# Definition of the Extended Kalamn Filter function


def EKF(z, x_1, P_1, dt, camera):
    """
    Compute the Extended Kalman Filter and return the estimated states and covariance
    """

    ### Definition of the constant matrices

    # Covariance matrix
    Q = np.array([[0.01, 0, 0, 0, 0],
                [0, 0.01, 0, 0, 0],
                [0, 0, 0.05, 0, 0],
                [0, 0, 0, 6, 0],
                [0, 0, 0, 0, 6]])

    # Measurement matrix if only the wheels sensor are relevant
    H = np.array([[0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1]])

    # Measurement matrix if the camera sees the robot
    H_cam = np.array([[1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])
    
    # Noise covariance
    R = np.array([[0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,0,0,6,0],
                  [0,0,0,0,6]])
    
    # Noise covariance if we have the camera
    R_cam = np.array([[0.01,0,0,0,0],
                  [0,0.01,0,0,0],
                  [0,0,0.01,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]])
    
    if camera is True:
        H = H_cam
        R = R_cam

    ### Update the matrix A

    # Defining the constants
    wheelbase = 90 # [mm]

    # Angle
    theta = x_1[2]

    # Wheel speeds
    Vleft = x_1[3]
    Vright = x_1[4]
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    A_1 = np.array([[0, 0, 0, cos_theta/2, cos_theta/2],
                    [0, 0, 0, sin_theta/2, sin_theta/2],
                    [0, 0, 0, -1/wheelbase, 1/wheelbase],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    
    # Compute the Jacobian
    F_1 = np.array([[0, 0, -1/2*(Vright+Vleft)*sin_theta, cos_theta/2, cos_theta/2],
                    [0, 0, 1/2*(Vright+Vleft)*cos_theta, sin_theta/2, sin_theta/2],
                    [0, 0, 0, -1/wheelbase, 1/wheelbase],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    ### Predict

    # Predicted state
    x = A_1 @ x_1

    # State coviariance estimate
    P = F_1@P_1 + F_1@F_1.T + Q

    ### Integrate

    x = x_1 + x*dt
    P = P_1 + P*dt

    ### Update

    # Calibration of the measurement
    if camera is False:
        CALIBRATION = 0.3375
        z = np.multiply(z,CALIBRATION)

    # Innovation (measurement residual)
    y = z - H @ x

    # Innovation covariance
    S = H @ P @ H.T + R

    # Near-optinal Kalman gain
    K = P @ H.T @ np.linalg.pinv(S)

    # Updated state estimate
    x = x + (K @ y)

    if camera is True:
        # Angle wrap
        if(x[2]>math.pi):
            x[2] = x[2]-2*math.pi
        if(x[2]<-math.pi):
            x[2] = x[2]+2*math.pi

    # Covariance estimate
    P = (np.eye(5) - K@H) @ P

    return x, P


def controller(node, goal, x_est, close, end, K):
        """
        P controller
        """

        # Compute the angle to the goal
        hyp = np.sqrt((goal[0]-x_est[0])**2 + (goal[1]-x_est[1])**2)

        if hyp>1e-1:
            cos_theta = (goal[0]-x_est[0])/(hyp)
            angle_controller = np.arccos(cos_theta)*np.sign(goal[1]-x_est[1])

        else:
            angle_controller = 0

        distance = np.sqrt((goal[0]-x_est[0])**2 + (goal[1]-x_est[1])**2)

        if angle_controller > 3.05:
            angle_controller = 2.5

        if angle_controller < -3.05:
            angle_controller = -2.5

        # Compute the error
        error = angle_controller - x_est[2]
        
        # Check if the Thymio is close from the last goal
        if distance < 15 and end is True:
            close = 0
            reached = True

        # Check if the Thymio is close from the current goal
        if distance < 15:
            reached = True
        else:
            reached = False

        # Compute the speed for each wheel
        motorl = int(150 - K*(error))*close
        motorr = int(150 + K*(error))*close

        aw(node.set_variables(motors(motorl, motorr)))

        return close, reached


def KalmanFilter(node, previous_time, x_1, P_1):
        
        z = np.array([0,0,0,0,0])

        lspeed, rspeed = get_wheel_speeds(node)

        # Obtain the wheel speed sensors measurement
        z[3] = lspeed
        z[4] = rspeed

        # Update the time counter between each iteration
        dt = time.time() - previous_time
        previous_time = time.time()

        # Compute the estimated states with the Extended Kalman Filter
        x_est, P_est = EKF(z, x_1, P_1, dt, False)

        # Update the variables for the next step
        x_1 = x_est
        P_1 = P_est

        return(x_1, P_1, previous_time)
        
def report_test_kalman(node, client):
    import numpy as np
    import time
    ### Test function for the Kalman Filter
    """
    This run the Extended Kalman Filter to test it, we also run the controller so that we can send the robot to a point and see on the graph the position as well as the goal
    """

    # Define the initial state
    x_1 = np.array([0, 0, 0, 0, 0])

    # Define the initial state covariance
    P_1 = np.zeros(5)

    # Initialization of the time counter
    previous_time = time.time()

    # Initilization of the variables for the plots
    x_coord = []
    y_coord = []
    STD_x = []
    STD_y = []
    STD_angle = []

    z = [0,0,0,0,0]

    while(True):
        # Obtain the wheel speed sensors measurement
        lspeed, rspeed = get_wheel_speeds(node)

        z[3] = lspeed
        z[4] = rspeed

        camera = False
        close = 1

        # Update the time counter between each iteration
        dt = time.time() - previous_time
        previous_time = time.time()

        # Compute the estimated states with the Extended Kalman Filter
        x_est, P_est = EKF(z, x_1, P_1, dt, camera)

        # Update the variables for the next step
        x_1 = x_est
        P_1 = P_est

        # Update the arrays for the plots
        x_coord.append(x_est[0])
        y_coord.append(x_est[1])

        STD_x.append(np.sqrt(P_est[0,0]))
        STD_y.append(np.sqrt(P_est[1,1]))
        STD_angle.append(np.sqrt(P_est[2,2]))

        # Time delay used for the sensors update on the Thymio
        aw(client.sleep(0.001))

        ### Controller

        # Set the goal: position where the Thymio should go
        goal = [500,500]

        # Stop at the end
        end = True

        K = 150
        close, reached = controller(node, goal, x_est, close, end, K)

        if reached is True:
            break

        ###

    ### Plot the results

    # Trajectory
    plt.plot(x_coord, y_coord,'k-')
    plt.plot(goal[0], goal[1], 'ro')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("Thymio path")
    plt.legend(['Thymio path', 'goal'])

    # x
    plt.figure()
    x_coord = np.array(x_coord)
    plt.plot(x_coord)
    plt.fill_between(np.arange(0,len(x_coord)), x_coord+STD_x, x_coord-STD_x, alpha=0.5)
    plt.xlabel("time step")
    plt.ylabel("x [mm]")
    plt.title("Error in x")
    plt.legend(['Thymio path in x', 'error'])

    # y
    plt.figure()
    y_coord = np.array(y_coord)
    plt.plot(y_coord)
    plt.fill_between(np.arange(0,len(y_coord)), y_coord+STD_y, y_coord-STD_y, alpha=0.5)
    plt.xlabel("time step")
    plt.ylabel("y [mm]")
    plt.title("Error in y")
    plt.legend(['Thymio path in y', 'error'])


def report_test_controller(node, client, goal):
    ### Test function for the controller
    """
    This code also run the Extended Kalman Filter as it is needed to get the current position
    """

    # Define the initial state
    x_1 = np.array([0, 0, 0, 0, 0])

    # Define the initial state covariance
    P_1 = np.zeros(5)

    # Initialization of the time counter
    previous_time = time.time()

    z = [0,0,0,0,0]

    while(True):
        # Obtain the wheel speed sensors measurement
        lspeed, rspeed = get_wheel_speeds(node)

        z[3] = lspeed
        z[4] = rspeed

        camera = False
        close = 1

        # Update the time counter between each iteration
        dt = time.time() - previous_time
        previous_time = time.time()

        # Compute the estimated states with the Extended Kalman Filter
        x_est, P_est = EKF(z, x_1, P_1, dt, camera)

        # Update the variables for the next step
        x_1 = x_est
        P_1 = P_est

        ### Controller

        # Stop at the end
        end = True

        K = 150
        close, reached = controller(node, goal, x_est, close, end, K)

        if reached is True:
            break

