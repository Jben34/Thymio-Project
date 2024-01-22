import numpy as np
import matplotlib.pyplot as plt
import time
import math


from tdmclient import aw

from ..filtering.KalmanFunction import get_wheel_speeds
from ..filtering.KalmanFunction import get_ground
from ..filtering.KalmanFunction import motors
from ..filtering.KalmanFunction import EKF
from ..filtering.KalmanFunction import controller

from ..local_navigation.local_nav import local_nav
from ..local_navigation.local_nav import update_sensor

import cv2

from ..vision.vision import VISION
from ..global_navigation.global_nav import build_map, find_optimal_path
v = VISION()

### Main code to run the filter
def main_motion_control(node, client):
    # Define the initial state
    x_1 = np.array([0, 0, 0, 0, 0])

    # Define the initial state covariance
    P_1 = np.zeros(5)

    # Initialization of the time counter
    previous_time = time.time()

    # Initilization of the variables for the plots
    x_coord = []
    y_coord = []
    angle = []

    STD_x = []
    STD_y = []
    STD_angle = []

    close = 1

    # Create a state machine
    state = "ground"
    camera = False

    z = [0,0,0,0,0]

    # ======= Local navigation =======

    threshold = 2500
    speed = 100
    x = []
    y = []

    # ======== Camera ========
    x_phy = 1055.0
    y_phy = 890.0
    goals = []
    # ======== Camera ========
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("error")
        exit()
    cv2.startWindowThread()
    cv2.namedWindow('Image')
    fail_img = cv2.imread('./src/vision/src/fail.jpg')
    fail_img = cv2.resize(fail_img, (1641,986), interpolation = cv2.INTER_LINEAR)
    # ======== Detect the path ========
    while(True):
        ret, image = cam.read()
        coord, goal, thymio, angle, obstacles, img = v.get_all_info(image)
        if coord is None or goal is None or thymio is None or angle is None or obstacles is None or img is None :
            continue
        else :
            
            # 
            fullMap, allPoints = build_map(thymio[1].tolist(), goal.tolist(), obstacles.tolist(), False)
            shortest_path, closedSet = find_optimal_path(fullMap, allPoints, False)

            # show
            for i in range(0,len(obstacles)):
                for j in range(len(obstacles[i])):
                    cv2.circle(img,(int(obstacles[i][j][0]),int(coord[1]-obstacles[i][j][1])), int(3), (0,0,255), 15)
            
            for i in range(len(shortest_path)-1):
                point1 = (int(shortest_path[i][0]),int(coord[1]-shortest_path[i][1]))
                point2 = (int(shortest_path[i+1][0]),int(coord[1]-shortest_path[i+1][1]))
                cv2.arrowedLine(img, point1, point2,  (255,0,0), thickness=10, tipLength=0.05)
            cv2.imwrite(f'./src/vision/src/path.jpg', img)

            # 
            for i in range(len(shortest_path)):
                each_gaol = [shortest_path[i][0]/coord[0]*x_phy, shortest_path[i][1]/coord[1]*y_phy]
                goals.append(each_gaol)
            goals = goals[1:]
            # 
            break

    i = 0
    imax = len(goals)
    end = False

    state = "ground"

    while(True):

        # ======== Camera ========
        # 
        ret, image = cam.read()
        coord_cam,thymio_cam,angle_cam,goal_cam,img = v.get_thymio_info(image)
        # CALCULATE
        if coord_cam is not None and thymio_cam is not None:
            z[0] = thymio_cam[1][0]/coord_cam[0]*x_phy
            z[1] = thymio_cam[1][1]/coord_cam[1]*y_phy
            z[2] = angle_cam

            cv2.circle(img,(int(thymio_cam[0][0]),int(coord[1]-thymio_cam[0][1])), int(3), (0,0,255), 10)
            cv2.circle(img,(int(thymio_cam[1][0]),int(coord[1]-thymio_cam[1][1])), int(3), (0,255,0), 20)

            for i_ in range(len(shortest_path)-1):
                point1 = (int(shortest_path[i_][0]),int(coord[1]-shortest_path[i_][1]))
                point2 = (int(shortest_path[i_+1][0]),int(coord[1]-shortest_path[i_+1][1]))
                cv2.arrowedLine(img, point1, point2,  (255,0,0), thickness=3, tipLength=0.05)

        # SHOW
        if img is None:
            img = fail_img
        else:
            cv2.circle(img,(int(shortest_path[i+1][0]),int(coord[1]-shortest_path[i+1][1])), int(3), (255,0,0), 30) # Goal
            cv2.imshow('Image',img)
        k = cv2.waitKey(0) 
        if k == ord('q'):
            break
                    
        # LOCAL
        
        sensor = [0,0,0,0,0]
        aw(node.wait_for_variables())
        update = update_sensor(sensor, node, client)
        
        if(state == "ground"):  
            
            # Check if need to go local
            if max(sensor) >= threshold:
                state = "local"
            
            lspeed, rspeed = get_wheel_speeds(node)

            # Obtain the wheel speed sensors measurement
            z[3] = lspeed
            z[4] = rspeed
            
            if coord_cam is not None and thymio_cam is not None:
                camera = True
            else:
                camera = False

            # Update the time counter between each iteration
            dt = time.time() - previous_time
            previous_time = time.time()

            # Compute the estimated states with the Extended Kalman Filter
            x_est, P_est = EKF(z, x_1, P_1, dt, camera)

            # Update the variables for the next step
            x_1 = x_est
            P_1 = P_est

            ### Controller
            goal = goals[i]

            # 
            if i == imax-1:
                end = True

            # 
            if goal[0] != -1 and goal[1]!=-1:

                K = 150
                close, reached = controller(node, goal, x_est, close, end, K)

            # 
            if reached is True:
                print("reached point")
                if i == imax-1:
                    print("Goal reached")
                    break
                else:
                    i = i + 1

            ###
            
            # Update the arrays for the plots
            x_coord.append(x_est[0])
            y_coord.append(x_est[1])

            STD_x.append(np.sqrt(P_est[0,0]))
            STD_y.append(np.sqrt(P_est[1,1]))
            STD_angle.append(np.sqrt(P_est[2,2]))

            # Time delay used for the sensors update on the Thymio
            aw(client.sleep(0.001))

            # Get the value of the ground sensors
            ground = get_ground(node)

            print(" ")
        
            # Update the state
            if(ground[0]<20 or ground[1]<20):
                state = "kidnapped"

        if(state == "kidnapped"):
            print(state)
            aw(node.set_variables(motors(0, 0)))
            while(True):
                aw(node.wait_for_variables())
                aw(client.sleep(0.1))
                # Get the value of the ground sensors
                ground = get_ground(node)
                if(ground[0]>100 or ground[1]>100):
                    state = "plan"
                    break
        
        if(state == "local"):
            x_1, P_1, previous_time = local_nav(threshold, speed, sensor, x_1, P_1, previous_time, node, client)
            state = "plan"
            
        if(state == "plan"):
            print(state)
            aw(node.set_variables(motors(0, 0)))
            # ======== Detect the path ========
            while(True):
                goals = []
                ret, image = cam.read()
                coord, goal, thymio, angle, obstacles, img = v.get_all_info(image)
                if coord is None or goal is None or thymio is None or angle is None or obstacles is None or img is None :
                    continue
                else :

                    # 
                    fullMap, allPoints = build_map(thymio[1].tolist(), goal.tolist(), obstacles.tolist(), False)
                    shortest_path, closedSet = find_optimal_path(fullMap, allPoints, False)

                    # show
                    for i in range(0,len(obstacles)):
                        for j in range(len(obstacles[i])):
                            cv2.circle(img,(int(obstacles[i][j][0]),int(coord[1]-obstacles[i][j][1])), int(3), (0,0,255), 15)

                    for i in range(len(shortest_path)-1):
                        point1 = (int(shortest_path[i][0]),int(coord[1]-shortest_path[i][1]))
                        point2 = (int(shortest_path[i+1][0]),int(coord[1]-shortest_path[i+1][1]))
                        cv2.arrowedLine(img, point1, point2,  (255,0,0), thickness=10, tipLength=0.05)
                    cv2.imwrite(f'./src/vision/src/path.jpg', img)

                    # 
                    for i in range(len(shortest_path)):
                        each_gaol = [shortest_path[i][0]/coord[0]*x_phy, shortest_path[i][1]/coord[1]*y_phy]
                        goals.append(each_gaol)
                    goals = goals[1:]
                    # 
                    break
            i = 0
            imax = len(goals)
            end = False
            print(goals)
            print("Path replanned")
            state = "ground"
            

    # Camera
    cam.release()
    cv2.destroyAllWindows()

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
