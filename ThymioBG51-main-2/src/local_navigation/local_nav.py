import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np

from tdmclient import aw

from ..filtering import KalmanFunction as kf

threshold = 2500
speed = 100


def update_sensor(sens, node, client) :
    #Update sens with current sensor values

    previous =  datetime.datetime.now()
    aw (client.sleep(0.1))
    aw (node.wait_for_variables())
    for i in range(5) :
        sens[i] = node["prox.horizontal"][i]
    elapsed_time = datetime.datetime.now()-previous
    return elapsed_time



def turn_right(node) :
    # Turn right for an indefinite duration
    v = {
    "motor.left.target": [speed],
    "motor.right.target": [-speed],
    }
    aw (node.set_variables(v))
    
    
def turn_left(node) :
    # Turn left for an indefinite duration
    v = {
    "motor.left.target": [-speed],
    "motor.right.target": [speed],
    }
    aw (node.set_variables(v))
    
    
def stop(node, client) :
    # Stop motors
    v = {
    "motor.left.target": [0],
    "motor.right.target": [0],
    }
    aw (node.set_variables(v))
    aw (client.sleep(0.1))

def forward(speed, node) :
    # Move forward for an indefinite duration
    v = {
    "motor.left.target": [speed],
    "motor.right.target": [speed],
    }
    aw (node.set_variables(v))
    


def angle_adaptive(direction, sens, x_1, P_1, previous_time, node, client) :
    aw (node.wait_for_variables())
    # If the obstacle is already parallel or close to parallel, no adjustment
    if max(sens) == 0 :
        return x_1, P_1, previous_time
    
    # The robot will turn left or right until its sensors do not see anything
    if (direction == "right") :
        turn_right(node)
        while sum(sens[i] > 0 for i in range(0,5)) > 0 : 
            x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
            time_sensor = update_sensor(sens, node, client)
        x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
        stop(node, client)

    else :
        turn_left(node)
        while sum(sens[i] > 0 for i in range(0,5)) > 0 : 
            x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
            time_sensor = update_sensor(sens, node, client)
        x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
        stop(node, client)
    return x_1, P_1, previous_time
        

            
def obstacle_check(direction, sens, obstacle, counter, x_1, P_1, previous_time, node, client) :
    # The robot makes a 90 degrees rotation in the direction of the obsracle and
    # checks its sensors, if the obtscale is still there it comes back to its original orientation  by making a 90 degrees turn in the other direction, else, it just returns.to
    
    aw (node.wait_for_variables())
    if (direction == "right") :
        elapsed_time = datetime.timedelta(seconds=0)
        while True :
            previous =  datetime.datetime.now()
            turn_left(node)
            x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
            time_sensor = update_sensor(sens, node, client)
            elapsed_time =  elapsed_time + datetime.datetime.now()-previous
            if (elapsed_time >= datetime.timedelta(seconds = 2.1)) :
                elapsed_time = datetime.timedelta(milliseconds=0)
                time_sensor = datetime.timedelta(milliseconds=0)
                x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
                stop(node, client)
                update = update_sensor(sens, node, client)
                break
        if (max(sens)) == 0:
            obstacle = False
            counter += 1
            if (counter == 1) :
                return obstacle, counter, x_1, P_1, previous_time
            else :
                return obstacle, counter, x_1, P_1, previous_time
        else :
            obstacle = True
            while True :
                print ("obstacle still in sight")
                previous =  datetime.datetime.now()
                turn_right(node)
                x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
                time_sensor = update_sensor(sens, node, client)
                elapsed_time =  elapsed_time + datetime.datetime.now()-previous
                print (elapsed_time)
                if (elapsed_time >= datetime.timedelta(seconds = 2.1)) :
                    print ("We should stop")
                    elapsed_time = datetime.timedelta(milliseconds=0)
                    time_sensor = datetime.timedelta(milliseconds=0)
                    x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
                    stop(node, client)
                    update = update_sensor(sens, node, client)
                    break
            return obstacle, counter, x_1, P_1, previous_time
            
       
    # Handles the left wall follow case, simply mirrors the right wall follow obstacle check
    else :
        elapsed_time = datetime.timedelta(seconds=0)
        while True :
            previous =  datetime.datetime.now()
            turn_right(node)
            x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
            time_sensor = update_sensor(sens, node, client)
            elapsed_time =  elapsed_time + datetime.datetime.now()-previous
            if (elapsed_time >= datetime.timedelta(seconds = 2.1)) :
                elapsed_time = datetime.timedelta(milliseconds=0)
                time_sensor = datetime.timedelta(milliseconds=0)
                x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
                stop(node, client)
                update = update_sensor(sens, node, client)
                break
        if (max(sens)) == 0:
            obstacle = False
            counter += 1
            if (counter == 1) :
                return obstacle, counter, x_1, P_1, previous_time
            else :
                return obstacle, counter, x_1, P_1, previous_time
        else :
            obstacle = True
            while True :
                previous =  datetime.datetime.now()
                turn_left(node)
                x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
                time_sensor = update_sensor(sens, node, client)
                elapsed_time =  elapsed_time + datetime.datetime.now()-previous
                if (elapsed_time >= datetime.timedelta(seconds = 2.1)) :
                    elapsed_time = datetime.timedelta(milliseconds=0)
                    time_sensor = datetime.timedelta(milliseconds=0)
                    x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
                    stop(node, client)
                    update = update_sensor(sens, node, client)
                    break
            return obstacle, counter, x_1, P_1, previous_time


def local_nav(threshold, speed, sensor, x_1, P_1, previous_time, node, client):
    stop(node, client)
    x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
    aw (node.wait_for_variables())
    elapsed_time = datetime.timedelta(milliseconds=0)
    if (sensor[0]+sensor[1]) > (sensor[4]+ sensor[3]) :   # If the object is closer to the left, we follow it from the right
        print("Turn right")
        direction = "right"
        # Adapt rotation angle so that the robot is nearly parallel to the obstacle.
        x_1, P_1, previous_time = angle_adaptive(direction, sensor, x_1, P_1, previous_time, node, client)
        aw (client.sleep(0.1))
        aw (node.wait_for_variables())
        obstacle = True
        counter = 0
        while True:
            # The robot goes forward for 1.5 seconds, adapts its angle again if it needs to and then does an obstacle check. If no obstacle detected, the robot will then see if it already has not detected an obstacle after an obstacle check before. If yes, local navigation will be done as the objects are rectangular. If not, the robot will now move forward in the newly free direction and repeat the same process.
            time_sensor = update_sensor(sensor, node, client)
            x_1, P_1, previous_time = angle_adaptive(direction, sensor, x_1, P_1, previous_time, node, client) 
            previous = datetime.datetime.now()
            forward(speed, node)
            x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
            time_sensor = update_sensor(sensor, node, client)
            elapsed_time =  elapsed_time + datetime.datetime.now()-previous
            if (elapsed_time >= datetime.timedelta(seconds = 1.5)+time_sensor) :
                obstacle, counter, x_1, P_1, previous_time = obstacle_check(direction, sensor, obstacle, counter, x_1, P_1, previous_time, node, client)
                elapsed_time = datetime.timedelta(milliseconds=0)
                time_sensor = datetime.timedelta(milliseconds=0)
                if (obstacle == False) :
                    if (counter == 2) :
                        print ("obstacle avoided")
                        return x_1, P_1, previous_time
                    else :
                        obstacle = True
                        update = update_sensor(sensor, node, client)
                        x_1, P_1, previous_time = angle_adaptive(direction, sensor, x_1, P_1, previous_time, node, client)
    # Mirrors the right wall follow case i.e only the turn directions change.
    else  :   # If the object is closer to the left, we follow it from the right
        print("Turn left")
        direction = "left"
        x_1, P_1, previous_time = angle_adaptive(direction, sensor, x_1, P_1, previous_time, node, client) 
        aw (client.sleep(0.1))
        aw (node.wait_for_variables())
        obstacle = True
        counter = 0
        while True:
            time_sensor = update_sensor(sensor, node, client)
            x_1, P_1, previous_time = angle_adaptive(direction, sensor, x_1, P_1, previous_time, node, client) 
            previous1 =  datetime.datetime.now()
            forward(speed, node)
            x_1, P_1, previous_time = kf.KalmanFilter(node, previous_time, x_1, P_1)
            time_sensor = update_sensor(sensor, node, client)
            elapsed_time =  elapsed_time + datetime.datetime.now()-previous1
            if (elapsed_time >= datetime.timedelta(seconds = 1.5)+time_sensor) :
                obstacle, counter, x_1, P_1, previous_time = obstacle_check(direction, sensor, obstacle, counter, x_1, P_1, previous_time, node, client)
                print(obstacle)
                print(counter)
                elapsed_time = datetime.timedelta(milliseconds=0)
                time_sensor = datetime.timedelta(milliseconds=0)
                if (obstacle == False) :
                    if (counter == 2) :
                        print ("obstacle avoided")
                        return x_1, P_1, previous_time
                    else :
                        obstacle = True
                        update = update_sensor(sensor, node, client)
                        x_1, P_1, previous_time = angle_adaptive(direction, sensor, x_1, P_1, previous_time, node, client)
      
def report_test_local_nav(node, client):
    # Define the initial state
    x_1 = np.array([0, 0, 0, 0, 0])
    # Define the initial state covariance
    P_1 = np.zeros(5)
    # Initialization of the time counter
    previous_time = time.time()
    threshold = 2500
    speed = 100
    while True :
        sensor = [0, 0, 0, 0, 0]
        sp = {
        "motor.left.target": [speed],
        "motor.right.target": [speed],
        }
        aw (node.set_variables(sp))
        aw (client.sleep(0.1))
        aw (node.wait_for_variables())
        update = update_sensor(sensor, node, client)
        if sum(sensor[i]  >= threshold for i in range(0,5)) > 0:
            aw (node.set_variables({"leds.top":[32, 0, 0]}))
            print("entered local nav")
            local_nav(threshold, speed, sensor, x_1, P_1, previous_time, node, client)
            aw (node.set_variables({"leds.top":[0, 0, 32]}))
            break
