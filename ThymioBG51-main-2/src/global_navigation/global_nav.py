# The functions in this file are used for the global navigation

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ..vision.vision import VISION

#to check if two segments cross
def crossing_segments(segment1, segment2):
    #we make it so that the x coordinate of the start of each segment is smaller
    #than the x coordinate of the end of the segment
    if (segment1[0][0] > segment1[1][0]):
        tmp_point = segment1[0]
        segment1[0] = segment1[1]
        segment1[1] = tmp_point
    if (segment2[0][0] > segment2[1][0]):
        tmp_point = segment2[0]
        segment2[0] = segment2[1]
        segment2[1] = tmp_point
    if ((segment1[0] == segment2[0]) or (segment1[0] == segment2[1]) or (segment1[1] == segment2[0]) or (segment1[1] == segment2[1])):
        #in this case, one segment directly follows the other, so there can be no intersection between them except for one point,
        #but in this case, there is no collision with an obstacle.
        return False
    
    #we compute the slopes of the segments
    if ((segment1[1][0] - segment1[0][0]) == 0):
        #knowing the size of the image, we can approximate infinity with 10000
        a1 = 10000
    else:
        a1 = (segment1[1][1] - segment1[0][1])/(segment1[1][0] - segment1[0][0])
        
    if ((segment2[1][0] - segment2[0][0]) == 0):
        #knowing the size of the image, we can approximate infinity with 10000
        a2 = 10000
    else:
        a2 = (segment2[1][1] - segment2[0][1])/(segment2[1][0] - segment2[0][0])
                
    #if the slopes are equal (i.e. they are parallel), we consider that there is no intersection because there
    #will not be a collision with an obstacle
    if (a1 == a2):
        return False
    #we compute the ordinate at the origin
    b1 = segment1[0][1] - (a1 * segment1[0][0])
    b2 = segment2[0][1] - (a2 * segment2[0][0])
    #we compute the coordinates of the potential crossing point between the two wegments
    intersectionPoint = [(b2 - b1)/(a1 - a2), (((b2 - b1)/(a1 - a2)) * a1) + b1]
    #finally we check if this point is on the segments
    #because of rounding errors, we can not use comparaison in cases where a line is
    #perpendicular to either the x or the y axis. So, we use a small error margin: epsilon_error
    epsilon_error = 10
    if ((intersectionPoint[0] >= (max(segment1[0][0], segment2[0][0]) - epsilon_error)) and (intersectionPoint[0] <= (min(segment1[1][0], segment2[1][0]) + epsilon_error))):
        if ((intersectionPoint[1] >= (max(min(segment1[0][1], segment1[1][1]), min(segment2[0][1], segment2[1][1])) - epsilon_error)) and (intersectionPoint[1] <= (min(max(segment1[0][1], segment1[1][1]), max(segment2[0][1], segment2[1][1])) + epsilon_error))):
        #only in this case do the segments intersect, because the intersection point's coordinates are in the
        #intervals created by the intersection of the intervals of the coordinates of each segment
            return True
    return False

#compute the distance between two points
def distance_two_points(point1, point2):
    return math.sqrt(math.pow((point1[0] - point2[0]), 2) + math.pow((point1[1] - point2[1]), 2))
    
#construct the map
def build_map(startPoint, goalPoint, obstacles, drawMap):
    #we list all the walls of obstacles that could cause a collision
    obstacleWalls = []
    allPoints = [startPoint]
    for obstacle in obstacles:
        obstacleWalls.append([obstacle[0], obstacle[1]])
        obstacleWalls.append([obstacle[1], obstacle[2]])
        obstacleWalls.append([obstacle[2], obstacle[3]])
        obstacleWalls.append([obstacle[3], obstacle[0]])
        allPoints.append(obstacle[0])
        allPoints.append(obstacle[1])
        allPoints.append(obstacle[2])
        allPoints.append(obstacle[3])
    allPoints.append(goalPoint)

    #we build the empty matrix that will contain the information about the distances
    #between different points on the map. -1 represents a distance that has not yet been canculated,
    #np.inf represents a distance that is not relevent because the robot going from one end to the
    #other would result in a collision.
    fullMap = -1*np.ones((len(allPoints), len(allPoints))) + np.identity(len(allPoints))
    
    #now we compute the distances and check if there is a collision between the robot and a wall
    for i in range(len(allPoints)):
        for j in range(len(allPoints)):
            if (fullMap[i][j] == 0):
                j = 0
                i = i+1
                continue
            collision = False
            for wall in obstacleWalls:
                if (crossing_segments([allPoints[i], allPoints[j]], wall)):
                    collision = True
                    break
            if collision:
                fullMap[i][j] = np.inf
                fullMap[j][i] = np.inf
                continue
            #here we eliminate the diagonals of the obstacles
            if (((i - j) == 2) and (i != 0) and (i != len(allPoints)) and (j != 0) and (j != len(allPoints))  and (((i-1)//4) == ((j-1)//4))):
                fullMap[i][j] = np.inf
                fullMap[j][i] = np.inf
                continue
            fullMap[i][j] = distance_two_points(allPoints[j], allPoints[i])
            fullMap[j][i] = distance_two_points(allPoints[j], allPoints[i])
            if (drawMap):
            #we show a representation of the map. The obstacles are in red, the start and goal are
            #in blue, the paths are in green
                plt.plot([allPoints[i][0], allPoints[j][0]], [allPoints[i][1], allPoints[j][1]], 'g-')
        
        if (drawMap):
        #we show a representation of the map. The obstacles are in red, the start and goal are
        #in blue, the paths are in green
            for obstacle in obstacles:
                plt.plot([obstacle[0][0], obstacle[1][0]], [obstacle[0][1], obstacle[1][1]], 'ro-')
                plt.plot([obstacle[1][0], obstacle[2][0]], [obstacle[1][1], obstacle[2][1]], 'ro-')
                plt.plot([obstacle[2][0], obstacle[3][0]], [obstacle[2][1], obstacle[3][1]], 'ro-')
                plt.plot([obstacle[3][0], obstacle[0][0]], [obstacle[3][1], obstacle[0][1]], 'ro-')
            plt.plot(startPoint[0], startPoint[1], 'bo')
            plt.plot(goalPoint[0], goalPoint[1], 'bo')
    
    return fullMap, allPoints
    
    #We get the points that are linked (are neighbors) to a reference point
def get_current_neighbors(current, fullMap, pointsList):
    neighborIndexes = []
    currentPointIndex = -1;
    for pointIndex in range(len(pointsList)):
        if (current == pointsList[pointIndex]):
            currentPointIndex = pointIndex
    for j in range(len(pointsList)):
        if ((fullMap[currentPointIndex][j] == 0) or (fullMap[currentPointIndex][j] == np.inf)):
            continue
        neighborIndexes.append(j)
    return currentPointIndex, neighborIndexes

def reconstruct_path(cameFrom, current, drawPath):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    totalPath = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        totalPath.insert(0, cameFrom[current]) 
        current=cameFrom[current]
    if drawPath:
        for point in range(1, len(totalPath)):
            #the optimal path is drawn in yellow
            plt.plot([totalPath[point-1][0], totalPath[point][0]], [totalPath[point-1][1], totalPath[point][1]], 'yo-')
    return totalPath

#We implement an A* algorythm to find the shortest path from start to goal
#Here, the code used in the solution of Exercice session 5 is reused
def find_optimal_path(fullMap, allPoints, drawPath):
    pointsList = list([(int(point[0]), int(point[1])) for point in allPoints])
    startPoint = pointsList[0]
    goalPoint = pointsList[-1]
    
    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [startPoint]
    
    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(pointsList, [np.inf for x in range(len(pointsList))]))
    gScore[startPoint] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(pointsList, [np.inf for x in range(len(pointsList))]))
    fScore[startPoint] = distance_two_points(allPoints[0], allPoints[-1])

    # while there are still elements to investigate
    while openSet != []:
        
        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        #If the goal is reached, reconstruct and return the obtained path
        if current == goalPoint:
            return reconstruct_path(cameFrom, current, drawPath), closedSet

        openSet.remove(current)
        closedSet.append(current)
        
        #for each neighbor of current:
        currentPointIndex, neighborIndexes = get_current_neighbors(current, fullMap, pointsList)
        for neighborIndex in neighborIndexes:            
            # if the node is occupied or has already been visited, skip
            if (pointsList[neighborIndex] in closedSet): 
                continue
                
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[pointsList[currentPointIndex]] + fullMap[currentPointIndex][neighborIndex]
            
            if pointsList[neighborIndex] not in openSet:
                openSet.append(pointsList[neighborIndex])
                
            if tentative_gScore < gScore[pointsList[neighborIndex]]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[pointsList[neighborIndex]] = current
                gScore[pointsList[neighborIndex]] = tentative_gScore
                fScore[pointsList[neighborIndex]] = gScore[pointsList[neighborIndex]] + distance_two_points(goalPoint, allPoints[neighborIndex])

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet



def report_compute_path(image, displayMap=False, displayMapOverlay=False):
    v=VISION()
    coord, goal, thymio, angle, obstacles, img = v.get_all_info(image)
    fullMap, allPoints = build_map(thymio[1].tolist(), goal.tolist(), obstacles.tolist(), displayMap)
    shortest_path, closedSet = find_optimal_path(fullMap, allPoints, displayMap)
    if (shortest_path == []):
        return np.nan
    if (displayMapOverlay == True):
        plt.imshow(v.calibrate_image(image)[::-1], origin='lower')
    return shortest_path, thymio, angle
