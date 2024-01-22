#@func   : the function for camera
#@author : Zhefei Gong
#@time   : Nov.2023

import cv2
import numpy as np

#@func : 
class VISION:

    def __init__(self):

        # ============== hyperparameters ==============

        # border
        self.border_area_min=1
        self.border_lower=np.array([125,75,25])
        self.border_upper=np.array([175,155,95])
        self.border_erode_iterations = 4

        # goal
        self.goal_lower=np.array([30,85,55])
        self.goal_upper=np.array([75,145,105])
        self.goal_erode_iterations = 4

        # Thymio
        self.Thymio_front_lower = np.array([110,125,100])
        self.Thymio_front_upper = np.array([145,205,165])

        self.Thymio_rear_lower = np.array([5,140,75])
        self.Thymio_rear_upper = np.array([35,220,165])
        self.Thymio_erode_iterations = 4
        
        # obstacles
        self.obstacle_lower = 90
        self.obstacle_upper = 255
        self.obstacle_area_min = 1000
        self.obstacle_area_max = 200000
        self.obstacle_zoom_ratio = 1.0
        
        self.obstacle_zoom_distance = 90

        self.delta = 0.040 # 0.009

    #@func : 
    def order_points(self, pts):
        # sorting points first by the 2nd the coordinate then 1st coordinate
        pts=sorted(pts, key=lambda x: (int(x[1]), int(x[0]))) #topleft,topright,bottomleft,bottomright
        # top left x > top right x => erreur et intervertit
        if pts[0][0] > pts[1][0]:
            pts[0], pts[1] = pts[1], pts[0]
        if pts[2][0] > pts[3][0]:
            pts[2], pts[3] = pts[3], pts[2]
        return pts

    #@func : 
    def transform_image(self, image, ordered_pts):
	    # obtain a consistent order of the points and unpack them
	    # individually
        (tl, tr, bl, br) = ordered_pts
	    # compute the width of the new image, which will be the
	    # maximum distance between bottom-right and bottom-left
	    # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
	    # compute the height of the new image, which will be the
	    # maximum distance between the top-right and bottom-right
	    # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
	    # now that we have the dimensions of the new image, construct
	    # the set of destination points to obtain a "birds eye view",
	    # (i.e. top-down view) of the image, again specifying points
	    # in the top-left, top-right, bottom-right, and bottom-left
	    # order
        # conversion needed corner points => np.array
        ordered_pts=np.array(ordered_pts , dtype = "float32")
        dst = np.array([[0, 0],[maxWidth - 1, 0],[0, maxHeight - 1],[maxWidth - 1, maxHeight - 1]],dtype = "float32")
	    # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    #@func : 
    def calibrate_image(self, image):
        
        # detect
        img_blur = cv2.GaussianBlur(image, (7, 7), 0)
        img_HSV = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
        mask_range = cv2.inRange(img_HSV, self.border_lower,self.border_upper)
        mask_erode = cv2.erode(mask_range, None, iterations=self.border_erode_iterations)
        mask = cv2.dilate(mask_erode, None, iterations=self.border_erode_iterations)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        # suppressing false corners
        initial_length=len(contours)
        for i in range(len(contours)):
            backwards_i=initial_length-i-1
            area = cv2.contourArea(contours[backwards_i])
            # Shortlisting the regions based on there area.
            if area < self.border_area_min:
                del contours[backwards_i]
    
        # finding corners center
        corner_points = []
        for i in range(len(contours)):
            if (cv2.contourArea(contours[i]) > self.border_area_min):
                mom = cv2.moments(contours[i])
                corner_points.append((int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])))

        # check
        if len(corner_points) != 4:
            return None
            raise TypeError(f"failure in identifying corners, the num is{len(corner_points)}")
    
        # transform
        corner_points=self.order_points(corner_points)
        warped_img=self.transform_image(image, corner_points)
    
        return warped_img
    
    #@func : 
    def detect_Thymio(self, image):

        pts=[]

        # detect
        img_blur = cv2.GaussianBlur(image, (7, 7), 0)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

        # front 
        mask_f = cv2.inRange(img_hsv, self.Thymio_front_lower, self.Thymio_front_upper)
        mask_f = cv2.erode(mask_f, None, iterations = self.Thymio_erode_iterations)
        mask_f = cv2.dilate(mask_f, None, iterations = self.Thymio_erode_iterations)
        fronts,_ = cv2.findContours(mask_f, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # rear
        mask_r = cv2.inRange(img_hsv, self.Thymio_rear_lower, self.Thymio_rear_upper)
        mask_r = cv2.erode(mask_r, None, iterations = self.Thymio_erode_iterations)
        mask_r = cv2.dilate(mask_r, None, iterations = self.Thymio_erode_iterations)
        rears,_ = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if (len(fronts) == 1):
            ((x_f,y_f),rayon_f) = cv2.minEnclosingCircle(fronts[0])
        else:
            return None
            raise TypeError("[ERROR] failure in identifying Thymio")
        if (len(rears) == 1):
            ((x_r,y_r),rayon_r) = cv2.minEnclosingCircle(rears[0])
        else:
            return None
            raise TypeError("[ERROR] failure in identifying Thymio")

        pts=[[x_f, y_f],[x_r, y_r]]

        return np.array(pts) # array
    
    #@func : 
    def detect_goal(self, image):

        coord = []
        coord = [0,0]

        # detect
        img_blur = cv2.GaussianBlur(image, (7, 7), 0)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, self.goal_lower, self.goal_upper)
        mask = cv2.erode(mask, None, iterations = self.goal_erode_iterations)
        mask = cv2.dilate(mask, None, iterations = self.goal_erode_iterations)
        elements,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(elements) == 1:
            for i in range(0,len(elements)):
                c = elements[i] 
                ((x,y),rayon) = cv2.minEnclosingCircle(c)
                coord = [int(x),int(y)]
        else:
            return None
            raise TypeError("[ERROR] failure in identifying goal")

        return np.array(coord) # array
    
    #@func : 
    def detect_obstacle(self, image):    

        list_polygon=[]
        polygon=[]

        # detect
        image = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
        _,threshold = cv2.threshold(image, self.obstacle_lower, self.obstacle_upper, cv2.THRESH_BINARY)
        contours,_=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Searching through every region selected to find the required polygon.
        for cnt in contours :
            area = cv2.contourArea(cnt)

            # Shortlisting the regions based on there area.
            if area > self.obstacle_area_min: 
                if area < self.obstacle_area_max:

                     # Checking if the no. of sides of the selected region is 4
                    approx = cv2.approxPolyDP(cnt, self.delta * cv2.arcLength(cnt, True), True)

                    if(len(approx) == 4): 
                        for i in range (len(approx)):
                            point=[approx[i][0][0],approx[i][0][1]]
                            polygon.append(point)
                        list_polygon.append(polygon)
                        polygon=[]
        
        if len(list_polygon)==0:
            return None
        else:
            return np.array(list_polygon) # array
    
    #@func : 
    def calculate_angle(self, pts):
        #pts[0] is the big_circle and pts[1] is the little circle
        dist = [pts[0][0]-pts[1][0],pts[0][1]-pts[1][1]]
        #inverse distance in y because y axis is inversed in openCv 
        dist[1] = -dist[1]
        ang = np.arctan2(dist[1],dist[0])
        return ang
    
    #@func :
    def zoom_obstacle(self, list_poly, max_x = 200, max_y = 200):
        # list_polygon has shape : [num_obstacle, 4, 2]

        # calculate
        center_poly = np.mean(list_poly,axis=1)
        center_poly = np.expand_dims(center_poly, axis=1) # [N,4,2]

        list_poly_zoom  = list_poly + self.obstacle_zoom_ratio * (list_poly - center_poly) 
        
        # clip
        x_coords = list_poly_zoom[:, :, 0]
        y_coords = list_poly_zoom[:, :, 1]
        clipped_x = np.clip(x_coords, 0, max_x)
        clipped_y = np.clip(y_coords, 0, max_y)
        list_poly_zoom_clip = np.stack((clipped_x, clipped_y), axis=-1)

        return list_poly_zoom_clip

    #@func :
    def get_all_info(self, image):
        
        #
        ErrorBack = None,None,None,None,None,None

        # 
        img = self.calibrate_image(image)
        if img is None : 
            print('[ERROR] calibration fail')
            return ErrorBack
        else:
            coord = np.array([img.shape[1],img.shape[0]])
        
        # 
        goal = self.detect_goal(img)
        if goal is None : 
            print('[ERROR] goal detection fail')
            return ErrorBack
        else:
            goal[1] = coord[1] - goal[1] # reverse
        
        # 
        thymio = self.detect_Thymio(img)
        if thymio is None:
            print('[ERROR] thymio detection fail')
            return ErrorBack
        else:
            angle = self.calculate_angle(thymio)
            thymio[:,1] = coord[1] - thymio[:,1] # reverse
        
        #
        obstacles = self.detect_obstacle(img)
        if obstacles is None:
            print('[ERROR] obstacles detection fail')
            return ErrorBack
        else:
            obstacles = self.zoom_obstacle(obstacles, coord[0], coord[1])
            obstacles[:,:,1] = coord[1] - obstacles[:,:,1] # reverse
        
        return coord, goal, thymio, angle, obstacles, img

    #@func :
    def get_thymio_info(self, image):

        #
        ErrorBack = None,None,None,None,None

        # 
        img = self.calibrate_image(image)
        if img is None : 
            print('[ERROR] calibration fail')
            return ErrorBack
        else:
            coord = np.array([img.shape[1],img.shape[0]])

        # 
        goal = self.detect_goal(img)
        if goal is None : 
            print('[ERROR] goal detection fail')
            return ErrorBack
        else:
            goal[1] = coord[1] - goal[1] # reverse
                
        # 
        thymio = self.detect_Thymio(img)
        if thymio is None:
            print('[ERROR] thymio detection fail')
            return ErrorBack
        else:
            angle = self.calculate_angle(thymio)
            thymio[:,1] = coord[1] - thymio[:,1] # reverse

        return coord,thymio,angle,goal,img


if __name__ == "__main__":
    print("test")
