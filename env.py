# Original Source
#! /usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor using AirSim python API

- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
- Date: 2019.06.20.
"""

# Edited by : Varun Pawar   
# E-mail : varunpwr897@gmail.com

# Contains necessary function to connect AirSim API and the DeepRL training function.
# Simpler form of gymWrapper implementation.

import math
import time
from PIL import Image

import numpy as np

import airsim
import cv2
import random
from numpy import random as rnd
import torch

class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print("Initial Position: ", self.state.x_val, self.state.y_val, self.state.z_val)
        self.quad_offset = (0, 0, 0)
        self.mindist = 100
        initX = 0
        initY = 0
        initZ = -3.5

        self.delta_t = 1
        self.drone_velocity = 1
        self.kp = 0.09
        
        self.DIMAGE = 0*np.random.rand(32,32)
        self.dsteps = 0

        self.start_collision = "Cube"
        self.next_collision = "Cube"
        self.cnt_collision = 0
        self.collision_change = False
        self.Objs = self.setObsRandom()
        
        # take the drone to 5 meters of height
        self.client.takeoffAsync().join()
        print("take off moving positon")
        self.client.moveToPositionAsync(initX, initY, initZ, 5).join()
        head = random.uniform(-np.pi, np.pi)*0 + 0*np.pi
        self.client.rotateToYawAsync(yaw = head*180/np.pi, timeout_sec = 3e+38, margin = 5).join()
        self.destination = airsim.Vector3r(203.0*math.cos(head), 203.0*math.sin(head),-3.5)
        self.ep = 0#first episode of the training loop.

    def step(self, action):
        """Step"""
        self.dsteps +=1
        print("doing step")
        # if steps_done%4 == 0:
        # self.quad_offset is (v_xval, v_yval, v_zval, yaw)
        self.quad_offset, qhead = self.interpret_action(action)
        print("quad_offset: ", self.quad_offset)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        next_quad_state = airsim.Vector3r(quad_state.x_val + self.delta_t*max(self.quad_offset[0], 0), 
                            quad_state.y_val + self.delta_t*self.quad_offset[1], 
                            -2.0)
        self.client.moveOnPathAsync([quad_state,
                                next_quad_state],
                        self.drone_velocity, 0.5*self.delta_t,
                        airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0)).join()

        collision_info = self.client.simGetCollisionInfo()


        if self.next_collision != collision_info.object_name:
            self.collision_change = True

        if collision_info.has_collided:
            if self.cnt_collision == 0:
                self.start_collision = collision_info.object_name
                self.next_collision = collision_info.object_name
                self.cnt_collision = 1
            else:
                self.next_collision = collision_info.object_name

        prev_quad_state = quad_state
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        print(
            "state x:",
            quad_state.x_val,
            " y: ",
            quad_state.y_val,
            " z: ",
            quad_state.z_val,
        )
        state = self.Dimage()
        result = self.compute_reward(state, quad_state, quad_vel)

        done = self.isDone(result,  prev_quad_state)

        if action is 1:
            result += 0.5
        #Prefer moving forward
        if done:
            result = -10

        #Penalize heavily if collides
        if quad_state.x_val >= 400:
            self.dsteps = 0
            done = True
            #If certain number of steps are completed then stop
        return state, result, done

    def reset(self):
        self.client.armDisarm(False)
        # time.sleep(1)
        self.client.reset()
        self.client.enableApiControl(False)
        """Reset to initial state"""
        self.dsteps = 0
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print(self.state.x_val, self.state.y_val, self.state.z_val)
        self.quad_offset = (0, 0, 0)
        initX = 0
        initY = 0
        initZ = -3.5

        self.start_collision = "Cube"
        self.next_collision = "Cube"
        self.cnt_collision = 0
        self.collision_change = False
        self.Objs = self.setObsRandom()
        self.client.takeoffAsync().join()
        print("take off moving positon")
        self.client.moveToPositionAsync(initX, initY, initZ, 5).join()
        head = random.uniform(-np.pi, np.pi)*0 +  0*np.pi
        self.client.rotateToYawAsync(yaw = head*180/np.pi, timeout_sec = 3e+38, margin = 5).join()
        responses = self.client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        )
        state = self.Dimage()
        return state

    def get_obs(self):
        """Get observation"""
        responses = self.client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        )
        obs = self.transform_input(responses)
        return obs

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array([-10, 10, -10])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def isDone(self, reward,quad_state):
        """Check if episode is done"""
        done = 0
        if (self.mindist < 1):            
            done = 1
        elif self.client.simGetCollisionInfo().has_collided:
            done = 1
        return done

    def transform_input(self, responses):
        """Transform input binary array to image"""
        response = responses[0]
        img1d = np.fromstring(
            response.image_data_uint8, dtype=np.uint8
        ) 
        img_rgba = img1d.reshape(
            response.height, response.width, 3
        ) 
        img2d = np.flipud(img_rgba)

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final

    def interpret_action(self, action):
        """Interprete action"""
        
        _, _, qhead  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        
        # Each action relative to current heading 
        d = self.client.getMultirotorState().kinematics_estimated.position.x_val
        e = self.client.getMultirotorState().kinematics_estimated.position.y_val
        delta = -1*qhead - math.atan2(0.08*e, self.drone_velocity)
        alpha = math.exp(-abs(0*e/10))
        k2 = np.exp(-(action-1)**2)
        ux = -max(min(d-500,1),-1)
        uy = -max(min(e-0, 1),-1)
        heading = alpha*self.kp*(action-1)*np.pi/2 + (1-alpha)*0.05*delta
        qhead = heading*self.delta_t + qhead
        print("-----Heading:", qhead)
        quad_offset = (self.drone_velocity*math.cos(qhead), self.drone_velocity*math.sin(qhead), 0)
        return quad_offset, qhead

    def RGBimage(self):
        frames = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        frame = frames[0]
        # get numpy array
        img1d = np.fromstring(frame.image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 4
        if img1d.shape == (1,1):
            isImage = False
            img_rgb = img1d
        else:
            isImage = True
            img_rgb = img1d.reshape(frame.height, frame.width,3)
        return img_rgb

    def Dimage(self):
        # "Returns the depth camera feed"
        frames = self.client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
        
        frame = frames[0]
        # if frame.height != 1 and frame.width != 1:
        img1d = np.array(frame.image_data_float, dtype=np.float)
        img1d = img1d*3.5+30 
        img1d[img1d>255] = 255
        
        if frame.height == 0 or frame.width == 0 :
            img2d = 255*np.random.rand(32,32)
        else:
            img2d = np.reshape(img1d, (frame.height, frame.width))
        
        depth = cv2.resize(img2d, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        state = np.array(depth, dtype=np.float32)
        return state

    def DVisimage(self):
        # "Returns the depth camera feed"
        frames = self.client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.DepthVis, pixels_as_float=True)])
        
        frame = frames[0]
        # if frame.height != 1 and frame.width != 1:
        img1d = np.array(frame.image_data_float, dtype=np.float)
        # img1d = img1d*3.5+30 
        img1d[img1d>255] = 255
        
        if frame.height == 0 or frame.width == 0 :
            img2d = 255*np.random.rand(224,224)
        else:
            img2d = np.reshape(img1d, (frame.height, frame.width))
        
        depth = cv2.resize(img2d, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        state = np.array(depth, dtype=np.float32)

        # return torch.from_numpy(state)
        return state

    #This will randomnly place the objects in the scene.
    def setObsRandom(self, state = [0, 0]):
        #check the obstacle names
        trans = random.uniform(-np.pi, np.pi)
        checks  = ['Wall', 'Obs']
        obj = self.client.simListSceneObjects(name_regex = '.*')
        Objs = [[i, self.client.simGetObjectPose(i)] for i in obj for check in checks if check in i ]
        for Obj in Objs:
            if 'Obs' in Obj[0]:
                Obj[1].position.y_val = random.uniform(-10,10)
                self.client.simSetObjectPose(Obj[0], Obj[1], teleport = True)
        # indxs = random.sample(range(0, 24), 24)#24 obstacles in the scene.
        # count = 0
        # for counter in range(24):
        #     if counter == 12:
        #         count +=1
        #     # print(count)
        #     Objs[indxs[counter]][1].position.x_val = math.cos(trans)*(15*(count//5) + -30 + random.uniform(-3,3) + state[0]) + math.sin(trans)*(15*(count%5) + -30 + random.uniform(-3,3) + state[1])
                                                         
        #     Objs[indxs[counter]][1].position.y_val = math.sin(trans)*(15*(count//5) + -30 + random.uniform(-3,3) + state[0]) - math.cos(trans)*(15*(count%5) + -30 + random.uniform(-3,3) + state[1]) 
                                                        
        #     self.client.simSetObjectPose(Objs[indxs[counter]][0], Objs[indxs[counter]][1], teleport = True)
            # count +=1
        Objs = [[i, self.client.simGetObjectPose(i)] for i in obj for check in checks if check in i ]
        
        # This the list of objs 
        return Objs

    # To set obstacles that are dynamic in nature
    def setObsDynamic(self):

        checks  = ['Dynamic']
        obj = self.client.simListSceneObjects(name_regex = '.*')
        Objs = [[i, self.client.simGetObjectPose(i)] for i in obj for check in checks if check in i ]
        count = 0
        for Obj in Objs:
            # Obj[1].position.x_val += self.delta_t*velocity(0,2)
            Obj[1].position.y_val += random.uniform(0,2)*self.delta_t*math.sin(time.time() + count*np.pi/2)
            # Obj[1].position.x_val = max(min(Obj[1].position.x_val, 10),-5)
            Obj[1].position.y_val = max(min(Obj[1].position.y_val, 8),-5) 
            self.client.simSetObjectPose(Obj[0], Obj[1], teleport = True)
            count += 1
        # indxs = random.sample(range(0, 24), 24)#24 obstacles in the scene.
        # count = 0
        # for counter in range(24):
        #     if counter == 12:
        #         count +=1
        #     # print(count)
        #     Objs[indxs[counter]][1].position.x_val = math.cos(trans)*(15*(count//5) + -30 + random.uniform(-3,3) + state[0]) + math.sin(trans)*(15*(count%5) + -30 + random.uniform(-3,3) + state[1])
                                                         
        #     Objs[indxs[counter]][1].position.y_val = math.sin(trans)*(15*(count//5) + -30 + random.uniform(-3,3) + state[0]) - math.cos(trans)*(15*(count%5) + -30 + random.uniform(-3,3) + state[1]) 
                                                        
        #     self.client.simSetObjectPose(Objs[indxs[counter]][0], Objs[indxs[counter]][1], teleport = True)
            # count +=1
        Objs = [[i, self.client.simGetObjectPose(i)] for i in obj for check in checks if check in i ]
        
        # This the list of objs 
        return Objs

    # Compute reward
    def compute_reward(self, image, quad_state, quad_vel):
        reward = 0
        # find the obstacles which are visible in the depth camera.
        distL = [1000]
        state = self.client.getMultirotorState()
        simga = 0.35
        h, w = image.shape
        # print(h,w)
        D = [1000]
        for Obj in self.Objs:
            if 'Wall' in Obj[0]:
                # print("This is a wall:", Obj[0])
                _, _, yaw = airsim.to_eularian_angles(Obj[1].orientation)
                m = -math.tan(yaw)
                norm_dist = abs(m*(quad_state.x_val - Obj[1].position.x_val) + (quad_state.y_val - Obj[1].position.y_val))/(1 + m**2)**0.5
                D.append(norm_dist)
            else:
                norm_dist = ((quad_state.x_val - Obj[1].position.x_val)**2 + (quad_state.y_val - Obj[1].position.y_val)**2)**0.5
                D.append(norm_dist)
            dist = np.linalg.norm(np.array([Obj[1].position.x_val,Obj[1].position.y_val,Obj[1].position.z_val])
                    -np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val]))
            distL.append(dist)
        
        offset = abs(quad_state.y_val*self.destination.x_val - quad_state.x_val*self.destination.y_val)/(self.destination.x_val**2 + self.destination.x_val**2)**0.5
        dest = np.linalg.norm(np.array([quad_state.x_val, quad_state.y_val, quad_state.z_val,]) - np.array([self.destination.x_val, self.destination.y_val, self.destination.z_val,]))
        init_dest = np.linalg.norm(np.array([quad_state.x_val, quad_state.y_val, quad_state.z_val,]))
        reward = min(1, (min(D) - 0.98)/(5 - 0.98))

        self.mindist = min(distL)
        # reward = -100*math.exp(-(min(distL)/5)**2)
        # if self.client.simGetCollisionInfo().has_collided:
            # reward = reward*10
        # if ((np.absolute(self.client.simGetVehiclePose().position.x_val)>30) or (np.absolute(self.client.simGetVehiclePose().position.y_val) > 30)):
        #     reward  = 0
        # print("Min dist:", self.mindist)
        # print("Reward:",reward)
        return np.float(reward)