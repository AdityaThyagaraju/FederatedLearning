import socket
import tensorflow as tf
import pickle
import threading
import image
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications.vgg16 import VGG16
import time

BATCHSIZE = 50
class App:
    def __init__(self,rows,cols):
        self.train_datagen = image.ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1)
        
        self.test_datagen= image.ImageDataGenerator(rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2, 
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1)
        
        self.size = (rows,cols)
        self.batchSize = BATCHSIZE
        
        base_for_model = tf.keras.applications.VGG16(weights='imagenet', input_shape=(224,224,3), include_top=False)
        self.baseModel = base_for_model
        
    def updateWeights(self,newWeights):
        self.baseModel.set_weights(newWeights)
    
    def federatedAverage(self,clientModel):
        averaged_weights = []
        server_layer_weights = self.model.get_weights()
        client_layer_weights = clientModel.get_weights()
        averaged_weights.append(server_layer_weights)
        averaged_weights.append(client_layer_weights)
        for i in range(len(averaged_weights[0])):
            layer_weights = [model[i] for model in averaged_weights]
            averaged_weights[i] = np.mean(layer_weights, axis=0)
        self.updateWeights(averaged_weights)
        
    def getInstance(self):
        model = Sequential()
        model.add(self.baseModel) 
        model.add(GaussianNoise(0.25)) 
        model.add(GlobalAveragePooling2D()) 
        model.add(Dense(512, activation = 'relu'))
        model.add(BatchNormalization()) 
        model.add(Dense(1, activation='sigmoid'))
        return model
    
class AppInterface(threading.Thread):
    def __init__(self,connection,bufferSize,timeOut,app):
        self.connection = connection
        self.bufferSize = bufferSize
        self.timeOut = timeOut
        self.app = app
        
    def reqHandler(self,req):
        if type(req)!=dict:
            print("Error message format not supported")
            return
        if "subject" not in req or "data" not in req:
            print("Incomplete message")
            return
        if req["subject"]=="request":
            message = {
                "subject":"model",
                "data":self.app.getInstance(),
                "test_datagen":self.app.test_datagen,
                "train_datagen":self.app.train_datagen,
                "target_size":self.app.size,
                "batch_size":self.app.batchSize
            }
            self.reply(message)
        elif req["subject"]=="model":
            model = req["data"]
            self.app.federatedAverage(model)
            message = {
                "subject":"model",
                "data":self.app.getInstance()
            }
            self.reply(message)
        else:
            print("Unrecognized request subject")
            
    def reply(self,message):
        try:
            data = pickle.dumps(message)
            try:
                self.connection.send(data)
            except BaseException as e:
                print("Connection error: {e}".format(e))
        except BaseException as e:
            print("Inappropriate message, generated exception {e}".format(e))
            
    def run(self):
        request = b""
        recv_start_time = None
        while True:
            try:
                data = self.connection.recv(self.bufferSize)
                if request!=b"":
                    try:
                        request+=data
                        request = pickle.loads(request)
                        self.reqHandler(request)
                    except BaseException as e:
                        if time.time()-recv_start_time>self.timeOut:
                            print("Request message timeout")
                            request = b""
                            recv_start_time = None
                elif data!=b"":
                    request += data
                    recv_start_time = time.time() 
            except BaseException as e:
                print("Connection error: {e}".format(e))

class IOthread:
    def __init__(self,app):
        pass
    def run(self):
        pass
    
    
    