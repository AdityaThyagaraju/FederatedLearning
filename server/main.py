import socket
import tensorflow as tf
import pickle
import threading
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications.vgg16 import VGG16
import time


ROWS = 224
COLS = 224
BATCHSIZE = 50
BUFFERSIZE = 1024
TIMEOUT = 50
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
        threading.Thread.__init__(self)
        self.connection = connection
        self.bufferSize = bufferSize
        self.timeOut = timeOut
        self.app = app
    
    def reply(self,message):
        try:
            data = pickle.dumps(message)
            print(message)
            try:
                self.connection.send(data)
            except BaseException as e:
                print("Connection error: {e}".format(e))
        except BaseException as e:
            print("Inappropriate message, generated exception {e}".format(e))
    
    def reqHandler(self,req):
        serverModel = self.app.getInstance()
        print(req)
        if type(req)!=type(dict()):
            print("Error message format not supported")
            return
        if "subject" not in req or "data" not in req:
            print("Incomplete message")
            return
        if req["subject"]=="request" or req["subject"]=="model":
            message = {
                "subject":"model",
                "data":serverModel.to_json(),
            }
            self.reply(message)
            message = {
                "subject":"weights",
                "data":serverModel.get_weights(),
            }
        else:
            print("Unrecognized request subject")
            
    def run(self):
        data = b""
        try:
            data += self.connection.recv(self.bufferSize)
            if data!=b"":
                try:
                    data = pickle.loads(data)
                    self.reqHandler(data)
                except BaseException as e:
                    print(e)
                    
            elif data==b"": 
                print("Empty request")
        except BaseException as e:
            print("Connection error: {e}".format(e=e))

class IOthread(threading.Thread):
    def __init__(self,app):
        threading.Thread.__init__(self)
        self.app=app
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    def run(self):
        self.soc.bind(("localhost",10000))
        self.soc.listen()
        print("Server started at PORT:10000")
        while True:
            try:
                connection, client_info = self.soc.accept()
                socket_thread = AppInterface(connection=connection,app=self.app,bufferSize=1024,timeOut=10)
                socket_thread.start()
                print("Client {client_info} is successfully connected".format(client_info=client_info))
            except BaseException as e:
                self.soc.close()
                print(e)
                break

app = App(ROWS,COLS)
iothread = IOthread(app)
iothread.start()
    
    