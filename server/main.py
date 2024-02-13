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
        received_data = b""
        while True: 
            try:
                self.kivy_app.soc.settimeout(self.recv_timeout)
                received_data += self.kivy_app.soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    break
                except BaseException:
                    print("Could not receive the complete data from server.")
                    self.kivy_app.label.text = "Could not receive the complete data from server."
                    pass

            except socket.timeout:
                print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(
                    recv_timeout = self.recv_timeout))
                self.kivy_app.label.text = "{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(
                    recv_timeout = self.recv_timeout)
                
            except BaseException as e:
                print("Error While Receiving Data from the Server: {msg}.".format(msg=e))
                self.kivy_app.label.text = "Error While Receiving Data from the Server"
                

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
            self.kivy_app.label.text = "Error Decoding the Client's Data"

class IOthread(threading.Thread):
    def __init__(self,app):
        threading.Thread.__init__(self)
        self.app=app
    def run(self):
        print("server started at port no 5000")
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 5000))
                s.listen()
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected by {addr}")
                        client=AppInterface(conn,BUFFERSIZE,TIMEOUT,app)
                        client.start()
                except BaseException as e:
                    print(e)

app = App(ROWS,COLS)
iothread = IOthread(app)
iothread.start()
    
    