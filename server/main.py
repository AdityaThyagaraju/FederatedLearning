import socket
import threading
import dill
import tqdm
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing import image


ROWS = 224
COLS = 224
TIMEOUT = 50
TRAIN_SET_DIR = r'server\data'


class App:
    def __init__(self,rows,cols):
        
        self.base_for_model = tf.keras.applications.VGG16(weights = 'imagenet', input_shape = (224,224,3), include_top = False)
        for layer in self.base_for_model.layers:
            layer.trainable = False
    
        model = None
        if not os.path.exists('server\\vgg16.keras'):
            model = Sequential()
            model.add(self.base_for_model) 
            model.add(GaussianNoise(0.25))
            model.add(GlobalAveragePooling2D())
            model.add(Dense(512, activation = 'relu'))
            model.add(BatchNormalization())
            model.add(Dense(1, activation = 'sigmoid'))
            
            self.adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
            model.compile(
                optimizer = self.adam, 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy','Precision','Recall','AUC']
            )

            #server Training 
            self.batch_size = 16
            self.target_size = (224,224)
            train_datagen = image.ImageDataGenerator(
                rotation_range = 15,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                width_shift_range = 0.1,
                height_shift_range = 0.1
            )
            train_generator = train_datagen.flow_from_directory(
                                    TRAIN_SET_DIR,
                                    target_size = self.target_size,
                                    batch_size = self.batch_size,
                                    class_mode = 'binary'
                                )
            
            model.fit(
                    train_generator,
                    steps_per_epoch = 15,
                    epochs = 1,
                )
            model.save('server\\vgg16.keras')
        else:
            model = tf.keras.models.load_model('server\\vgg16.keras')
            
        self.model=model

        
    def updateWeights(self,newWeights):
        self.model.set_weights(newWeights)
        
        
    def recAverage(self,l1,l2):
        avg = []
        if type(l1)==type(l2):
            if type(l1)==type(list()):
                for i in range(len(l1)):
                    avg.append(self.recAverage(l1[i],l2[i]))
                return avg
            return (l1+l2)/2
    
    def federatedAverage(self,client_layer_weights):
        server_layer_weights = self.model.get_weights()
        averaged_weights = self.recAverage(client_layer_weights,server_layer_weights)
        self.updateWeights(averaged_weights)
        print("Federated averaging completed, Updated weights successfully")
        
    def get_weights(self):
        return self.model.get_weights()
    
    def image_generators(self):
        train_datagen = image.ImageDataGenerator(
            rotation_range = 15,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            width_shift_range = 0.1,
            height_shift_range = 0.1
        )
        
        test_datagen = image.ImageDataGenerator(    
            rotation_range = 15,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            width_shift_range = 0.1,
            height_shift_range = 0.1
        )
        
    
        
        return train_datagen, test_datagen, self.target_size, self.batch_size
    
    def model_architecture(self):        
        optimizer = self.adam
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 'Precision', 'Recall', 'AUC']
        
        return self.model, optimizer, loss, metrics
    
class AppInterface(threading.Thread):
    def __init__(self,connection,bufferSize,timeOut,app,broadCastFunc=None):
        threading.Thread.__init__(self)
        self.connection = connection
        self.bufferSize = bufferSize
        self.timeOut = timeOut
        self.app = app
        self.broadCastFunc = broadCastFunc
    
    def reply(self, message):
        data_pickle = dill.dumps(message)
        data_size = len(data_pickle)
        
        self.connection.sendall(data_size.to_bytes(8, byteorder = 'big'))
        
        chunk_size = 1024*1024
        bytes_sent = 0
        
        try:
            with tqdm.tqdm(total = data_size, unit = 'B', unit_scale = True, desc = 'Sending Data to Client') as pbar:
                while bytes_sent < data_size:
                    chunk = data_pickle[bytes_sent:bytes_sent + chunk_size]
                    self.connection.sendall(chunk)
                    bytes_sent += len(chunk)
                    pbar.update(len(chunk))
        except BaseException as e:
            print("Error Connecting to the client: {msg}".format(msg=e))
        print('\n')
    
    def reqHandler(self,req):      
        lock = threading.Lock()
        lock.acquire(blocking=True)  
        if "subject" not in req:
            print("Incomplete message")
        else:
            if req['subject'] == 'Request for architecture':
                model, optimizer, loss, metrics = self.app.model_architecture()
                message = {
                    'model': model,
                    'optimizer': optimizer,
                    'loss': loss,
                    'metrics': metrics,
                }
                self.reply(message)
            
            elif req['subject'] == 'Request for weights and image generators':
                train_datagen, test_datagen, target_size, batch_size = self.app.image_generators()
                weights = self.app.get_weights()
                message = {
                    'train_datagen': train_datagen, 
                    'test_datagen': test_datagen, 
                    'target_size': target_size, 
                    'batch_size': batch_size,
                    'weights': weights
                }
                self.reply(message)
                            
            elif req['subject'] == 'Request for weights':
                message = {
                    'weights': self.app.get_weights()
                }
                self.reply(message)
                
            elif req['subject']=="Weights for update":
                clientWeights = req['weights']
                self.app.federatedAverage(clientWeights) 
                if self.broadCastFunc is not None:
                    message = {
                    'weights': self.app.get_weights()
                    }
                    self.broadCastFunc(message)
            else:
                print("Unrecognized subject")            
        lock.release()         
                
    def run(self):
        while True:
            data_size_bytes = self.connection.recv(8)
            if not data_size_bytes:
                continue
            data_size = int.from_bytes(data_size_bytes, byteorder='big')

            data_received = b""
            received_data = b""

            try:
                with tqdm.tqdm(total = data_size, unit = 'B', unit_scale = True, desc = 'Receiving Data from Client') as pbar:
                    while len(data_received) < data_size:
                        chunk = self.connection.recv(min(data_size - len(data_received), 1024*1024))
                        if not chunk:
                            break
                        data_received += chunk
                        pbar.update(len(chunk))
            
            except BaseException as e:
                print("Error While Receiving Data from the Client: {msg}.".format(msg=e))
                continue

            try:
                received_data = dill.loads(data_received)
            except BaseException as e:
                print("Error Decoding the Data: {msg}.\n".format(msg=e))
                continue
            print('\n')

            self.reqHandler(received_data)
            
class Sock:
    def __init__(self,app):
        self.channelList = []
        self.app = app
        
    def getSocketThread(self,connection):
        socket_thread = AppInterface(connection = connection, app = self.app, bufferSize = 1024, timeOut = 10, broadCastFunc=self.broadcast)
        self.channelList.append(socket_thread)
        return socket_thread
    
    def broadcast(self,message):
        for channel in self.channelList:
            channel.reply(message)

class IOthread(threading.Thread):
    def __init__(self,app):
        threading.Thread.__init__(self)
        self.app = app
        self.sockObj = Sock(app)
    def run(self):
        print("Server started at PORT:10000")
        while True:
            try:
                soc = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
                soc.bind(("localhost",10000))
                soc.listen()
                connection, client_info = soc.accept()
                socket_thread = self.sockObj.getSocketThread(connection=connection)
                socket_thread.start()
                print("Client {client_info} is successfully connected.\n".format(client_info = client_info))
            except BaseException as e:
                soc.close()
                print(e)
                break

app = App(ROWS,COLS)
iothread = IOthread(app)
iothread.start()
    
    