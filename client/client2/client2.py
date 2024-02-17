import os
import socket
import threading
import tqdm
import dill

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing import image

import kivy.app
import kivy.uix.button
import kivy.uix.label
import kivy.uix.boxlayout
import kivy.uix.textinput

TRAIN_SET_DIR = r"client\client2\dataset"
TEST_SET_DIR = r"client\client2\test"

Model = None

class ClientApp(kivy.app.App):

    def __init__(self):
        super().__init__()
        self.ioobject = IOclass(
            kivy_app = self,
            timeout = 10
        )

    def create_socket(self, *args):
        self.soc = socket.socket(
            family = socket.AF_INET, 
            type = socket.SOCK_STREAM
        )
        
        self.label.text = "Socket Created"
        self.create_socket_btn.disabled = True
        self.connect_btn.disabled = False
        self.close_socket_btn.disabled = False

    def connect(self, *args):
        try:
            self.soc.connect((self.server_ip.text, int(self.server_port.text)))
            self.label.text = "Successful Connection to the Server"
            self.kivy_app=self

            self.connect_btn.disabled = True
            self.recv_train_model_btn.disabled = False
            self.get_updated_weights_btn.disabled=False

            message = {
                'subject': 'Request for architecture'
            }
            
            self.ioobject.send(message)
            received_data, status = self.ioobject.recv()
            
            if status == 0:
                self.kivy_app.label.text = "Nothing Received from the Server."
            else:
                global Model
                self.kivy_app.label.text = "Model received from server."
                
                model = received_data['model']
                optimizer = received_data['optimizer']
                loss = received_data['loss']
                metrics = received_data['metrics']
                
                Model = model
                
                Model.compile(
                    optimizer = optimizer, 
                    loss = loss, 
                    metrics = metrics 
                )
                self.detect_btn.disabled = False

        except BaseException as e:
            self.label.text = "Error Connecting to the Server"
            print("Error Connecting to the Server: {msg}".format(msg=e))

            self.connect_btn.disabled = False
            self.detect_btn.disabled = True
            self.recv_train_model_btn.disabled = True

    def recv_train_model(self, *args):
        self.recv_train_model_btn.disabled = True
        recv = Train(
            kivy_app = self, 
            ioobject = self.ioobject
        )
        recv.run()
        
    def detect(self, *args):
        self.detect_btn.disabled = True
        detect = Detect(kivy_app = self)
        detect.run()
        self.detect_btn.disabled = False


    def get_updated_weights(self, *args):
        update = Update(
            kivy_app = self,
            ioobject = self.ioobject
        )
        update.run()

    def close_socket(self, *args):
        self.soc.close()
        self.label.text = "Socket Closed"

        self.create_socket_btn.disabled = False
        self.connect_btn.disabled = True
        self.recv_train_model_btn.disabled = True
        self.close_socket_btn.disabled = True

    def build(self):
        self.create_socket_btn = kivy.uix.button.Button(text = "Create Socket")
        self.create_socket_btn.bind(on_press = self.create_socket)

        self.server_ip = kivy.uix.textinput.TextInput(
            hint_text = "Server IPv4 Address", 
            text = "localhost"
        )
        self.server_port = kivy.uix.textinput.TextInput(
            hint_text = "Server Port Number", 
            text = "10000"
        )

        self.server_info_boxlayout = kivy.uix.boxlayout.BoxLayout(orientation = "horizontal")
        self.server_info_boxlayout.add_widget(self.server_ip)
        self.server_info_boxlayout.add_widget(self.server_port)

        self.connect_btn = kivy.uix.button.Button(
            text = "Connect to Server", 
            disabled = True
        )
        self.connect_btn.bind(on_press = self.connect)

        self.recv_train_model_btn = kivy.uix.button.Button(
            text = "Receive & Train Model", 
            disabled = True
        )
        self.recv_train_model_btn.bind(on_press = self.recv_train_model)
        
        self.detect_btn = kivy.uix.button.Button(
            text = "Detect Oral Cancer"
        )
        self.detect_btn.bind(on_press = self.detect)
        self.detect_btn.disabled=True

        self.get_updated_weights_btn = kivy.uix.button.Button(
            text = "Get Updated Model"
        )
        self.get_updated_weights_btn.bind(on_press = self.get_updated_weights)
        self.get_updated_weights_btn.disabled=True

        self.close_socket_btn = kivy.uix.button.Button(
            text = "Close Socket", 
            disabled = True
        )
        self.close_socket_btn.bind(on_press = self.close_socket)

        self.label = kivy.uix.label.Label(text = "Socket Status")

        self.box_layout = kivy.uix.boxlayout.BoxLayout(orientation = "vertical")
        self.box_layout.add_widget(self.create_socket_btn)
        self.box_layout.add_widget(self.server_info_boxlayout)
        self.box_layout.add_widget(self.connect_btn)
        self.box_layout.add_widget(self.recv_train_model_btn)
        self.box_layout.add_widget(self.get_updated_weights_btn)
        self.box_layout.add_widget(self.detect_btn)
        self.box_layout.add_widget(self.close_socket_btn)
        self.box_layout.add_widget(self.label)

        return self.box_layout

class IOclass:
    def __init__(self, kivy_app, timeout):
        self.kivy_app = kivy_app
        self.recv_timeout = timeout
        
    def recv(self):
        data_size_bytes = self.kivy_app.soc.recv(8)
        data_size = int.from_bytes(data_size_bytes, byteorder='big')

        data_received = b""
        received_data = b""

        try:
            self.kivy_app.soc.settimeout(self.recv_timeout)
            
            with tqdm.tqdm(total = data_size, unit = 'B', unit_scale = True, desc = 'Receiving Data from Server') as pbar:
                while len(data_received) < data_size:
                    chunk = self.kivy_app.soc.recv(min(data_size - len(data_received), 1024*1024))
                    if not chunk:
                        break
                    data_received += chunk
                    pbar.update(len(chunk))

        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(
                recv_timeout = self.recv_timeout))
            self.kivy_app.label.text = "{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(
                recv_timeout = self.recv_timeout)
            return None, 0
        
        except BaseException as e:
            print("Error While Receiving Data from the Server: {msg}.".format(msg=e))
            self.kivy_app.label.text = "Error While Receiving Data from the Server"
            return None, 0

        try:
            received_data = dill.loads(data_received)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
            self.kivy_app.label.text = "Error Decoding the Data"
            return None, 0
        
        print('\n')

        return received_data, 1
    
    def send(self, message):
        data_byte = dill.dumps(message)
        data_size = len(data_byte)
        
        try:
            self.kivy_app.soc.sendall(data_size.to_bytes(8, byteorder='big'))
            chunk_size = 1024*1024
            bytes_sent = 0
            
            with tqdm.tqdm(total = data_size, unit = 'B', unit_scale = True, desc = 'Sending Data to Server') as pbar:
                while bytes_sent < data_size:
                    chunk = data_byte[bytes_sent:bytes_sent + chunk_size]
                    self.kivy_app.soc.sendall(chunk)
                    bytes_sent += len(chunk)
                    pbar.update(len(chunk))
            
        except BaseException as e:
            self.kivy_app.label.text = "Error Connecting to the Server. The server might has been closed."
            print("Error Connecting to the Server: {msg}".format(msg=e))
        
        print('\n')

class Update:

    def __init__(self, kivy_app, ioobject):
        global Model
        self.kivy_app = kivy_app
        self.ioobject = ioobject
        self.model = Model
    
    def run(self):
        message = {
            "subject": 'Request for weights'
        }
        
        self.ioobject.send(message)
        received_data, status = self.ioobject.recv()
        
        if status == 0:
            self.kivy_app.label.text = "Nothing Received from the Server."
        else:
            self.model.set_weights(received_data['weights'])
            self.kivy_app.label.text = "Model Updated."
            
            
class Detect:
    
    def __init__(self, kivy_app):
        global Model
        self.kivy_app = kivy_app
        self.directory = r"client\detect"
        self.model = Model
    
    def run(self):
        filename = '7.jpg'
        img_path = os.path.join(self.directory, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        y_pred = self.model.predict(img_array)
        
        if y_pred < 0.5:
            self.kivy_app.label.text = "Cancer Detected."
        else:
            self.kivy_app.label.text = "No Cancer Detected."

class Train:

    def __init__(self, kivy_app, ioobject):
        self.kivy_app = kivy_app
        self.ioobject = ioobject

    def run(self):
        message = {
            'subject': 'Request for weights and image generators'
        }        
        
        self.ioobject.send(message)
        received_data, status = self.ioobject.recv()

        if status == 0:
            self.kivy_app.label.text = "Nothing Received from the Server."
        else:
            global Model
            self.kivy_app.label.text = "Received weights from the Server."
            
            model = Model
            
            train_datagen = received_data['train_datagen']
            test_datagen = received_data['test_datagen']
            target_size = received_data['target_size']
            batch_size = received_data['batch_size']   
            model_weights = received_data['weights']
            model.set_weights(model_weights)
            
            train_generator = train_datagen.flow_from_directory(
                                TRAIN_SET_DIR,
                                target_size = target_size,
                                batch_size = batch_size,
                                class_mode = 'binary'
                            )
            
            test_generator = test_datagen.flow_from_directory(
                                TEST_SET_DIR,
                                target_size = target_size,
                                batch_size = batch_size,
                                class_mode = 'binary'
                            )
            
            model.fit(
                train_generator,
                steps_per_epoch = 46,
                epochs = 1,
                validation_data = test_generator
            )
            
            acc = model.evaluate(test_generator)[2] * 100
            self.kivy_app.label.text = "Trained model has an accuracy of {:.2f}%.".format(acc)

            Model = model
            message = {
                'subject': 'Weights for update',
                'weights': model.get_weights()
            }
            self.ioobject.send(message)
            
            received_data, status = self.ioobject.recv()
            
            if status == 0:
                self.kivy_app.label.text = "Nothing Received from the Server."
            else:
                Model.set_weights(received_data['weights'])
            
            
clientApp = ClientApp()
clientApp.title = "Client 2 App"
clientApp.run()
