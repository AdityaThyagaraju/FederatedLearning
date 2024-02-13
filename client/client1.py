import os
import socket
import pickle
import threading

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image

import kivy.app
import kivy.uix.button
import kivy.uix.label
import kivy.uix.boxlayout
import kivy.uix.textinput


class ClientApp(kivy.app.App):

    def __init__(self):
        super().__init__()
        if os.path.exists('mymodel.hdf5'):
            self.model_exists = True
        else:
            self.model_exists = False

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

            self.connect_btn.disabled = True
            self.recv_train_model_btn.disabled = False

        except BaseException as e:
            self.label.text = "Error Connecting to the Server"
            print("Error Connecting to the Server: {msg}".format(msg=e))

            self.connect_btn.disabled = False
            self.recv_train_model_btn.disabled = True

    def recv_train_model(self, *args):
        self.recv_train_model_btn.disabled = True
        recvThread = RecvThread(
            kivy_app = self, 
            buffer_size = 1024, 
            recv_timeout = 10
        )
        recvThread.start()
        self.detect_btn.disabled = False
        
    def detect(self, *args):
        self.detect_btn.disabled = True
        detectThread = DetectThread(
            kivy_app = self, 
            buffer_size = 1024, 
            recv_timeout = 10
        )
        detectThread.start()
        self.detect_btn.disabled = False

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
        
        if self.model_exists:
            self.detect_btn.disabled = False
        else:
            self.detect_btn.disabled = True

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
        self.box_layout.add_widget(self.detect_btn)
        self.box_layout.add_widget(self.close_socket_btn)
        self.box_layout.add_widget(self.label)

        return self.box_layout


class DetectThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.detect_dir = r"F:\vscode\Oral_Cancer_Dataset\detect"
        self.files = os.listdir(self.detect_dir)
        self.model = tf.keras.models.load_model('mymodel.hdf5')
    
    def run(self):
        for file in self.files:                
            img = image.load_img(file, target_size = (224, 224))
            img_arr = image.img_to_array(img)
            image = np.expand_dims(img_arr,axis=0)  
            predict = self.model.predict(image)
            
            if predict > 0.5:
                predict = "Normal"
            else:
                predict = "Cancer"
                
            print(f'{file}: {predict}'.format(file = file, predict = predict))

            
class RecvThread(threading.Thread):

    def __init__(self, kivy_app, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.kivy_app = kivy_app
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
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
                return None, 0
            except BaseException as e:
                print("Error While Receiving Data from the Server: {msg}.".format(msg=e))
                self.kivy_app.label.text = "Error While Receiving Data from the Server"
                return None, 0

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
            self.kivy_app.label.text = "Error Decoding the Client's Data"
            return None, 0

        return received_data, 1

    def run(self):

        subject = "request"
        model = None
        train_set = r"F:\vscode\Oral_Cancer_Dataset\client1"
        test_set = r"F:\vscode\Oral_Cancer_Dataset\test"
        test_datagen = None
        train_datagen = None
        target_size = None
        batch_size = None
        test_generator = None
        train_generator = None
        callback = None

        while True:
            data = {"subject": subject, "data": model}
            data_byte = pickle.dumps(data)

            self.kivy_app.label.text = "Sending a {subject} to the Server".format(
                subject=subject)
            try:
                self.kivy_app.soc.sendall(data_byte)
            except BaseException as e:
                self.kivy_app.label.text = "Error Connecting to the Server. The server might has been closed."
                print("Error Connecting to the Server: {msg}".format(msg=e))
                break

            self.kivy_app.label.text = "Receiving Reply from the Server"
            received_data, status = self.recv()

            if status == 0:
                self.kivy_app.label.text = "Nothing Received from the Server"
                break
            else:
                self.kivy_app.label.text = "New Message from the Server"

            subject = received_data["subject"]

            if subject == "model":
                if model:
                    model.save('mymodel.hdf5')
                else:
                    test_datagen = received_data['test_datagen']
                    train_datagen = received_data['train_datagen']
                    
                    target_size = received_data['target_size']
                    batch_size = received_data['batch_size'] if batch_size in received_data else 8
                    
                    train_generator = train_datagen.flow_from_directory(
                                        train_set,
                                        target_size = target_size,
                                        batch_size = batch_size,
                                        class_mode = 'binary'
                                    )
                    
                    test_generator = test_datagen.flow_from_directory(
                                        test_set,
                                        target_size = target_size,
                                        batch_size = batch_size,
                                        class_mode = 'binary'
                                    )
                    
                    mp = tf.keras.callbacks.ModelCheckpoint(
                        filepath='mymodel.hdf5', 
                        verbose=2, 
                        save_best_only=True
                    )
                    
                    es = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', 
                        min_delta=0.05, 
                        patience=3
                    )
                    
                    callback = [es, mp]
                
                model = received_data["data"]

                history = model.fit(
                            train_generator,
                            steps_per_epoch = 80,
                            epochs = 20,
                            validation_data = test_generator,
                            callbacks = callback
                        )
                
                print(history.history)
                
                model = tf.keras.models.load_model('mymodel.hdf5')
                
                file_path = 'mymodel.hdf5'
                if os.path.exists(file_path):
                    os.remove(file_path)
                        
            # elif subject == "done":
            #     self.kivy_app.label.text = "Model is Trained"
            #     break
            else:
                self.kivy_app.label.text = "Unrecognized Message Type: {subject}".format(
                    subject=subject)
                break

            subject = 'model'            


clientApp = ClientApp()
clientApp.title = "Client 1 App"
clientApp.run()
