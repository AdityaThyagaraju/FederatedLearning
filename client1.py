import socket
import pickle
import numpy
import threading

import pygad
import pygad.nn
import pygad.gann

import kivy.app
import kivy.uix.button
import kivy.uix.label
import kivy.uix.boxlayout
import kivy.uix.textinput

class ClientApp(kivy.app.App):
    
    def __init__(self):
        super().__init__()

    def create_socket(self, *args):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
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
        global GANN_instance

        self.recv_train_model_btn.disabled = True
        recvThread = RecvThread(kivy_app=self, buffer_size=1024, recv_timeout=10)
        recvThread.start()

    def close_socket(self, *args):
        self.soc.close()
        self.label.text = "Socket Closed"

        self.create_socket_btn.disabled = False
        self.connect_btn.disabled = True
        self.recv_train_model_btn.disabled = True
        self.close_socket_btn.disabled = True

    def build(self):
        self.create_socket_btn = kivy.uix.button.Button(text="Create Socket")
        self.create_socket_btn.bind(on_press=self.create_socket)

        self.server_ip = kivy.uix.textinput.TextInput(hint_text="Server IPv4 Address", text="localhost")
        self.server_port = kivy.uix.textinput.TextInput(hint_text="Server Port Number", text="10000")

        self.server_info_boxlayout = kivy.uix.boxlayout.BoxLayout(orientation="horizontal")
        self.server_info_boxlayout.add_widget(self.server_ip)
        self.server_info_boxlayout.add_widget(self.server_port)

        self.connect_btn = kivy.uix.button.Button(text="Connect to Server", disabled=True)
        self.connect_btn.bind(on_press=self.connect)

        self.recv_train_model_btn = kivy.uix.button.Button(text="Receive & Train Model", disabled=True)
        self.recv_train_model_btn.bind(on_press=self.recv_train_model)

        self.close_socket_btn = kivy.uix.button.Button(text="Close Socket", disabled=True)
        self.close_socket_btn.bind(on_press=self.close_socket)

        self.label = kivy.uix.label.Label(text="Socket Status")

        self.box_layout = kivy.uix.boxlayout.BoxLayout(orientation="vertical")
        self.box_layout.add_widget(self.create_socket_btn)
        self.box_layout.add_widget(self.server_info_boxlayout)
        self.box_layout.add_widget(self.connect_btn)
        self.box_layout.add_widget(self.recv_train_model_btn)
        self.box_layout.add_widget(self.close_socket_btn)
        self.box_layout.add_widget(self.label)

        return self.box_layout

class RecvThread(threading.Thread):

    def __init__(self, kivy_app, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.kivy_app = kivy_app
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True: # str(received_data)[-2] != '.':
            try:
                self.kivy_app.soc.settimeout(self.recv_timeout)
                received_data += self.kivy_app.soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop and a break statement should be excuted.
                    break
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

            except socket.timeout:
                print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=self.recv_timeout))
                self.kivy_app.label.text = "{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(recv_timeout=self.recv_timeout)
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

        subject = "echo"
        GANN_instance = None
        best_sol_idx = -1

        while True:
            data = {"subject": subject, "data": GANN_instance, "best_solution_idx": best_sol_idx}
            data_byte = pickle.dumps(data)

            self.kivy_app.label.text = "Sending a Message of Type {subject} to the Server".format(subject=subject)
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
            print(received_data)
            print(type(received_data))
            subject = received_data["subject"]
            if subject == "model":
                GANN_instance = received_data["data"]
            elif subject == "done":
                self.kivy_app.label.text = "Model is Trained"
                break
            else:
                self.kivy_app.label.text = "Unrecognized Message Type: {subject}".format(subject=subject)
                print(received_data)
                break

            print(GANN_instance)

            # subject = "model"
            # best_sol_idx = ga_instance.best_solution()[2]

clientApp = ClientApp()
clientApp.title = "Client App"
clientApp.run()