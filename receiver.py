import socket
import time

import numpy as np
from threading import Thread


class Receiver(Thread):
    def __init__(self):
        super().__init__()
        self.end = False
        self.last_reward = None
        self.reward = 0
        self.state = None
        self.__new_state = False
        self.__new_reward = False
        self.__new_action = False
        self.__action = None
        self.__observed = False

        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect(("127.0.0.1", 80))

    def __get_state(self, bin):
        state_ = []

        for s in bin.decode("ascii").split(','):
            state_.append(int(s))
            if int(s) > 8:
                print(bin)

        return tuple(state_)

    def run(self):
        lt = time.time()
        while True:
            buff = self.soc.recv(1024)
            try:
                label = chr(buff[0])
                data = bytes(buff[1:])
            except:
                print(buff)

            if label == "s":
                self.__new_state = True
                self.state = self.__get_state(data)
                # print("state", self.state)
                self.soc.send(b'0')

            elif label == "r":
                self.__new_reward = True
                nr = int(data.decode("ascii"))
                if self.reward == -500:
                    print("bug")

                if self.reward == 50 or nr == 50:
                    print("---------- GOT REWARD -------------", nr, "old", self.reward)
                # print("reward", self.reward)

                self.reward = nr

            elif label == "e":
                self.end = True
                self.last_reward = int(data.decode("ascii"))
                self.soc.send(b'0')
            else:
                print(label, data)

            if self.__new_state and self.__new_reward:
                while not self.__new_action:
                    pass

                # print("action", self.__action, self.__new_action)
                lt = time.time()
                time.sleep(0.1)
                self.soc.sendall(self.__action.to_bytes(1, "big"))
                self.__new_state = False
                self.__new_reward = False
                self.__new_action = False
                self.__observed = False

            if time.time() - lt > 1:
                lt = time.time()
                # print("action", self.__action, self.__new_action)
                self.soc.send(self.__action.to_bytes(1, "big"))
    def act(self, action):
        self.__new_action = True
        self.__action = int(action)

    def observe(self):
        while self.__observed:
            time.sleep(0.1)
        self.__observed = True
        return self.state, self.reward

    def check_end(self):
        if self.end:
            self.end = False
            return True, self.last_reward
        return False, None

if __name__ == "__main__":
    r = Receiver()
    r.start()
    input()
