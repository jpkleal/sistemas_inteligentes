import numpy as np
import random

'''
# [ 0] = ⬜"inp_nothing"          **  sente nada (tem nada na casa requisitada)
# [ 1] = 🌀"inp_breeze"           **  sente brisa (uma casa antes de um buraco)
# [ 2] = 😈"inp_danger"           **  sente perigo (casa requisitada/atual tem um Wumpus ou um buraco - morre)
# [ 3] = ✨"inp_flash"            **  sente flash (uma casa antes do ouro ele vê o brilho do ouro)
# [ 4] = 🏆"inp_goal"             **  sente meta (casa requisitada/atual tem ouro - reward, que é a meta)
# [ 5] = 🏁"inp_initial"          **  sente início (casa requisitada/atual é o ponto de partida/saída)
# [ 6] = ⬛"inp_obstruction"      **  sente obstução (mandou request,d e vem obstrução é porque vai colidir em 'd')
# [ 7] = 🦨"inp_stench"           **  sente fedor (uma casa antes de um Wumpus)
# [ 8] = ✨"inp_bf"               **  sente brisa/flash (na casa 'd' tem sinais de brisa e flash)
# [ 9] = ✨"inp_bfs"              **  sente brisa/flash/stench (na casa 'd' tem brisa + flash + fedor)
# [10] = 🌀"inp_bs"               **  sente brisa/stench (na casa 'd' tem brisa + fedor)
# [11] = ✨"inp_fs"               **  sente flash/stench (na casa 'd' tem flash + fedor)
# [12] = ❌"inp_boundary"         **  colidiu com borda (mandou mover forward,d e colidiu com a borda do EnviSim)

# [ 0] = "out_act_grab"         **  ação de pegar/agarrar o ouro (reward)
# [ 1] = "out_act_leave"        **  ação de deixar a caverna (no mesmo local de partida)
# [ 3] = "out_mov_forward"      **  ação de mover adiante
# [11] = "out_rot_left"         **  ação de rotacionar esq.{"rotate":["left",2]}=90°; {"rotate":["left",1]}=45°
# [12] = "out_rot_right"        **  ação de rotacionar esq.{"rotate":["right",2]}=90°; {"rotate":["right",1]}=45°
# [13] = "out_rot_back"         **  ação de rotacionar back.{"rotate":["back",0]}={"rotate":["right",4]}=180°
'''

baseMap = np.array([
    [0,7,0,1,2,1,0,0,1,0],
    [7,2,7,0,1,0,0,1,2,1],
    [0,7,0,0,0,0,0,1,1,0],
    [0,0,0,5,0,0,1,2,1,0],
    [0,1,0,0,0,0,0,10,0,0],
    [1,2,1,0,0,0,7,2,7,0],
    [0,1,7,0,0,0,1,11,0,0],
    [0,7,2,7,0,1,2,4,3,0],
    [0,0,7,0,0,0,1,3,0,0],
    [0,0,0,0,0,0,0,0,0,0]
])

person_dict = {'u' : '⬆️', 'd' : '⬇️', 'l' : '⬅️', 'r' : '➡️'}
char_vector = ['⬜','🌀','😈','✨','🏆','🏁','⬛','🦨', '✨', '✨', '🌀', '✨', '❌']

def print_state(map, pos, direction):
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if (i, j) == pos:
                print(person_dict[direction], end='  ')
            else:
                print(char_vector[map[i][j]], end=' ')
        print()

class Env:
    def __init__(self):
        self.directions = ['u', 'r', 'd', 'l']
        self.dir_dic = {
            'u': (-1, 0),
            'd': (1, 0),
            'l': (0, -1),
            'r': (0, 1)
        }
        self.movements = ['f', 'l', 'r']
        self.reverse_dir_dic = {
            (-1, 0): 'u',
            (1, 0): 'd',
            (0, -1): 'l',
            (0, 1): 'r'
        }

        self.map: np.ndarray = None
        self.energy: int = None
        self.grabbed: bool = None
        self.dir: str = None
        self.pos: tuple[int, int] = None

    def start(self, table, energy=500):
        self.map = table
        self.energy = energy
        self.grabbed = False
        self.dir = random.choice(self.directions)

        x, y = np.where(self.map == 5)
        self.pos = (x[0], y[0])

        return self.sense_vector(self.movements), False

    def sense_vector(self, orientation):
        vector = np.zeros((len(orientation), 13), dtype=np.int32)
        for i, j in enumerate(orientation):
            if j == 'f':
                vector[i, self.sense(self.dir)] = 1
            elif j == 'l':
                vector[i, self.sense(self.directions[(self.directions.index(self.dir) - 1) % 4])] = 1
            elif j == 'r':
                vector[i, self.sense(self.directions[(self.directions.index(self.dir) + 1) % 4])] = 1
            elif j == 'b':
                vector[i, self.sense(self.directions[(self.directions.index(self.dir) + 2) % 4])] = 1
        self.energy -= len(self.movements)

        return vector

    def sense(self, dir=None):
        if dir is None:
            return self.map[self.pos]

        shape = self.map.shape
        dir = self.dir_dic[dir]
        dest = (self.pos[0] + dir[0], self.pos[1] + dir[1])

        if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
            return 6

        return self.map[dest]

    def move(self, command):
        reward = 0
        end = False

        # turn
        if command == 11:
            self.dir = self.directions[self.directions.index(self.dir) - 1]
        elif command == 12:
            self.dir = self.directions[(self.directions.index(self.dir) + 1) % 4]
        elif command == 13:
            self.dir = self.directions[(self.directions.index(self.dir) + 2) % 4]

        # forward
        elif command == 3:
            shape = self.map.shape
            calc_dir = self.dir_dic[self.dir]
            dest = (self.pos[0] + calc_dir[0], self.pos[1] + calc_dir[1])
            if 0 <= dest[0] < shape[0] and 0 <= dest[1] < shape[1]:
                self.pos = (self.pos[0] + calc_dir[0], self.pos[1] + calc_dir[1])
                self.dir = self.reverse_dir_dic[calc_dir]

        # grab
        elif command == 0:
            if self.map[self.pos] == 4:
                if not self.grabbed:
                    reward = 50
                self.grabbed = True

        # leave
        elif command == 1:
            if self.map[self.pos] == 5 and self.grabbed:
                reward = 50
                end = True

        # check death
        if self.map[self.pos] == 2:
            reward = -20
            end = True

        return reward, end

    def step(self, action):
        reward, done = self.move([0, 1, 3, 11, 12, 13][action])
        state = self.sense_vector(self.movements)

        if self.energy <= 0:
            done = True

        return state, reward, done
