import numpy as np
import random
import time
import os

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
directions = ['u','r','d','l']
dir_dic = {'u' : (-1,0), 'd' : (1,0), 'l' : (0,-1), 'r' : (0,1)}
reverse_dir_dic = {(-1, 0): 'u', (1, 0): 'd', (0, -1): 'l', (0, 1): 'r'}
person_dict = {'u' : '⬆️', 'd' : '⬇️', 'l' : '⬅️', 'r' : '➡️'}
char_vector = ['⬜','🌀','😈','✨','🏆','🏁','⬛','🦨', '✨', '✨', '🌀', '✨', '❌']
mapped_movements = {0: 'g', 1: 'v', 3: 'f',  11: 'l', 12: 'r', 13: 'b'}

def print_state(map, pos, direction):
  for i in range(map.shape[0]):
    for j in range(map.shape[1]):
      if (i, j) == pos:
        print(person_dict[direction], end='  ')
      else:
        print(char_vector[map[i][j]], end=' ')
    print()

def sense(map, pos, dir = -1):
  if dir == -1: return map[pos]

  shape = map.shape
  dir = dir_dic[dir]
  dest = (pos[0] + dir[0], pos[1] + dir[1])

  if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
    return 6

  return map[dest]

def senseVector(map, pos, dir, orientation):
  vector = np.zeros((len(orientation), 13), dtype=np.int32)
  for i in range(len(orientation)):
    if orientation[i] == 'f':
        vector[i][sense(map, pos, dir)] = 1
    elif orientation[i] == 'l':
        vector[i][sense(map, pos, directions[(directions.index(dir) - 1) % 4])] = 1
    elif orientation[i] == 'r':
        vector[i][sense(map, pos, directions[(directions.index(dir) + 1) % 4])] = 1
    elif orientation[i] == 'b':
        vector[i][sense(map, pos, directions[(directions.index(dir) + 2) % 4])] = 1
  return vector

def move(map, pos, dir, command, grabbed, win):
  if command == 11:
    dir = directions[directions.index(dir) - 1] 
  elif command == 12:
    dir = directions[(directions.index(dir) + 1) % 4]
  elif command == 13:
    dir = directions[(directions.index(dir) + 2) % 4]
  elif command == 3:
    shape = map.shape
    calc_dir = dir_dic[dir]
    dest = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
    if dest[0] >= 0 and dest[1] >= 0 and shape[0] > dest[0] and shape[1] > dest[1]:
      pos = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
      dir = reverse_dir_dic[calc_dir];
  elif command == 0:
    if map[pos] == 4:
      grabbed = True
  elif command == 1:
    if map[pos] == 5 and grabbed:
      win = True
  return pos, dir, grabbed, win, map[pos] == 2


def infer(vecInpSens: np.int32) -> int:
    return random.choice([0, 1, 3, 11, 12, 13])

def game(infer, movements):
    map = np.array(baseMap, copy=True)
    pos = (3,3)
    energy = 500
    dir = random.choice(directions)
    win, grabbed, dead = False, False, False
    len_movements = len(movements)
    while energy >= 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        vector = senseVector(map, pos, dir, movements)
        command = infer(vector)
        pos, dir, grabbed, win, dead = move(map, pos, dir, command, grabbed, win)
        print_state(map, pos, dir) # remover quando for usar para treino
        print('Energia: ', energy) # remover quando for usar para treino
        if grabbed: print('Pegou o ouro') # remover quando for usar para treino
        energy -= len_movements
        if dead:
            print('Morreu')
            break
        if win:
            print('Venceu')
            break
        time.sleep(0.1) # remover quando for usar para treino

if _name_ == '_main_':
    game(infer, ['f','l','r'])