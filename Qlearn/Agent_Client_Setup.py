# Este arquivo contém as variáveis e constantes de configuração inicial usadas no arquivo principal
import socket
from enum import Enum
import json
import numpy as np
import sys

# ---.---  variáveis e constantes ---.---

# Isto define a variável 'Stt' e a preenche com os estados da Máquina de Estados Finitos (enum)
class Stt(Enum):
    BEGIN = 1  # estado inicial da FSM no programa principal
    RECEIVING = 2  # estado quando recebe/espera uma mensagem do EnviSim
    INTERPRETING = 3  # estado quando interpreta uma mensagem recebida
    SENDING = 4  # estado quando envia uma mensagem para o EnviSim
    DECIDING = 5  # estado para tomar um índice_out (após interpretar uma mensagem)
    REQUESTING = 6  # estado para solicitar informações do EnviSim
    RESTARTING = 7  # estado da FSM que reinicia uma jornada dentro do EnviSim
    EXCEPTIONS = 8  # estado para atender mensagens de grande interesse: morreu, sucesso, não pode
    ERRORS = 9  # estado para lidar com erros
    FOR_TESTS = 10  # estado para qualquer teste, se necessário (pode ser excluído)


# Enumeração dos estados da submáquina de estados (subFSM)
class SubStt(Enum):
    RES = 1  # estado depois de reiniciar o cenário (começa a solicitar informações)
    START = 2  # estado para começar a solicitar informações
    ASK = 3  # estado enquanto a subFSM está solicitando informações do EnviSim
    SAVE = 4  # estado para salvar respostas após solicitar informações
    CMD = 5  # estado para enviar um comando para o EnviSim
    CNT = 6  # estado para decidir se a sessão terminou ou continua
    WAITRQ = 7  # estado para AGUARDAR uma resposta após uma solicitação de informação
    WAITCM = 8  # estado para AGUARDAR uma resposta de feedback após um comando

sttMM = Stt.BEGIN  # inicia sttMM no estado BEGIN (no loop principal)
sttSUBfsm = SubStt.RES  # inicia o status da subFSM como SubStt.RES


# --- Essa lista (InfoReqSeq) contém a sequência de REQUESTS a ser feita p/ o EnviSim
# uma por uma, o agente envia/recebe informações sobre, por exemplo:
# 1) para frente em 1 posição; # 2) virar à direita (90 graus) em 3 posições,
# 3) virar à esquerda (45 graus) em 2 posições, etc; seguindo a ordem da lista.
# É preciso ter pelo menos 1 request (ex: [["fwd",1]]). Possíveis requests:
#   [ ["fwd", 1], ["fwd", 2], ["fwd", d], ["r45", 1], ["r45", 2], ["r45", d],
#   ["r90", 1], ["r90", 2], ["r90", d], ["l45", 1], ["l45", 2], ["l45", d],
#   ["l90", 1], ["l90", 2], ["l90", d] ]
# a var 'd' indica quantidade de casas no grid, a partir da posição atual.
InfoReqSeq = [["fwd", 0], ["fwd", 1], ["r90", 1], ["l90", 1]]
nofInfoRequest = len(InfoReqSeq)  # Número de requests que o agente faz p/ o EnviSim antes de decidir
# a cada request, o programa salvará em um array (com nofInfoRequest elementos x 32 bits)
CurrentSensBits = np.zeros((nofInfoRequest, 32), dtype=np.int32)  # array c/ nofInfoRequest vals de 32 bits
# print('reqInfo:', InfoReqSeq)
# print('CurrentSensBits: ', CurrentSensBits)
# --- sys.exit(0)


# --x-- cada resposta do EnviSim muda esse vetor. O prog converte uma msg em bits do sensor
InpSensors = ["inp_nothing", "inp_breeze", "inp_danger", "inp_flash", "inp_goal",
              "inp_initial", "inp_obstruction", "inp_stench", "inp_bf", "inp_bfs",
              "inp_bs", "inp_fs", "inp_boundary", "inp_obstacle", "inp_wall",
              "inp_cannot", "inp_died", "inp_grabbed", "inp_none", "inp_restarted",
              "inp_success", "inp_pheromone", "inp_dir_n", "inp_dir_ne", "inp_dir_e",
              "inp_dir_se", "inp_dir_s", "inp_dir_sw", "inp_dir_w", "inp_dir_nw",
              "inp_deviation", "go", "_0", "_1", "mGB", "mFLSH"]  # a ordem dos bits é a seguinte:
# [ 0] = "inp_nothing"          **  sente nada (tem nada na casa requisitada)
# [ 1] = "inp_breeze"           **  sente brisa (uma casa antes de um buraco)
# [ 2] = "inp_danger"           **  sente perigo (casa requisitada/atual tem um Wumpus ou um buraco - morre)
# [ 3] = "inp_flash"            **  sente flash (uma casa antes do ouro ele vê o brilho do ouro)
# [ 4] = "inp_goal"             **  sente meta (casa requisitada/atual tem ouro - reward, que é a meta)
# [ 5] = "inp_initial"          **  sente início (casa requisitada/atual é o ponto de partida/saída)
# [ 6] = "inp_obstruction"      **  sente obstução (mandou request,d e vem obstrução é porque vai colidir em 'd')
# [ 7] = "inp_stench"           **  sente fedor (uma casa antes de um Wumpus)
# [ 8] = "inp_bf"               **  sente brisa/flash (na casa 'd' tem sinais de brisa e flash)
# [ 9] = "inp_bfs"              **  sente brisa/flash/stench (na casa 'd' tem brisa + flash + fedor)
# [10] = "inp_bs"               **  sente brisa/stench (na casa 'd' tem brisa + fedor)
# [11] = "inp_fs"               **  sente flash/stench (na casa 'd' tem flash + fedor)
# [12] = "inp_boundary"         **  colidiu com borda (mandou mover forward,d e colidiu com a borda do EnviSim)
# [13] = "inp_obstacle"             não usaremos nesse jogo
# [14] = "inp_wall"                 não usaremos nesse jogo
# [15] = "inp_cannot"           **  retornou que não é possível fazer a ação (out_act_grab, ut_act_leave)
# [16] = "inp_died"             *   seu agente morreu (no mundo EnviSim - caiu na casa do Wumpus ou no buraco)
# [17] = "inp_grabbed"          **  retorna que a ação de pegar ouro (out_act_grab) foi um sucesso
# [18] = "inp_none"                 não usaremos nesse jogo
# [19] = "inp_restarted"        *   depois de {"request":["restart",0]}, o EnviSim restaura a cena e envia essa msg
# [20] = "inp_success"          *   retorna que o agente saiu da caverna com sucerro (ut_act_leave) (com ou sem ouro)
# [21] = "inp_pheromone"            não usaremos nesse jogo
# [22] = "inp_dir_n"                não usaremos nesse jogo
# [23] = "inp_dir_ne"               não usaremos nesse jogo
# [24] = "inp_dir_e"                não usaremos nesse jogo
# [25] = "inp_dir_se"               não usaremos nesse jogo
# [26] = "inp_dir_s"                não usaremos nesse jogo
# [27] = "inp_dir_sw"               não usaremos nesse jogo
# [28] = "inp_dir_w"                não usaremos nesse jogo
# [29] = "inp_dir_nw"               não usaremos nesse jogo
# [30] = "inp_deviation"            não usaremos nesse jogo
# [31] = reservado
nofInpSensors = len(InpSensors)  # número de sensores de entrada
# print('inp: ', InpSensors)
# print('num: ', nofInpSensors)
# --x-- Para esse prog você usará as entradas '**'. As marcadas '*' o programa trata.


# -o-o- cada decisão resulta em uma ação/comando. O programa converte esse vetor em uma msg p/ EnviSim
OutNeurons = ["out_act_grab", "out_act_leave", "out_act_nill", "out_mov_forward", "out_req_forward", "out_req_left",
              "out_req_left45", "out_req_orientation", "out_req_restart", "out_req_right", "out_req_right45",
              "out_rot_left", "out_rot_right", "out_rot_back"]
# nomes dos atuadores/saídas. O que cada bit significa:
# [ 0] = "out_act_grab"         **  ação de pegar/agarrar o ouro (reward)
# [ 1] = "out_act_leave"        **  ação de deixar a caverna (no mesmo local de partida)
# [ 2] = "out_act_nill"             fazer nada (faz ação nenhuma nessa iteração)
# [ 3] = "out_mov_forward"      **  ação de mover adiante
# [ 4] = "out_req_forward"          requisição de info adiante (d casas) - automático, segue a lista InfoReqSeq
# [ 5] = "out_req_left"             requisição de info esquerda 90 (d casas) - automático, segue a lista InfoReqSeq
# [ 6] = "out_req_left45"           requisição de info esq 45 graus (d casas) - auto, segue a lista InfoReqSeq
# [ 7] = "out_req_orientation"      requisição de orientação - não usaremos nesse programa
# [ 8] = "out_req_restart"          requisição de recomeçar a jornada - automático no ini, depois de morte, sucesso
# [ 9] = "out_req_right"            requisição de info direita 90 (d casas) - auto, segue a lista InfoReqSeq
# [10] = "out_req_right45"          requisição de info dir 45 graus (d casas) - auto, segue a lista InfoReqSeq
# [11] = "out_rot_left"         **  ação de rotacionar esq.{"rotate":["left",2]}=90°; {"rotate":["left",1]}=45°
# [12] = "out_rot_right"        **  ação de rotacionar esq.{"rotate":["right",2]}=90°; {"rotate":["right",1]}=45°
# [13] = "out_rot_back"         **  ação de rotacionar back.{"rotate":["back",0]}={"rotate":["right",4]}=180°
# [14] = "_0"                       reservado p/ memorizar um estado de máq
# [15] = "_1"                       reservado p/ memorizar um estado de máq
# [16] = "mGB"                      reservado p/ memorizar um estado de máq (mem de que está segurando ouro)
# [17] = "mFLSH"                    reservado p/ memorizar um estado de máq (mem de que viu flash - ouro por perto)
# [..] = reservado
# -o-o- Para esse programa só usaremos as saídas/ações/comandos marcados com  **

# ---- mensagens trocadas entre AGENT <-> Wumpus ----
# format:    lists [['key_name'], ['string_val1','string_val2',...]],
# esta é a lista com todas as mensagens/comandos que o AGENTE envia para o Wumpus (server)
LstMsgsAGtoES = [[['act'], ['grab', 'leave', 'nill']],
                 [['move'], ['forward']],
                 [['request'], ['forward', 'left', 'left45', 'orientation', 'restart', 'right', 'right45']],
                 [['rotate'], ['left', 'right', 'back']]]

# estas macros (vars) substituem as strings de msgs pela dos componentes da lista acima
keyMagACT: str = LstMsgsAGtoES[0][0][0]  # use estes nomes de 'key' em vez das strings
keyMagMOV: str = LstMsgsAGtoES[1][0][0]
keyMagREQ: str = LstMsgsAGtoES[2][0][0]
keyMagROT: str = LstMsgsAGtoES[3][0][0]

ACTgrb: str = LstMsgsAGtoES[0][1][0]  # use estas vars em vez das strings (e.g.: o payload = 'grabbed')
ACTlev: str = LstMsgsAGtoES[0][1][1]
ACTnil: str = LstMsgsAGtoES[0][1][2]

MOVfor: str = LstMsgsAGtoES[1][1][0]

REQfwd: str = LstMsgsAGtoES[2][1][0]
REQlft: str = LstMsgsAGtoES[2][1][1]
REQl45: str = LstMsgsAGtoES[2][1][2]
REQori: str = LstMsgsAGtoES[2][1][3]
REQrst: str = LstMsgsAGtoES[2][1][4]
REQrgt: str = LstMsgsAGtoES[2][1][5]
REQr45: str = LstMsgsAGtoES[2][1][6]

ROTlft: str = LstMsgsAGtoES[3][1][0]
ROTrgt: str = LstMsgsAGtoES[3][1][1]
ROTbck: str = LstMsgsAGtoES[3][1][2]

# esta é a lista com todas as mensagens que o WUMPUS envia para o Agente (client)
LstMsgEStoAG = [[['sense'], ['breeze', 'danger', 'flash', 'goal', 'initial', 'obstruction', 'stench']],
                [['collision'], ['boundary', 'obstacle', 'wall']],
                [['outcome'], ['cannot', 'died', 'grabbed', 'none', 'restarted', 'success']],
                [['server'], ['connected', 'invalid', 'normal', 'paused', 'ready', 'stopped']],
                [['pheromone'], []],
                [['position'], []],
                [['direction'], ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']],
                [['deviation'], []]]

# estas macros (vars) substituem as strings de msgs pela dos componentes da lista acima
keyMwpSNS: str = LstMsgEStoAG[0][0][0]  # use estes nomes de 'key' em vez das strings
keyMwpCOL: str = LstMsgEStoAG[1][0][0]
keyMwpOUT: str = LstMsgEStoAG[2][0][0]
keyMwpSRV: str = LstMsgEStoAG[3][0][0]
keyMwpPHR: str = LstMsgEStoAG[4][0][0]
keyMwpPOS: str = LstMsgEStoAG[5][0][0]
keyMwpDIR: str = LstMsgEStoAG[6][0][0]
keyMwpDVA: str = LstMsgEStoAG[7][0][0]

SNSbrz: str = LstMsgEStoAG[0][1][0]  # use estas vars em vez das strings (e.g.: o payload = 'breeze')
SNSdng: str = LstMsgEStoAG[0][1][1]
SNSfsh: str = LstMsgEStoAG[0][1][2]
SNSgol: str = LstMsgEStoAG[0][1][3]
SNSini: str = LstMsgEStoAG[0][1][4]
SNSobs: str = LstMsgEStoAG[0][1][5]
SNStch: str = LstMsgEStoAG[0][1][6]
SNSnth: str = 'nothing'

CLDbnd: str = LstMsgEStoAG[1][1][0]
CLDobs: str = LstMsgEStoAG[1][1][1]
CLDwll: str = LstMsgEStoAG[1][1][2]

OUTcnt: str = LstMsgEStoAG[2][1][0]
OUTdie: str = LstMsgEStoAG[2][1][1]
OUTgrb: str = LstMsgEStoAG[2][1][2]
OUTnon: str = LstMsgEStoAG[2][1][3]
OUTrst: str = LstMsgEStoAG[2][1][4]
OUTsuc: str = LstMsgEStoAG[2][1][5]

SRVcnn: str = LstMsgEStoAG[3][1][0]
SRVinv: str = LstMsgEStoAG[3][1][1]
SRVnor: str = LstMsgEStoAG[3][1][2]
SRVpsd: str = LstMsgEStoAG[3][1][3]
SRVrdy: str = LstMsgEStoAG[3][1][4]
SRVstp: str = LstMsgEStoAG[3][1][5]

DIRn: str = LstMsgEStoAG[6][1][0]
DIRne: str = LstMsgEStoAG[6][1][1]
DIRe: str = LstMsgEStoAG[6][1][2]
DIRse: str = LstMsgEStoAG[6][1][3]
DIRs: str = LstMsgEStoAG[6][1][4]
DIRsw: str = LstMsgEStoAG[6][1][5]
DIRw: str = LstMsgEStoAG[6][1][6]
DIRnw: str = LstMsgEStoAG[6][1][7]
# --(end) -- mensagens trocadas entre AGENTE <-> Wumpus ----

# as variáveis usadas no loop de simulação
iterNum = 0  # conta o número de iterações executadas até agora
energy = 500  # quantidade total de energia (número de iterações) disponível para o agente
carryRWD: int = 0  # se o agente carrega a recompensa
cntNofReqs: int = 0  # contador para o número de solicitações já feitas
msg = ''  # string que concatena a mensagem que será enviada para EnviSim
answES = ''  # string que armazena a resposta da mensagem acabou de ser recebida do EnviSim
posX = 0  # *(use se quiser saber a posição X do agente no grid)
posY = 0  # *(use se quiser saber a posição Y do agente no grid)
direction: str = 'e'  # opcional (força a direção do agente para leste, voltada para a direita, ou pode ser aleatória)
pherom = 0  # *(EnviSim diz o valor das feromônios em uma posição do grid)
devAngle = 0  # *(EnviSim diz o ângulo de desvio entre a direção do agente e a guidingStar)
# vars marcadas com * são opcionais

strCode = ''  # str que recebe o código para exceções, erros, etc.
idxInpSensor: int = 0  # índice que indica qual sensor de entrada foi ativado
decision: int = 0  # índice que é retornado de Cogniton.infer() -> int

delaySec = 0  # atraso em segundos apenas para efeitos visuais

# ---.(end).---  variables and constants ---.---
# sys.exit(0)

reward = 0
