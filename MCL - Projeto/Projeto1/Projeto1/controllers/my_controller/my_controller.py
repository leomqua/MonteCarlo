
######################## Localização de Monte Carlo ##########################


# Bibliotecas a serem utilizadas no código:
from controller import Robot, Keyboard, Lidar
import random as rnd
from math import *
import numpy as np
import matplotlib.pyplot as plt
import copy
from numpy import random, set_printoptions

# Instanciando o robô
robot = Robot()
timestep = int(robot.getBasicTimeStep())

teclado = Keyboard()
teclado.enable(timestep)

# Lidar
sick = robot.getDevice("LDS-01")
sick.enable(timestep)
sick.enablePointCloud()

# Rodas
front_left = robot.getDevice("front left wheel")
front_right = robot.getDevice("front right wheel")
back_left = robot.getDevice("back left wheel")
back_right = robot.getDevice("back right wheel")
front_left.setPosition(float('inf'))
front_right.setPosition(float('inf'))
back_left.setPosition(float('inf'))
back_right.setPosition(float('inf'))

# Encoders
front_left_encoder = robot.getDevice("front left wheel sensor")
front_right_encoder = robot.getDevice("front right wheel sensor")
front_left_encoder.enable(timestep)
front_right_encoder.enable(timestep)

raio = 0.111 # Raio das rodas do pioneer
L = 0.6 # Distância entre as rodas
pose = [1.46, 6, 6.28]

# Lista com as posições de cada landmark somado com 5 em x e 6 em y 
# para centralizar no gráfico
landmarks = [[0.75, 3.86], [8.25, 11.28], [6.7, 1.82], [4.21, 9.27]] 

medidas = [0,0] # Encoder frente esq e dir
distancia = [0,0] # Dist percorrida pelas rodas esq e dir frente 
ultimas_medidas = [0, 0] # Ultimas medidas feitas da esquerda e direita, respectivamente
distancias = [0, 0] # Lista das distâncias medidas para a rodas esquerda e direita, respectivamente
tamanho_x = 10 # Tamanho do mundo em x
tamanho_y = 12 # Tamanho do mundo em y

fig, ax = plt.subplots() # cria o ambiente para o gráfico existir, fig é a figura e ax é o grafico

########################################################################################
# Classe Robô
class Robot:
  # Construtor
  def __init__(self, x, y, yaw, ruido_frente, ruido_virar, ruido_sensor): 
    # Atributos
    self.x = x # Posição do robô em x
    self.y = y # Posição do robô em y
    self.yaw = yaw # Rotação do robô 
    self.ruido_frente = ruido_frente # Ruído relacionado ao andar do robô 
    self.ruido_virar = ruido_virar # Ruído relacionado ao virar do robô 
    self.ruido_sensor = ruido_sensor # Ruído relacionado ao sensor do robô 
    self.finished = False # Variável usada para a rotação do robô
 
  def odometria(self):
    medidas[0] = round(front_left_encoder.getValue(), 3) # Encoder esquerdo
    medidas[1] = round(front_right_encoder.getValue(), 3) # Encoder direito
    
    for i in range(2):
        diff = medidas[i] - ultimas_medidas[i] 
        distancias[i] = round(diff * raio, 3) 
          
    # Cálculo das distâncias linear e angular percorrida no timestep
    deltaS = round((distancias[0] + distancias[1]) / 2.0, 3)
    deltaTheta = round((distancias[0] - distancias[1]) / L, 3)
    
    pose[2] = (pose[2] + deltaTheta) % 6.28 # Calcula e atualiza a orientação do robô
    
    deltaSx = deltaS * cos(pose[2]) # Calcula a variação de distância em x do robô
    deltaSy = deltaS * sin(pose[2]) # Calcula a variação de distância em y do robô
    
    pose[0] = pose[0] + deltaSx # Calcula e atualiza a posição do robô
    pose[1] = pose[1] + deltaSy # Calcula e atualiza a posição do robô
    
    # print("Postura:", pose)
    
    for i in range(2): # Atualiza a lista de últimas medições como as medições atuais
        ultimas_medidas[i] = medidas[i]
        
        
  def mover(self, virar, frente, atual_x, atual_y, atual_yaw):
     self.odometria() 
     yaw_antes = pose[2]
     yaw_obj = (yaw_antes + virar) % 6.28
     print(yaw_obj)
     print(pose[2])
     while self.finished == False:
         if  pose[2] < yaw_obj - 0.01 or pose[2] > yaw_obj + 0.01 :
             front_left.setVelocity(2)
             front_right.setVelocity(-2)
             back_left.setVelocity(2)
             back_right.setVelocity(-2)
             print(pose[2]) 
             self.odometria() 
         else:
             self.finished = True
             front_left.setVelocity(0)
             front_right.setVelocity(0)
             back_left.setVelocity(0)
             back_right.setVelocity(0)
         robot.step(timestep)    
             
     self.finished = False         
         
     while self.finished == False:
         if sqrt((atual_x - pose[0])**2 + 
                 (atual_y - pose[1])**2) < frente:
             front_left.setVelocity(4)
             front_right.setVelocity(4)
             back_left.setVelocity(4)
             back_right.setVelocity(4)
             self.odometria()
         else:
             self.finished = True
             front_left.setVelocity(0)
             front_right.setVelocity(0)
             back_left.setVelocity(0)
             back_right.setVelocity(0)
         robot.step(timestep)
     self.finished = False
             

  def lidar(self):
     dist = []
     deteccao = False
     laser = sick.getRangeImage()
     for i in range(270):
         if deteccao == True and laser[i] != inf:
             if abs(dist[-1] - laser[i]) > 0.5:
                 deteccao = False
         if deteccao == False and laser[i] != inf:
             dist.append(laser[i])
             deteccao = True
     return dist
  
  def medir_landmark(self):
      angulos = []
      for i in range(len(landmarks)):
          x = landmarks[i][0] - pose[0]
          y = landmarks[i][1] - pose[1]
          theta = atan2(y,x)
          angulos.append(theta)
      print(angulos)
      return angulos
          
          
            
#######################################################################   
# Classe Partícula
class Particula:
  def __init__(self, ruido_frente, ruido_virar, ruido_sensor): 
    # Atributos
    self.x = random.random() * tamanho_x
    self.y = random.random() * tamanho_y 
    self.yaw = random.random() * 2 * pi
    self.ruido_frente = ruido_frente
    self.ruido_virar = ruido_virar
    self.ruido_sensor = ruido_sensor
    self.w = 0.0

  def mover(self, virar, frente):
      orientacao = self.yaw + virar + rnd.gauss(0, self.ruido_virar) 
      orientacao = orientacao % (2*pi) 
      self.yaw = orientacao 

      dist = frente + rnd.gauss(0, self.ruido_frente) 
      self.x = (self.x + (cos(self.yaw) * dist)) % tamanho_x # Mundo cíclico
      self.y = (self.y + (sin(self.yaw) * dist)) % tamanho_y

  def gaussian(self, mu, sigma, x): 
    # Método recebe média, desvio pad, x e calcula a gaussiana
    return exp(-((mu-x)**2) / (sigma**2) / 2) / sqrt(2*pi*(sigma**2))
  
  def medir_prob(self, medida_robo,yaw_landmark): 
    dist = []
    prob = 1
    for i in range(len(landmarks)):
        if yaw_landmark[i]<2.35 and yaw_landmark[i]> -2.35:
            dist.append(sqrt((self.x - landmarks[i][0])**2 + 
                               (self.y - landmarks[i][1])**2))
       
    print(dist)                    
    dist.sort()
    medida_robo.sort()
    
    for i in range(min(len(dist),len(medida_robo))):
        prob = prob * self.gaussian(dist[i],self.ruido_sensor, medida_robo[i])
    self.w = prob

  def __repr__(self): 
    # Método para representação de cada objeto partícula
    return str(self.x) + ' ' + str(self.y) + ' ' + str(self.yaw) + ' ' + str(self.w) + ' ' + str(hex(id(self))) + '\n' 
 

def selecionaParticula(lista): 
  # Função para selecionar as melhores partículas
  w_soma = sum([particula.w for particula in lista]) # Soma todos os pesos das partícula
  probs = [particula.w / w_soma for particula in lista] # Normalização do peso de cada partícula
  return lista[np.random.choice(len(lista), p=probs)] # Escolhe partículas aleatórias levando em 
  # consideração a probabilidade (as com maiores probabilidades de representarem o robô tem mais chances de continuarem)
     

####################################################################################### 

  # Robô começa parado
front_left.setVelocity(0)
back_left.setVelocity(0)
front_right.setVelocity(0)
back_right.setVelocity(0)
# Criando um objeto robô no meio do mundo
meuRobo = Robot(1.5, 6, 0, 0.3, 0.05, 1.0) 

#todo:
# MCL
N = 1000
p = []

for i in range(N):
    p.append(Particula(0.3,0.05,1.0))
    
while robot.step(timestep) != -1:
    ax.clear()
    ax.scatter([particula.x for particula in p], 
                [particula.y for particula in p], s=3)
    ax.set_xlim([0,10])
    ax.set_ylim([0,12])
    plt.pause(0.5)
    
    atual_x = pose[0]
    atual_y = pose[1]
    atual_yaw = pose[2]
    
    tecla = -1
    frente = 0
    giro = 0
    while tecla == -1:
        robot.step(timestep)
        print("Esperando tecla..")
        tecla = teclado.getKey()
    if tecla == ord('W'):
        frente = 0.2    
    elif tecla == ord('A'):
        giro = 1.57
        
    meuRobo.mover(giro, frente, atual_x, atual_y, atual_yaw)
    
    for j in range(N):
        p[j].mover(frente, giro)
        
    Z = meuRobo.lidar()
    Y = meuRobo.medir_landmark()
    for j in range(N):
        p[j].medir_prob(Z,Y)
        
    p_nova = []
    for i in range(N):
        particula = selecionaParticula(p)
        particula.x = particula.x + rnd.gauss(0, 0.1)
        particula.y = particula.y + rnd.gauss(0, 0.1)
        particula.yaw = particula.yaw + rnd.gauss(0, 0.05)
        p_nova.append(copy.deepcopy(particula))
    p = p_nova
    
    
    
        
    
        
    
    
    
    
    
    
    
    
    
    
    





