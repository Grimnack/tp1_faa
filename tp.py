#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP1 FAA Matthieu Caron 2016

import matplotlib.pyplot as plt
import numpy as np
import math




################# LECTURE #################

def lecture(pathname) :
    fichier = open(pathname,'r')
    res = []
    for ligne in fichier :
        res.append(float(ligne))
    fichier.close()
    return res 

################# VARIABLES #################

a = 2
b = 3
e = 1

N = 100

x = np.linspace(4,15,N)
y = a*x + b
z = np.ones(len(x))

temps = np.array(lecture('t.txt'),float)
position = np.array(lecture('p.txt'),float)
teta = np.array([b,a],float)

x1 = np.zeros((2,N))
x1[1,:] = temps
x1[0,:] = z

############# MESURE DE PERF #############

def mesureAbs(x1,y,teta,N=100) :
    vecteur = y - np.dot(x1.T,teta)
    return np.sum(np.absolute(vecteur))/N

def mesureNormal2(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.dot(vecteur.T, vecteur)/N

def mesureNormal1(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    interm = np.dot(vecteur.T, vecteur)
    return math.sqrt(interm)/N

def mesureNormalinf(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.amax(np.absolute(vecteur))


################# SCRIPT #################

print "ABS = ", mesureAbs(x1,position,teta)
print "JL1 = ", mesureNormal1(x1,position,teta)
print "JL2 = ", mesureNormal2(x1,position,teta)
print "JLINF = ", mesureNormalinf(x1,position,teta)

plt.plot(temps,position,'.')
plt.plot(x,y)

plt.xlabel('temps (s)')
plt.ylabel('position (m)')
plt.title('TP1 FAA')
plt.savefig("schema.png")
plt.show()

