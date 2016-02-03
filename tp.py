#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP1 FAA Matthieu Caron 2016

import matplotlib.pyplot as plt
import numpy as np
import math
import random as r




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


def moindresCarres(matrix, vecteur):
    gauche = np.dot(matrix, matrix.T)
    droite = np.dot(matrix,vecteur) 
    return  np.dot(np.linalg.inv(gauche),droite)

def alpha(t) :
    return 1./(1.+4000.*float(t))

def descenteGradient(x1,y,teta,t, epsilon=0.00000001,N=100) :
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        intermediaire = np.dot(x1,(y - np.dot(x1.T,tetaActuel))) #gradient
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        #test de convergence 
        if math.fabs(mesureNormal2(x1,y,tetaPlusPlus)-mesureNormal2(x1,y,tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def descenteGradientStochastique(matrix,y,teta,t,epsilon=0.00000001,N=100) :
    '''
    a corriger
    '''
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        i = r.randint(0,99)
        vecX = np.array([matrix[0][i],matrix[1][i]],float)
        intermediaire = np.dot(vecX,(y[i] - np.dot(tetaActuel.T,vecX))) #gradient
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        #test de convergence 
        if math.fabs(mesureNormal2(vecX,y[i],tetaPlusPlus)-mesureNormal2(vecX,y[i],tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1




################# SCRIPT #################


# print "les erreurs pour y = 2*x + b"
# print "ABS = ", mesureAbs(x1,position,teta)
# print "JL1 = ", mesureNormal1(x1,position,teta)
# print "JL2 = ", mesureNormal2(x1,position,teta)
# print "JLINF = ", mesureNormalinf(x1,position,teta)
# resMoindreCarres = moindresCarres(x1,position)
# print "moindresCarres = ", resMoindreCarres
# nouveauY = resMoindreCarres[1]*x + resMoindreCarres[0]
# print "les erreurs après moindres carres"
# print "ABS = ", mesureAbs(x1,position,resMoindreCarres)
# print "JL1 = ", mesureNormal1(x1,position,resMoindreCarres)
# print "JL2 = ", mesureNormal2(x1,position,resMoindreCarres)
# print "JLINF = ", mesureNormalinf(x1,position,resMoindreCarres)
# print 'la différence des deux ensembles'
# print "ABS = ", mesureAbs(x1,position,teta) - mesureAbs(x1,position,resMoindreCarres)
# print "JL1 = ", mesureNormal1(x1,position,teta) - mesureNormal1(x1,position,resMoindreCarres)
# print "JL2 = ", mesureNormal2(x1,position,teta) - mesureNormal2(x1,position,resMoindreCarres)
# print "JLINF = ", mesureNormalinf(x1,position,teta) - mesureNormalinf(x1,position,resMoindreCarres)

print x1[1][99]

tetaDescente = np.array([7,7],float)
# print "descenteGradient = ", descenteGradient(x1,position,tetaDescente,1)
print "descenteGradientStochastique = ", descenteGradientStochastique(x1,position,tetaDescente,1)

# plt.plot(temps,position,'.')
# line1, = plt.plot(x,y, label='y')
# line2, = plt.plot(x,nouveauY,label='y modif')
# plt.legend(handles = [line1,line2])

# plt.xlabel('temps (s)')
# plt.ylabel('position (m)')
# plt.title('TP1 FAA')
# plt.savefig("schema.png")
# plt.show()

