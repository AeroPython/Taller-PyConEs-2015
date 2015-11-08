#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ejercicio del Algoritmo de Colonia de Hormigas

Taller de la PyConEs 2015: Simplifica tu vida con sistemas complejos y algoritmos genéticos

Este script contiene las funciones y clases necesarias para ejecutar una simulación del
Algoritmo de colonia de hormigas para optimizar el problema del viajante.


Este script usa arrays de numpy, aunque no debería ser difícil para alguien con experiencia
sustituírlos por otras estructuras si es necesario.

También usa la librería Matplotlib en las funciones que dibujan resultados."""


import numpy as np
import matplotlib.pyplot as plt

#Primero vamos a definir los dos tipos de objeto que necesitamos: 
class Mapa:
    '''Este objeto contiene el mapa de las ciudades que se van a visitar,
    con todos los datos necesarios (como las feromonas y distancias entre ciudades)
    y las hormigas que lo recorren'''
    def __init__ (self, n_ciud = 10, mapsize = 10):
        self.n_ciud = n_ciud
        self.mapsize = mapsize
        self.ciudades = mapsize * np.random.rand(n_ciud, 2)
        self.distances = np.zeros([n_ciud, n_ciud])
        for i in range(n_ciud):
            for j in range(i+1, n_ciud):
                vector = self.ciudades[i,:] - self.ciudades[j,:]
                modulo = np.linalg.norm(vector)
                self.distances[i,j] = modulo
                self.distances[j,i] = modulo
        self.feromap = np.zeros_like(self.distances)
        self.feromultiplier = 1
        self.conjunto_analisis = []
        self.pathdata = []
        self.bestpath = [0,1]
        self.bestpathlenght = n_ciud * mapsize
        
    def show_distances_matrix(self):
        '''Muestra la matriz de distancias en código de colores'''
        plt.matshow(self.distances)
        plt.title('Matriz de distancias entre ciudades')
        
    def show_feromones_matrix(self):
        '''Muestra la matriz de feromonas en código de colores'''
        plt.matshow(self.feromap)
        plt.title('Matriz de feromonas entre ciudades')
        
    def draw_distances(self):
        '''Dibuja un mapa con las ciudades, unidas entre sí por líneas que son más gruesas
        cuanto más cercanas son. Es una manera gráfica de comprobar la matriz de distancias.'''
        plt.figure(None, figsize=(8,8))
        plt.scatter(self.ciudades[:,0], self.ciudades[:,1], s = 100, c = '#5599FF',zorder=2)
        for i in range(self.n_ciud):
            for j in range(i+1, self.n_ciud):
                path = np.zeros([2,2])
                path[0,:] = self.ciudades[i,:]
                path[1,:] = self.ciudades[j,:]
                dist = self.distances[i,j]
                thickness = 7 - dist * (7 / self.mapsize)
                plt.plot(path[:,0], path[:,1],'#88AA22', linewidth=thickness,zorder=1)
        plt.title('Mapa de ciudades con sus distancias')
                
    def draw_feromones(self, rescale_lines = True):
        '''Dibuja un mapa con las ciudades, unidas entre sí por líneas que son más gruesas
        cuanto más feromonas contiene la ruta que las une. Es una manera gráfica 
        de comprobar la matriz de feromonas.'''
        plt.figure(None, figsize=(8,8))
        plt.scatter(self.ciudades[:,0], self.ciudades[:,1], s = 100, c = '#5599FF',zorder=2)
        if rescale_lines:
            maxfer = np.max(self.feromap)
        for i in range(self.n_ciud):
            for j in range(i+1, self.n_ciud):
                path = np.zeros([2,2])
                path[0,:] = self.ciudades[i,:]
                path[1,:] = self.ciudades[j,:]
                if rescale_lines:
                    fer = self.feromap[i,j]
                    if maxfer > 0:
                        fer *= 7/maxfer
                    
                else: 
                    fer = self.feromap[i,j]
                
                plt.plot(path[:,0], path[:,1],'#DD2222', linewidth=fer,zorder=1)
        plt.title('Mapa de ciudades con sus rastros de feromonas')
        
    def draw_best_path(self):
        '''Dibuja un mapa con las ciudades, unidas entre sí por la mejor ruta encontrada hasta el momento.'''
        plt.figure(None, figsize=(8,8))
        plt.scatter(self.ciudades[:,0], self.ciudades[:,1], s = 100, c = '#5599FF',zorder=2)
        ruta = self.ciudades[[self.bestpath]]
        plt.plot(ruta[:,0], ruta[:,1],'#2222AA', linewidth=8,zorder=1)
        plt.title('Mapa de ciudades con mejor ruta encontrada')
        
    def draw_results(self, relative_scale = False):
        '''Dibuja la longitud máxima, mínima y media de los caminos que siguen las hormigas,
        y la longitud mínima que el algoritmo ha encontrado'''
        plt.figure(None, figsize=(8,5))
        patharray = np.array(self.pathdata)
        for i in range(3):
            plt.plot(patharray[:,i])
        longx = len(patharray[:,0])
        plt.plot([0, longx], [self.bestpathlenght, self.bestpathlenght])
        plt.title('Longitud máxima, mínima, media y mejor camino encontrado')
        if not relative_scale : plt.ylim(0)
    
    def draw_best_results(self, relative_scale = False):
        '''Dibuja la longitud mínima  de los caminos que siguen las hormigas,
        para todas las veces que el algoritmo se ha ejecutado,
        y la longitud mínima que el algoritmo ha encontrado'''
        plt.figure(None, figsize=(8,5))
        longx = 0
        for i in range(len(self.conjunto_analisis)):
            patharray = np.array(self.conjunto_analisis[i])        
            plt.plot(patharray[:,1])
            longx = max(longx, len(patharray[:,0]))
            
        patharray = np.array(self.pathdata)
        longx = max(longx, len(patharray[:,0]))
        plt.plot(patharray[:,1])
        
        plt.plot([0, longx], [self.bestpathlenght, self.bestpathlenght])
        plt.title('Longitud mínima para cada ejecución y mejor camino encontrado')
        if not relative_scale : plt.ylim(0)
                
    def swarm_create(self, n_ant = 10):
        '''Crea una población de hormigas en el mapa'''
        self.lista_hormigas = []
        for i in range(n_ant):
            nueva = Hormiga(self)
            self.lista_hormigas.append(nueva)
        del(nueva)
        
    def swarm_show(self):
        '''Dibuja un mapa con las ciudades y las hormigas.
        Es una manera gráfica de comprobar dónde se encuentran.'''
        plt.figure(None, figsize=(8,8))
        plt.scatter(self.ciudades[:,0], self.ciudades[:,1], s = 100, c = '#5599FF')
        ant_pos = np.zeros([len(self.lista_hormigas), 2])
        for i in range(len(self.lista_hormigas)):
            hormiga = self.lista_hormigas[i]
            city = hormiga.position
            exact_position = self.ciudades[city,:]
            aprox_position = exact_position + 0.03 *self.mapsize * (np.random.rand(2) - 0.5)
            #print(exact_position)
            #print(aprox_position)
            ant_pos[i,:] = aprox_position
        plt.scatter(ant_pos[:,0], ant_pos[:,1], s = 5, c = 'k')
        plt.title('Mapa de ciudades y hormigas')
        
    def feromone_reset(self):
        '''Devuelve a 0 el mapa de feromonas para repetir el análisis
        de un mapa dado.
        Los datos alcanzados hasta ahora, se guardarán para posterior consulta.'''
        self.feromap = np.zeros_like(self.distances)
        self.conjunto_analisis.append(self.pathdata)
        self.pathdata = []
        
    def feromone_fine_tune(self):
        '''Permite controlar en detalle la cantidad de feromonas que se evaporan cada turno.
        Un factor mayor aumenta la cantidad de feromonas, y viceversa.
        Por defecto, el factor = 1.'''
        ok = False
        while not ok:
            x = input('introduzca un valor, p.ej. 1: ')
            try:
                x = float(x)
                ok = True
            except:
                print('Valor incorrecto')
        
        print('Valor cambiado')        
        self.feromultiplier = x
        
    def swarm_delete(self):
        '''Elimina a todas las hormigas del mapa'''
        del(self.lista_hormigas)
        self.lista_hormigas = []
        
    def swarm_generation(self):
        '''Realiza una generación completa de hormigas:
        1. Las mueve paso a paso hasta completar la ruta
        2. Analiza los resultados
        3. Deposita las feromonas
        4. Elimina las hormigas viejas y crea una nueva población
        5. Evapora las feromonas'''
        n_ant = len(self.lista_hormigas)
        self.pathlens = []
        for i in range(self.n_ciud - 1):
            for hormiga in self.lista_hormigas:
                hormiga.journey_step()
        for hormiga in self.lista_hormigas:
            hormiga.back_home()
            length = hormiga.calc_route_length()
            self.pathlens.append(length)
            if length < self.bestpathlenght :
                self.bestpathlenght = length
                self.bestpath = hormiga.route
            hormiga.feromone_spray()
        
        maxpath = max(self.pathlens)
        minpath = min(self.pathlens)
        meanpath = np.mean(self.pathlens)
        self.pathdata.append([maxpath, minpath, meanpath])
        self.swarm_delete()
        self.swarm_create(n_ant)
        self.feromap /= (3 / self.feromultiplier)
        self.feromap -=0.5
        self.feromap = np.where(self.feromap>0, self.feromap, np.zeros_like(self.feromap))
        
        
class Hormiga:
    '''Estas hormigas recorren el mapa acumulando información 
    sobre la longitud del viaje'''
    def __init__(self, mapa):
        self.mapa = mapa
        self.city_list = list(range(mapa.n_ciud))
        self.start = self.city_list.pop(np.random.randint(mapa.n_ciud))
        self.position = self.start
        self.route = [self.start,]
        self.distance_weight = 0.2
        
    def journey_step(self):
        '''Avanza a la siguiente ciudad'''
        all_dist = self.mapa.distances[self.position, :]
        posible_dist = all_dist[self.city_list] 
        all_ferom = self.mapa.feromap[self.position, :]
        posible_ferom = all_ferom[self.city_list] 
        probabilities = (self.distance_weight / (0.1 + posible_dist) +
                         (1 - self.distance_weight) * (0.1 + posible_ferom))
        probabilities = probabilities / np.sum(probabilities)
        indexes = np.arange(len(self.city_list))
        new_city_index = np.random.choice(indexes, 1, p = probabilities)
        self.position = self.city_list.pop(new_city_index)
        self.route.append(self.position)
        
    def back_home(self):
        '''Devuelve a la hormiga a su ciudad inicial tras recorrer todo el mapa'''
        self.position = self.start
        self.route.append(self.start)
        
    def calc_route_length(self):
        '''Calcula la longitud de la ruta de la hormiga'''
        self.route_length = 0
        for i in range(1, len(self.route)):
            city1 = self.route[i-1]
            city2 = self.route[i]
            dist = self.mapa.distances[city1, city2]
            self.route_length += dist
        return self.route_length
            
    def feromone_spray(self):
        '''Deposita sobre la ruta recorrida una cantidad de feromonas
        que depende de la longitud del viaje'''
        feromone_amount = (2 * self.mapa.n_ciud * self.mapa.mapsize)/self.route_length**2
        for i in range(1, len(self.route)):
            city1 = self.route[i-1]
            city2 = self.route[i]
            self.mapa.feromap[city1, city2] += feromone_amount
            self.mapa.feromap[city2, city1] += feromone_amount
    
#Sólo necesitamos estos objetos para ejecutar nuestras simulaciones


    
    
#Ejemplo:
if __name__ == '__main__':
    
    map1 = ants.Mapa(8)
    
    
