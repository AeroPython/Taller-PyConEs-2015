#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ejercicio del Algoritmo de optimización ingenieril

Taller de la PyConEs 2015: Simplifica tu vida con sistemas complejos y algoritmos genéticos

Este script contiene las funciones necesarias para calcular un rotor o hélice.


Este script usa arrays de numpy, aunque no debería ser difícil para alguien con experiencia
sustituírlos por otras estructuras si es necesario.

"""


import numpy as np

def airf_aerodata(c, v, alpha, x, rho):
    '''Devuelve los coeficientes de sustentación y resistencia.
    c = cuerda
    v = velocidad
    alpha = ángulo de ataque'''
    
    #Calcular reynolds
    factor_Re = 0.5 + 8 *(1.5 / (1 + c * v * rho) )
    factor_Re_2 = 1 / (1.2 - factor_Re)
    #calcular cl
    alpha_rad = alpha * np.pi/180
    s_fun = 0.5 - np.arctan((alpha - 20) * 0.4)/np.pi
    s_fun_2 = 0.5 - np.arctan((-alpha - 20) * 0.4)/np.pi
    cl_base = (alpha_rad + 0.) * 2 * np.pi
    cl = cl_base * s_fun * s_fun_2 * (0.7 + 0.3 * np.sqrt(x))
    cd_base = 0.1 * cl_base**2 + 0.01 * factor_Re
    cd_base_2 = 0.55 #+ 0.01 * factor_Re
    cd = 0.4 * cd_base * s_fun * s_fun_2 + cd_base_2 * (1 - s_fun) + cd_base_2 * (1 - s_fun_2)
#    print(factor_Re, factor_Re_2)
    
    return cl, cd#, factor_Re


def airf_slope (alpha, v, c, x, rho = 0.0208):
    '''
    Calcula la pendiente de la curva de sustentación.
    Se alimenta con 4 arrays de dimensión N:
     - c, cuerda del perfil en metros
     - v, velocidad del en metros
     - alpha, ángulo de ataque del perfil en  RADIANES
     - x, posición del perfil (0 = raíz, 0 = punta)
     
    Devuelve en un vector de dimensión N las pendientes de la curva de sustentación para cada perfil.
    
    '''
    alpha_0 = alpha *180 / np.pi #La función que calcula cl usa grados
    
    cl = airf_aerodata(c, v, alpha_0, x, rho = rho)[0]
    
    #Esta parte es por si hay algún alpha = 0, para evitar el error de matemáticas.
    for ii in range(alpha.shape[0]):
        if abs(alpha[ii]) < 0.00000001:
            alpha[ii] = 0.00000001
    
    slope = cl / alpha
    
    return slope

def chord(x, law = 0.05):
    '''
    Devuelve un array de longitudes de cuerda. 
    Entradas:
       -x : array de posiciones (0 = raíz, 1 = punta)
       -law : describe la forma de la pala, y puede ser:
            - un número float: la cuerda es constante, con ese valor en metros
            - una lista de la forma ['l', x1, x2 ]:
                La cuerda es lineal con la posición, y mide x1 m en la raíz y x2 m en la punta
    '''
    c_0 = np.array([1])
    if type(x) == np.ndarray:
        c_0 = np.ones_like(x)
    
     
   
    
    if type(law) == float :
        #print('law is float')
        c = c_0 * law
    elif type(law) == list :
        if law[0] == 'l':
            c = law[1] + (law[2] - law[1]) * x
        else:
            c = 'ERROR '
            print('Law not recogniced: ', type(law))
            print(5 * c)
    else:
        c = 'ERROR '
        print('Law not recogniced: ', type(law))
        print(5 * c)
    return c


def torsion(x, law = 'c', p = 10):
    
    '''
    Devuelve un array de torsiones. 
    Entradas:
       -x : array de posiciones (0 = raíz, 1 = punta)
       -p : parámetro 
       -law : describe la forma de la ley de torsiones, y puede ser:
            - 'c': distribución de torsión constante = p
            - 'l': distribución de torsión lineal, con p[0] en la raíz y p[1] en la punta
            - 'h': distribución de torsión hiperbólica, con torsión p en la punta
    '''
    
    c_0 = np.array([1])
    if type(x) == np.ndarray:
        c_0 = np.ones_like(x)

    if law == 'c' :
        t = c_0 * p
    elif law == 'h' :

        t0 =   (p / x)
        t = (t0 * (0.5 * (np.sign(90 - t0) + 1)) +
             90 * (0.5 * (np.sign(t0 - 90) + 1)) )
    elif law == 'l':
        t = p[0] + (p[1] - p[0]) * x
    else:
        t = 'ERROR'
    return t


def integrar(x, y):
    step = x[1:] - x[:-1]
    y0 = (y[:-1] + y[1:])/2
    return np.sum(step * y0)

def rotor_adim(omega, vz, R, b, 
                     x_min = 0.05,  n = 500, 
                     theta0 = 0, tors_param = ['c', 10], 
                     chord_params = 0.05, 
                     rho = 0.0208):
    '''Calcula un rotor teniendo en cuenta perdidas de punta de pala'''
    
    x = np.linspace(x_min, 1, n)
    dx = x[1] - x[0]
    r = x * R
    dr = dx * R
    vr = omega * r
    c = chord(x, chord_params)
    sigma = b * c / (np.pi * R)
    
    
    theta = theta0 + torsion(x, tors_param[0], tors_param[1]) * np.pi / 180
    a0 = 2 * np.pi
    a = a0
    vza = vz / (omega * R)
    
    #teoría elemento de pala + tubo de corriente:

    
    

    for ii in range(5):
        ao = a * sigma / 2
        via = 0.5 * ( - (vza + ao/4) + np.sqrt((vza + ao/4)**2 + ao *(x * theta - vza)))
        vi = omega * R * via
        ut = vr
        up = -(vz * np.ones_like(x) + vi)
        ur = np.sqrt(ut **2 + up ** 2)
        fi = np.arctan(up / ut)
        alpha = theta + fi
        a_1 = a
        a = airf_slope( alpha, ur, c, x, rho = rho)
        #print(np.mean(a), np.mean(a/a0), np.mean(a/a_1))
    
    
    cl, cd = airf_aerodata(c, ur, alpha * 180 / np.pi, x, rho = rho)#[0:2]
        
        
    dct = ao * (theta - (vza + via)/x) * x**2
    dcpi = - fi * x * dct
    dcp0 = sigma * cd * x**3 / 2
    
    ct = integrar(x, dct)
    cpi = integrar(x, dcpi)
    cp0 = integrar(x, dcp0)
    cp = cpi + cp0
    
    
    #Calculamos el factor B para representar las pérdidas en Punta de Pala: 
    
    B = 1 - np.sqrt(abs(2 * ct)) / b
    #print('B = ', B)
    
    dct = np.where(x < B, dct, 0)
    dcpi = np.where(x < B, dcpi, 0)
    
    ct = integrar(x, dct)
    cpi = integrar(x, dcpi)
    cp0 = integrar(x, dcp0)
    cp = cpi + cp0
    
    return (ct, cp, vza)

def densidad(h):
    t = 288.15 - 0.0065 * h
    rho = 1.225 * (t/288.15) ** (9.8 / (287 * 0.0065) - 1)
    return rho

def calcular_rotor(omega, vz, R, b, 
                    h = 0, 
                    theta0 = 0.174, tors_param = ['h', 14], 
                    chord_params = 0.05):
    '''
    Calcula las propiedades de una hélice. 
    
    Argumentos obligatorios:
    
        - omega: velocidad de giro de la hélice, en rad/s
        - vz: velocidad de avance, en m/s
        - R : radio de la hélice
        - b : número de palas
        
    Argumentos opcionales:
    
        - h : altitud de vuelo, en metros sobre el nivel del mar
        - theta0 : ángulo de paso colectivo
        - tors_param : parámetros de torsión de la hélice:
            formato: [ley, p]
                p: Parámetro: número o lista
                Ley:describe la forma de la ley de torsiones, y puede ser:
                    - 'c': distribución de torsión constante = p
                    - 'l': distribución de torsión lineal, con p[0] en la raíz y p[1] en la punta
                    - 'h': distribución de torsión hiperbólica, con torsión p en la punta
        - chord_params : parámetros de distribución de cuerda de la hélice.
          Describe la forma de la pala, y puede ser:
            - un número float: la cuerda es constante, con ese valor en metros
            - una lista de la forma ['l', x1, x2 ]:
                La cuerda es lineal con la posición, y mide x1 m en la raíz y x2 m en la punta  
    Devuelve:
    
        - T : tracción de la hélice, en Newtons
        - P : potencia de la hélice, en Watios
        - efic : eficiencia o rendimiento de la hélice (a v=0 es 0 por definición)
        - mach_punta : número de mach de las puntas de la hélice
    '''
    x_min = 0.05
    n = 100 
    temp = 288.15 - 0.0065 * h
    v_son = np.sqrt(1.4 * 8.314 * temp / 0.029)
    rho = densidad(h)
    sup = np.pi * R**2
    ct, cp, vza = rotor_adim(omega, vz, R, b, x_min, n, theta0, tors_param, chord_params, rho)
    T = max(0, ct * rho * sup * omega**2 * R**2)
    P = max(0, cp * rho * sup * omega**3 * R**3)
    if cp > 0:
        efic = T * vz / P
    else:
        efic = 0
    v_punta = np.sqrt(vz**2 + (omega * R)**2)
    mach_punta = v_punta / v_son
    
    return T, P, efic, mach_punta
