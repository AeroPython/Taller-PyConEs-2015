import numpy as np
from numpy.lib.stride_tricks import as_strided
import random as random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def create_world (lines = 10, columns = 0, proportion = 0.5):
    """
    This function creates a random world consisting of a matrix of individuals.
    There are two types of individuals, in a certain proportion
    """
    
    if columns == 0:
        columns = lines
    
    len_world = lines * columns
    world = round(proportion * len_world) * [1] 
    world += (len_world - len(world)) * [0]
    random.shuffle(world)
        
    return np.array(world).reshape(lines, columns)


def plot_world (world):
    '''
    This function takes a numpy 2D array with string elements 
    and scatters it with the colours corresponding to these strings
    '''
    
    #This sets the colours for the plot
    cmap = plt.cm.jet
    world = world*160 + 90

    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, aspect='equal')
        
    for ii in range (world.shape[0]):
        
        x = range(world.shape[1])
        y = world.shape[1] * [ii]
        colours = world [ii]
        
        for (x, y, c) in zip(x, y, colours):
            ax.add_artist(Circle(xy=(x, y), radius=0.45, color=cmap(c)))
        
    plt.axis('off')
    ax.set_xlim(-1, world.shape[1])
    ax.set_ylim(-1, world.shape[0])



def sliding_window(arr, window_size):
    """ 
    #by pv on stack overflow
    Construct a sliding window view of the array
    """
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def neighbourhood(arr, i, j, d):
    """
    #by pv on stack overflow
    Return d-th neighbors of cell (i, j)
    """

    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()

def weighted_choice_sub(weights):
    """
    weights = [0.9, 0.05, 0.05]
    N = 100000
    lista = [weighted_choice_sub(weights) for ii in range(N)]
    print( lista.count(0)/N, lista.count(1)/N, lista.count(2)/N)
    """
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i