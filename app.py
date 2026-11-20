import gradio as gr
import numpy as np

# Simple forward-prop structures for independent execution
def linear_forward(A_prev, W, b):
    return np.dot(W.T, A_prev) + b

def L_layer_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L+1):
        A_prev = A
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        A = linear_forward(A_prev, Wl, bl)
    return A, A_prev

print("System ready for deployment.")
