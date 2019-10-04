import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def Leer_Datos(file_name):
    return np.genfromtxt(file_name, delimiter=',', skip_header=1)

def Normalizar_Datos(data):
    mean_data = np.mean(data)
    standard_dev = np.std(data)
    data = data - mean_data
    data = data / standard_dev
    return data, mean_data, standard_dev

def Separar_X_y(data):
    n = data.shape[1]
    X = data[:, :n-1]
    y = data[:, n-1:]
    return X, y

def Crear_Entrenamiento_Prueba(data):
    num_rows = data.shape[0]
    train_percentage = 0.7
    row_split_data = int(num_rows * train_percentage)
    training, test = data[:row_split_data, :], data[row_split_data:, :]
    return training, test

def Crear_Pesos(X):
    return np.random.rand(X.shape[1])

def Calcular_Costo(X, y, W):
    result = np.matmul(X, W)
    result = result - y
    result = np.square(result)
    result = np.sum(result)
    result = result / (2 * y.shape[0])
    return result

def Gradiente_Descendiente(X, y, W, num_iter, learn_rate):
    costs = np.zeros(num_iter)
    for i in range(num_iter):
        result = np.matmul(X, W)
        result = result - y
        result = np.matmul(np.transpose(X), result)
        result = np.divide(result, y.shape[0])
        result = np.multiply(result, learn_rate)
        W = W - result
        costs[i] = Calcular_Costo(X, y, W)
    return W, costs

def Ecuacion_Normal(X, y):
    X_t = np.transpose(X)
    return np.matmul(np.linalg.inv(np.matmul(X_t, X)), np.matmul(X_t, y))
