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



def Experimento1():
    print("Experimento 1:\n")
    data_files = ["petrol_consumption.csv", "ex1data1.csv", "oceano_simple.csv"]
    cost_table = [["Costo de entrenamiento", "Costo de prueba"]]
    for name_data_file in data_files:
        cost = []
        original_data = Leer_Datos(name_data_file)
        original_X, original_y = Separar_X_y(original_data)
        data, mean_data, standard_dev = Normalizar_Datos(original_data)
        X, y = Separar_X_y(data)
        X = np.c_[X, np.ones(X.shape[0])]        #bias
        X_train, X_test = Crear_Entrenamiento_Prueba(X) 
        y_train, y_test = Crear_Entrenamiento_Prueba(y)
        y_train = np.reshape(y_train, y_train.shape[0])
        y_test = np.reshape(y_test, y_test.shape[0])
        print("X_train_shape = ", X_train.shape)
        print("y_train_shape = ", y_train.shape)
        W_n = Ecuacion_Normal(X_train, y_train)
        cost_train = Calcular_Costo(X_train, y_train, W_n)
        cost_test = Calcular_Costo(X_test, y_test, W_n)
        cost.append(cost_train)
        cost.append(cost_test)
        cost_table.append(cost)
        print("Archivo: ", name_data_file)
        print("Pesos de Ecuacion normal: ", W_n)
        print("Costo de entrenamiento: ", cost_train)
        print("Costo de prueba: ", cost_test, "\n")
    print("\n")

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    data_files.insert(0, "")
    
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=data_files,
        line_color='darkslategray',
        fill_color=headerColor,
        align=['left','center'],
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=cost_table,
        line_color='darkslategray',
        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 11)
        ))
    ])

    fig.show()

def Experimento2():
    print("Experimento 2:\n")
    data_files = ["petrol_consumption.csv", "ex1data1.csv", "oceano_simple.csv"]
    num_iters = [500,1000,1500,2000,2500,3000,3500]
    num_iters_label = num_iters.copy()
    num_iters_label.insert(0, "Tasas de aprendizaje \ Numero de iteraciones")
    learn_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    for name_data_file in data_files:
        result_table = [learn_rates]
        original_data = Leer_Datos(name_data_file)
        original_X, original_y = Separar_X_y(original_data)
        data, mean_data, standard_dev = Normalizar_Datos(original_data)
        X, y = Separar_X_y(data)
        X = np.c_[X, np.ones(X.shape[0])]        #bias
        X_train, X_test = Crear_Entrenamiento_Prueba(X) 
        y_train, y_test = Crear_Entrenamiento_Prueba(y)
        y_train = np.reshape(y_train, y_train.shape[0])
        y_test = np.reshape(y_test, y_test.shape[0])
        for num_iter in num_iters:
            learn_rate_row = []
            for learn_rate in learn_rates:
                W = Crear_Pesos(X)
                W, costs = Gradiente_Descendiente(X_train, y_train, W, num_iter, learn_rate)
                cost_train = Calcular_Costo(X_train, y_train, W)
                cost_test = Calcular_Costo(X_test, y_test, W)
                learn_rate_row.append("%.4f" % cost_test)
                print("Archivo: ", name_data_file)
                print("num_iter = ", num_iter)
                print("learn_rate = ", learn_rate)
                print("Pesos de Gradiente descendiente: ", W)
                print("Costo de entrenamiento: ", cost_train)
                print("Costo de prueba: ", cost_test, "\n")
            result_table.append(learn_rate_row)
        
        headerColor = 'grey'
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'
        
        fig = go.Figure(data=[go.Table(
        header=dict(
            values=num_iters_label,
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left','center'],
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=result_table,
            line_color='darkslategray',
            fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor]*6],
            align = ['left', 'center'],
            font = dict(color = 'darkslategray', size = 11)
            ))
        ])
        fig.show()
    print("\n")

def Experimento3():
    print("Experimento 3:\n")
    name_data_file = "ex1data1.csv"
    original_data = Leer_Datos(name_data_file)
    original_X, original_y = Separar_X_y(original_data)
    original_X_train, original_X_test = Crear_Entrenamiento_Prueba(original_X)
    original_y_train, original_y_test = Crear_Entrenamiento_Prueba(original_y)
    data, mean_data, standard_dev = Normalizar_Datos(original_data)

    X, y = Separar_X_y(data)
    X = np.c_[X, np.ones(X.shape[0])] #bias

    X_train, X_test = Crear_Entrenamiento_Prueba(X)
    y_train, y_test = Crear_Entrenamiento_Prueba(y)

    y_train = np.reshape(y_train, y_train.shape[0])
    y_test = np.reshape(y_test, y_test.shape[0])

    W = Crear_Pesos(X_train)
    num_iter = 100
    learn_rate = 0.1
    W, costs = Gradiente_Descendiente(X_train, y_train, W, num_iter, learn_rate)
    print("Pesos de Gradiente Descendiente:\n", W)
    cost_test = Calcular_Costo(X_test, y_test, W)
    print("Costo de prueba: ", cost_test, "\n")

    plt.subplot(2, 1, 1)
    plt.plot(original_X_train.reshape(original_X_train.shape[0]), original_y_train.reshape(original_y_train.shape[0]), 'ro')
    s = np.array([3.0, 25.0])
    s_n = (s - mean_data) / standard_dev
    t = ((W[0] * s_n + W[1]) * standard_dev) + mean_data
    plt.plot(s, t)

    W_n = Ecuacion_Normal(X_train, y_train)
    print("Pesos de Ecuacion normal:\n", W_n)
    cost_test = Calcular_Costo(X_test, y_test, W_n)
    print("Costo de prueba: ", cost_test, "\n")
    t = ((W_n[0] * s_n + W_n[1]) * standard_dev) + mean_data
    plt.subplot(2, 1, 2)
    plt.plot(original_X_train.reshape(original_X_train.shape[0]), original_y_train.reshape(original_y_train.shape[0]), 'ro')
    plt.plot(s, t)

    plt.show()

    print("\n")

def Experimento4():
    print("Experimento 4:\n")
    data_files = ["petrol_consumption.csv", "ex1data1.csv", "oceano_simple.csv"]
    for name_data_file in data_files:
        original_data = Leer_Datos(name_data_file)
        original_X, original_y = Separar_X_y(original_data)
        data, mean_data, standard_dev = Normalizar_Datos(original_data)
        X, y = Separar_X_y(data)
        X = np.c_[X, np.ones(X.shape[0])]        #bias
        X_train, X_test = Crear_Entrenamiento_Prueba(X) 
        y_train, y_test = Crear_Entrenamiento_Prueba(y)
        y_train = np.reshape(y_train, y_train.shape[0])
        y_test = np.reshape(y_test, y_test.shape[0])
        W = Crear_Pesos(X)
        num_iter = 100
        learn_rate = 0.1
        W, costs = Gradiente_Descendiente(X_train, y_train, W, num_iter, learn_rate)
        plt.plot(range(len(costs)), costs)
        print("Archivo: ", name_data_file)
        print("Pesos de Gradiente Descendiente: ", W, "\n")
        plt.show()
    print("\n")


Experimento1()
#Experimento2()
#Experimento3()
#Experimento4()