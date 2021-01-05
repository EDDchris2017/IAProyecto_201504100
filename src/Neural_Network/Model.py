import numpy as np
#np.set_printoptions(threshold=100000) #Esto es para que al imprimir un arreglo no me muestre puntos suspensivos


class NN_Model:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        self.capas = len(layers) - 1
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        #print('layers:', layers)
        for l in range(1, L):
            #np.random.randn(layers[l], layers[l-1])
            #Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            #np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))
            #print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            #print(np.sqrt(layers[l-1]))
            #print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data, self.capas)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp, self.capas)
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')


    def propagacion_adelante(self, dataSet, capas):
        # Se extraen las entradas
        X = dataSet.x
        Ares = X
        temp = {}

        # === CAPAS OCULTAS ===
        for i in range(1,capas):
            # Extraccion de Pesos
            WN = self.parametros["W"+str(i)]
            bn = self.parametros["b"+str(i)]

            activacion = "relu"
            #Funcion de activacion y dropout invertido
            ZN = np.dot(WN, Ares) + bn
            AN = self.activation_function(activacion, ZN)

            ANmenos = Ares
            if i == 1 : ANmenos = AN
            #Se aplica el Dropout invertido
            DN = np.random.rand(AN.shape[0], ANmenos.shape[1]) #Se generan número aleatorios para cada neurona
            DN = (DN < self.kp).astype(int) #Mientras más alto es kp mayor la probabilidad de que la neurona permanezca
            AN *= DN
            AN /= self.kp
            # Guardar ultimo AN
            Ares = AN
            temp["Z"+str(i)] = ZN
            temp["A"+str(i)] = AN
            temp["D"+str(i)] = DN
        
        # === CAPA SALIDA ===
        Wres = self.parametros["W"+str(capas)]
        bres = self.parametros["b"+str(capas)]
        activacion = 'sigmoide'
        Zres = np.dot(Wres, Ares) + bres
        Ares = self.activation_function(activacion, Zres)
        temp["Z"+str(capas)] = Zres
        temp["A"+str(capas)] = Ares

        #En A_res va la predicción o el resultado de la red neuronal
        return Ares, temp

    def propagacion_atras(self, temp, capas):
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x

        # === GRADIENTES ===
        gradientes = {}

        # === CAPA SALIDA ===
        Ares  = temp["A"+str(capas)]
        Wres  = self.parametros["W"+str(capas)]
        dZres = Ares - Y
        dWres = ( 1 / m) * np.dot(dZres, temp["A"+str(capas-1)].T) + (self.lambd / m) * Wres
        dbres = (1 / m) * np.sum(dZres, axis=1, keepdims=True)
        gradientes["dZ"+str(capas)] = dZres
        gradientes["dW"+str(capas)] = dWres
        gradientes["db"+str(capas)] = dbres

        # === CAPAS OCULTAS ===
        for i in reversed(range(1,capas)):
            WN = self.parametros["W"+str(i)]
            WNmas = self.parametros["W"+str(i+1)]
            dZmas = gradientes["dZ"+str(i + 1)]
            DN    = temp["D"+str(i)]
            AN    = temp["A"+str(i)]
            ANmenos = None
            # Verificar si es capa de entrada
            if i > 1 : ANmenos = temp["A"+str(i-1)] 
            else: ANmenos = X
            dAN = np.dot(WNmas.T, dZmas)
            dAN *= DN
            dAN /= self.kp
            dZN = np.multiply(dAN, np.int64(AN > 0))
            dWN = 1. / m * np.dot(dZN, ANmenos.T) + (self.lambd / m) * WN
            dbN = 1. / m * np.sum(dZN, axis=1, keepdims=True)

            #Guardar gradientes
            gradientes["dA"+str(i)] = dAN
            gradientes["dZ"+str(i)] = dZN
            gradientes["dW"+str(i)] = dWN
            gradientes["db"+str(i)] = dbN
        
        return gradientes

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd/(2*m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet, self.capas)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0, ]))
        #print("Exactitud: " + str(exactitud))
        return exactitud


    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1/(1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)
        
        #print('name:', name, 'result:', result)
        return result