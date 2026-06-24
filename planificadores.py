import numpy as np

def obtener_ruta_coordenadas(viewpoints, camino):
    """
    Convierte un camino de índices en una ruta de coordenadas 3D.

    La función recibe un conjunto de viewpoints y una lista de índices que
    representa el orden de visita de una ruta. A partir de esos índices, extrae
    las coordenadas 3D correspondientes y devuelve la trayectoria final como un
    array de NumPy.

    Parámetros
    ----------
    viewpoints : np.ndarray de shape (N, 3)
        Conjunto completo de puntos de vista disponibles. Cada fila representa
        un viewpoint con coordenadas (x, y, z).

    camino : list de int
        Lista de índices que define el orden de visita de los viewpoints. Cada
        índice debe corresponder a una fila válida de viewpoints.

    Retorna
    -------
    ruta_3d : np.ndarray de shape (K, 3)
        Secuencia de coordenadas 3D correspondiente al camino indicado. K es el
        número de índices contenidos en camino.

    Notas
    -----
    Esta función no calcula la ruta óptima ni comprueba la validez del camino.
    Únicamente transforma una solución expresada como índices en una secuencia
    de coordenadas 3D apta para visualización o exportación.
    """
    return np.asarray([viewpoints[i] for i in camino])

def comprobar_conectividad(dist_matrix):
    """
    Comprueba si un grafo representado mediante matriz de distancias es conexo.

    La función realiza un recorrido en profundidad desde el primer nodo del grafo. 
    A partir de la matriz de distancias, considera como vecinos aquellos nodos
    cuya distancia sea finita. Si desde el nodo inicial se pueden visitar todos
    los nodos, el grafo se considera conexo.

    Parámetros
    ----------
    dist_matrix : np.ndarray de shape (N, N)
        Matriz de distancias del grafo. Las posiciones con valores finitos
        representan aristas existentes entre nodos. Las posiciones con np.inf
        representan ausencia de conexión directa.

    Retorna
    -------
    es_conexo : bool
        True si todos los nodos del grafo son alcanzables desde el nodo inicial.
        False en caso contrario.

    visitados : set
        Conjunto de índices de los nodos alcanzables desde el nodo inicial.

    Notas
    -----
    La diagonal de la matriz suele contener ceros, ya que representa la distancia
    de cada nodo consigo mismo. Estos valores se ignoran al buscar vecinos.

    Esta función asume que el grafo no está vacío. Si se utiliza con una matriz
    de tamaño cero, sería necesario añadir una comprobación previa.
    """
    n = dist_matrix.shape[0]
    visitados = set()
    pila = [0]

    while pila:
        nodo = pila.pop()

        if nodo in visitados:
            continue

        visitados.add(nodo)

        vecinos = np.where(np.isfinite(dist_matrix[nodo]))[0]
        vecinos = [v for v in vecinos if v != nodo]

        for v in vecinos:
            if v not in visitados:
                pila.append(v)

    return len(visitados) == n, visitados

def aplicar_aco(
    dist_matrix,
    n_hormigas=40,
    n_iteraciones=150,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.4,
    q=1.0,
    ciclo_cerrado=False,
    seed=17,
    verbose=True
):
    """
    Aplica el algoritmo Ant Colony Optimization sobre un grafo de viewpoints.

    La función busca una ruta de bajo coste que visite todos los nodos del grafo
    representado mediante una matriz de distancias. Cada nodo corresponde a un
    viewpoint y cada arista válida representa una conexión directa posible entre
    dos viewpoints.

    El algoritmo simula el comportamiento de varias hormigas artificiales. Cada
    hormiga construye una ruta seleccionando probabilísticamente el siguiente
    nodo a visitar en función de dos factores: la cantidad de feromona acumulada
    en la arista y la heurística asociada a la distancia. Las aristas más cortas
    y con más feromona tienen mayor probabilidad de ser elegidas.

    Al final de cada iteración, las feromonas se evaporan parcialmente y las
    rutas válidas depositan nueva feromona en las aristas utilizadas. De esta
    forma, las mejores rutas tienden a reforzarse progresivamente durante el
    proceso de búsqueda.

    Parámetros
    ----------
    dist_matrix : np.ndarray de shape (N, N)
        Matriz de distancias del grafo. La posición dist_matrix[i, j] contiene
        la distancia entre los nodos i y j si existe conexión directa. Si no hay
        conexión, contiene np.inf. La diagonal principal contiene ceros.

    n_hormigas : int, opcional
        Número de hormigas artificiales utilizadas en cada iteración. Un valor
        mayor permite explorar más rutas por iteración, aunque aumenta el coste
        computacional. Por defecto es 40.

    n_iteraciones : int, opcional
        Número total de iteraciones del algoritmo. A mayor número de iteraciones,
        mayor oportunidad de mejorar la solución, aunque también aumenta el
        tiempo de ejecución. Por defecto es 150.

    alpha : float, opcional
        Peso de la feromona en la decisión de cada hormiga. Valores altos hacen
        que las hormigas tiendan a seguir más intensamente las aristas ya
        reforzadas por soluciones anteriores. Por defecto es 1.0.

    beta : float, opcional
        Peso de la heurística basada en la distancia. Valores altos hacen que
        las hormigas prioricen más las conexiones cortas. Por defecto es 4.0.

    evaporacion : float, opcional
        Tasa de evaporación de feromonas en cada iteración. Debe estar entre
        0.0 y 1.0. Valores altos reducen más rápido la influencia de rutas
        anteriores, favoreciendo la exploración. Por defecto es 0.4.

    q : float, opcional
        Factor de depósito de feromona. Controla la cantidad de feromona añadida
        por cada ruta válida. El depósito aplicado es q / distancia_total.
        Por defecto es 1.0.

    ciclo_cerrado : bool, opcional
        Si es True, la ruta debe volver al nodo inicial al finalizar el recorrido,
        formando un ciclo cerrado. Si es False, la ruta termina en el último nodo
        visitado sin regresar al inicio. Por defecto es False.

    seed : int, opcional
        Semilla utilizada para inicializar el generador aleatorio de NumPy.
        Permite obtener resultados reproducibles. Por defecto es 17.

    verbose : bool, opcional
        Si es True, muestra por consola el progreso del algoritmo cada 10
        iteraciones, incluyendo la mejor distancia encontrada hasta el momento.
        Por defecto es True.

    Retorna
    -------
    mejor_camino : list de int
        Lista de índices que representa el mejor camino encontrado. Cada índice
        corresponde a un viewpoint. Si ciclo_cerrado es True, el primer nodo
        aparece también al final de la lista.

    mejor_distancia : float
        Distancia total asociada al mejor camino encontrado.

    Lanza
    -----
    ValueError
        Si el grafo no es conexo, es decir, si no todos los nodos son alcanzables
        desde el nodo inicial. En ese caso, no es posible construir una ruta que
        visite todos los viewpoints.

    Notas
    -----
    Antes de ejecutar el algoritmo, se comprueba que el grafo sea conexo mediante
    la función comprobar_conectividad(). Si el grafo no es conexo, se recomienda
    aumentar el número de vecinos usados al construir el grafo o generar más
    puntos intermedios.

    La heurística utilizada es la inversa de la distancia: 1 / distancia. Por
    tanto, las aristas más cortas tienen un valor heurístico mayor.

    Las aristas inexistentes se representan con np.inf y no participan en la
    selección de candidatos ni en la actualización de feromonas.

    Esta implementación resuelve una variante del problema de visitar todos los
    viewpoints. Si ciclo_cerrado es False, se obtiene una ruta abierta. Si
    ciclo_cerrado es True, se obtiene una ruta cerrada similar al problema del
    viajante.
    """
    rng = np.random.default_rng(seed)

    n = dist_matrix.shape[0]

    conectado, visitados = comprobar_conectividad(dist_matrix)

    if not conectado:
        raise ValueError(
            f"El grafo no es conexo. Solo se alcanzan {len(visitados)} de {n} nodos. "
            f"Prueba a aumentar k_vecinos o a generar más puntos intermedios."
        )

    mascara_aristas = np.isfinite(dist_matrix) & (dist_matrix > 0)

    heuristica = np.zeros_like(dist_matrix)
    heuristica[mascara_aristas] = 1.0 / dist_matrix[mascara_aristas]

    feromonas = np.zeros_like(dist_matrix)
    feromonas[mascara_aristas] = 1.0

    mejor_camino = None
    mejor_distancia = np.inf

    for it in range(n_iteraciones):
        caminos_iteracion = []

        for _ in range(n_hormigas):
            inicio = rng.integers(0, n)

            camino = [inicio]
            no_visitados = set(range(n))
            no_visitados.remove(inicio)

            actual = inicio
            distancia_total = 0.0
            valido = True

            while no_visitados:
                candidatos = [
                    j for j in no_visitados
                    if np.isfinite(dist_matrix[actual, j])
                ]

                if len(candidatos) == 0:
                    valido = False
                    break

                valores = []

                for j in candidatos:
                    tau = feromonas[actual, j] ** alpha
                    eta = heuristica[actual, j] ** beta
                    valores.append(tau * eta)

                valores = np.asarray(valores, dtype=float)

                if valores.sum() == 0:
                    probabilidades = np.ones(len(candidatos)) / len(candidatos)
                else:
                    probabilidades = valores / valores.sum()

                siguiente = rng.choice(candidatos, p=probabilidades)

                distancia_total += dist_matrix[actual, siguiente]
                camino.append(siguiente)
                no_visitados.remove(siguiente)

                actual = siguiente

            if valido and ciclo_cerrado:
                inicio = camino[0]

                if np.isfinite(dist_matrix[actual, inicio]):
                    distancia_total += dist_matrix[actual, inicio]
                    camino.append(inicio)
                else:
                    valido = False

            if valido:
                caminos_iteracion.append((camino, distancia_total))

                if distancia_total < mejor_distancia:
                    mejor_distancia = distancia_total
                    mejor_camino = camino

        feromonas *= (1.0 - evaporacion)

        for camino, distancia in caminos_iteracion:
            deposito = q / distancia

            for a, b in zip(camino[:-1], camino[1:]):
                feromonas[a, b] += deposito
                feromonas[b, a] += deposito

        if verbose and (it + 1) % 10 == 0:
            print(
                f"Iteración {it + 1}/{n_iteraciones} | "
                f"Mejor distancia: {mejor_distancia:.4f}"
            )

    return mejor_camino, mejor_distancia

def calcular_distancia_camino(dist_matrix, camino, ciclo_cerrado=False):
    """
    Calcula la distancia total de un camino sobre un grafo de viewpoints.

    La función recibe una matriz de distancias y una secuencia de índices que
    representa un camino. Recorre cada par de nodos consecutivos del camino y
    acumula la distancia correspondiente en la matriz.

    Si alguna conexión entre dos nodos consecutivos no existe, es decir, si su
    distancia es np.inf, la función devuelve np.inf para indicar que el camino
    no es válido.

    Opcionalmente, puede calcularse la distancia como ciclo cerrado, añadiendo
    también la conexión desde el último nodo del camino hasta el primero.

    Parámetros
    ----------
    dist_matrix : np.ndarray de shape (N, N)
        Matriz de distancias del grafo. La posición dist_matrix[i, j] contiene
        la distancia entre los nodos i y j si existe conexión directa. Si no hay
        conexión, debe contener np.inf.

    camino : list de int
        Lista de índices que representa el orden de visita de los nodos del
        grafo. Cada índice debe corresponder a una fila y columna válidas de
        dist_matrix.

    ciclo_cerrado : bool, opcional
        Si es True, se añade al cálculo la distancia desde el último nodo del
        camino hasta el primer nodo, formando un ciclo cerrado. Si es False, se
        calcula únicamente la distancia del camino abierto. Por defecto es False.

    Retorna
    -------
    distancia : float
        Distancia total del camino. Devuelve np.inf si alguna de las conexiones
        necesarias no existe en la matriz de distancias.
    """
    distancia = 0.0

    for a, b in zip(camino[:-1], camino[1:]):
        d = dist_matrix[a, b]

        if not np.isfinite(d):
            return np.inf

        distancia += d

    if ciclo_cerrado:
        d = dist_matrix[camino[-1], camino[0]]

        if not np.isfinite(d):
            return np.inf

        distancia += d

    return distancia

def crear_camino_greedy_aleatorio(dist_matrix, rng, max_intentos=200):
    """
    Crea una ruta inicial intentando seguir conexiones válidas.
    Si no consigue una ruta completamente válida, devuelve la mejor parcial
    completada aleatoriamente.
    """
    n = dist_matrix.shape[0]

    mejor_camino = None
    mejor_visitados = -1

    for _ in range(max_intentos):
        inicio = rng.integers(0, n)
        camino = [inicio]

        no_visitados = set(range(n))
        no_visitados.remove(inicio)

        actual = inicio
        valido = True

        while no_visitados:
            candidatos = [
                j for j in no_visitados
                if np.isfinite(dist_matrix[actual, j])
            ]

            if len(candidatos) == 0:
                valido = False
                break

            distancias = np.array([dist_matrix[actual, j] for j in candidatos])
            pesos = 1.0 / (distancias + 1e-12)
            probabilidades = pesos / pesos.sum()

            siguiente = rng.choice(candidatos, p=probabilidades)

            camino.append(siguiente)
            no_visitados.remove(siguiente)
            actual = siguiente

        if valido:
            return camino

        if len(camino) > mejor_visitados:
            mejor_visitados = len(camino)
            mejor_camino = camino.copy()

    restantes = [i for i in range(n) if i not in mejor_camino]
    rng.shuffle(restantes)

    return mejor_camino + restantes

def aplicar_ga(
    dist_matrix,
    n_poblacion=60,
    n_generaciones=200,
    prob_cruce=0.85,
    prob_mutacion=0.25,
    elite=2,
    tam_torneo=3,
    ciclo_cerrado=False,
    seed=17,
    verbose=True
):
    """
    Aplica un Algoritmo Genético para resolver una variante abierta o cerrada
    del TSP sobre un grafo representado mediante una matriz de distancias.

    Cada individuo representa una ruta como una permutación de nodos. El objetivo
    es encontrar el orden de visita que minimiza la distancia total recorrida.

    Parámetros
    ----------
    dist_matrix : np.ndarray
        Matriz de distancias del grafo. Las aristas no válidas deben estar
        representadas con np.inf.

    n_poblacion : int
        Número de individuos de la población.

    n_generaciones : int
        Número de generaciones del algoritmo.

    prob_cruce : float
        Probabilidad de aplicar cruce entre dos padres.

    prob_mutacion : float
        Probabilidad de aplicar mutación a un individuo.

    elite : int
        Número de mejores individuos que pasan directamente a la siguiente
        generación.

    tam_torneo : int
        Tamaño del torneo usado para seleccionar padres.

    ciclo_cerrado : bool
        Si es True, la ruta vuelve al nodo inicial. Si es False, la ruta es abierta.

    seed : int
        Semilla para reproducibilidad.

    verbose : bool
        Si es True, muestra el progreso cada 10 generaciones.

    Retorna
    -------
    mejor_camino : list de int
        Mejor ruta encontrada.

    mejor_distancia : float
        Distancia total de la mejor ruta.
    """

    rng = np.random.default_rng(seed)

    n = dist_matrix.shape[0]

    conectado, visitados = comprobar_conectividad(dist_matrix)

    if not conectado:
        raise ValueError(
            f"El grafo no es conexo. Solo se alcanzan {len(visitados)} de {n} nodos. "
            f"Prueba a aumentar k_vecinos o a generar más puntos intermedios."
        )

    def evaluar(individuo):
        return calcular_distancia_camino(
            dist_matrix,
            individuo,
            ciclo_cerrado=ciclo_cerrado
        )

    def seleccionar_torneo(poblacion, distancias):
        """
        Selecciona un individuo mediante torneo.
        """
        indices = rng.choice(len(poblacion), size=tam_torneo, replace=False)
        mejor_idx = indices[np.argmin([distancias[i] for i in indices])]
        return poblacion[mejor_idx].copy()

    def cruce_ox(padre1, padre2):
        """
        Order Crossover (OX), adecuado para permutaciones.
        Mantiene un segmento del padre1 y completa con el orden del padre2.
        """
        size = len(padre1)

        i, j = sorted(rng.choice(size, size=2, replace=False))

        hijo = [-1] * size
        hijo[i:j + 1] = padre1[i:j + 1]

        pos = (j + 1) % size

        for gen in padre2:
            if gen not in hijo:
                hijo[pos] = gen
                pos = (pos + 1) % size

        return hijo

    def mutar(individuo):
        """
        Aplica una mutación aleatoria sobre la ruta.
        Usa swap, inserción o inversión de subsecuencia.
        """
        individuo = individuo.copy()
        tipo = rng.choice(["swap", "insert", "reverse"])

        i, j = sorted(rng.choice(len(individuo), size=2, replace=False))

        if tipo == "swap":
            individuo[i], individuo[j] = individuo[j], individuo[i]

        elif tipo == "insert":
            nodo = individuo.pop(j)
            individuo.insert(i, nodo)

        elif tipo == "reverse":
            individuo[i:j + 1] = reversed(individuo[i:j + 1])

        return individuo

    # ------------------------------------------------------------
    # Inicialización de población
    # ------------------------------------------------------------
    poblacion = [crear_camino_greedy_aleatorio(dist_matrix, rng) for _ in range(n_poblacion)]

    mejor_camino = None
    mejor_distancia = np.inf

    # ------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------
    for gen in range(n_generaciones):
        distancias = np.array([evaluar(ind) for ind in poblacion])
        validos = np.sum(np.isfinite(distancias))

        idx_mejor = np.argmin(distancias)

        if distancias[idx_mejor] < mejor_distancia:
            mejor_distancia = distancias[idx_mejor]
            mejor_camino = poblacion[idx_mejor].copy()

        # Ordenar población por calidad
        indices_ordenados = np.argsort(distancias)

        nueva_poblacion = []

        # Elitismo
        for idx in indices_ordenados[:elite]:
            nueva_poblacion.append(poblacion[idx].copy())

        # Generar nuevos individuos
        while len(nueva_poblacion) < n_poblacion:
            padre1 = seleccionar_torneo(poblacion, distancias)
            padre2 = seleccionar_torneo(poblacion, distancias)

            if rng.random() < prob_cruce:
                hijo = cruce_ox(padre1, padre2)
            else:
                hijo = padre1.copy()

            if rng.random() < prob_mutacion:
                hijo = mutar(hijo)

            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        if verbose and (gen + 1) % 10 == 0:
            print(
                f"Generación {gen + 1}/{n_generaciones} | "
                f"Mejor distancia: {mejor_distancia:.4f}"# | "
                # f"Rutas válidas: {validos}/{n_poblacion}"
            )

    return mejor_camino, mejor_distancia

def aplicar_abc_discreto(
    dist_matrix,
    n_fuentes=50,
    n_iteraciones=300,
    limite=40,
    prob_2opt=0.25,
    ciclo_cerrado=False,
    seed=17,
    verbose=True
):
    """
    Aplica Artificial Bee Colony discreto para resolver una variante abierta
    o cerrada del TSP sobre un grafo representado mediante matriz de distancias.

    Cada fuente de alimento representa una ruta completa, codificada como una
    permutación de nodos. La calidad de cada fuente se evalúa mediante la
    distancia total del camino.

    El algoritmo utiliza tres fases:
    1. Abejas empleadas: exploran vecinos de sus fuentes actuales.
    2. Abejas observadoras: seleccionan mejores fuentes y exploran sus vecinos.
    3. Abejas exploradoras: sustituyen fuentes estancadas por nuevas rutas.

    Parámetros
    ----------
    dist_matrix : np.ndarray
        Matriz de distancias del grafo. Las aristas inexistentes deben estar
        representadas con np.inf.

    n_fuentes : int
        Número de fuentes de alimento, equivalente al tamaño de población.

    n_iteraciones : int
        Número total de iteraciones del algoritmo.

    limite : int
        Número máximo de intentos sin mejora antes de abandonar una fuente.

    prob_2opt : float
        Probabilidad de aplicar un movimiento tipo 2-opt como operador local.

    ciclo_cerrado : bool
        Si es True, la ruta vuelve al nodo inicial. Si es False, la ruta es abierta.

    seed : int
        Semilla para reproducibilidad.

    verbose : bool
        Si es True, muestra progreso cada 10 iteraciones.

    Retorna
    -------
    mejor_camino : list de int
        Mejor ruta encontrada.

    mejor_distancia : float
        Distancia total de la mejor ruta.
    """

    rng = np.random.default_rng(seed)

    n = dist_matrix.shape[0]

    conectado, visitados = comprobar_conectividad(dist_matrix)

    if not conectado:
        raise ValueError(
            f"El grafo no es conexo. Solo se alcanzan {len(visitados)} de {n} nodos. "
            f"Prueba a aumentar k_vecinos o a generar más puntos intermedios."
        )

    def evaluar(camino):
        return calcular_distancia_camino(
            dist_matrix,
            camino,
            ciclo_cerrado=ciclo_cerrado
        )

    def generar_vecino(camino):
        """
        Genera una ruta vecina mediante operadores discretos:
        swap, insert, reverse o 2-opt.
        """
        vecino = camino.copy()

        if len(vecino) < 3:
            return vecino

        if rng.random() < prob_2opt:
            tipo = "2opt"
        else:
            tipo = rng.choice(["swap", "insert", "reverse"])

        i, j = sorted(rng.choice(len(vecino), size=2, replace=False))

        if tipo == "swap":
            vecino[i], vecino[j] = vecino[j], vecino[i]

        elif tipo == "insert":
            nodo = vecino.pop(j)
            vecino.insert(i, nodo)

        elif tipo == "reverse":
            vecino[i:j + 1] = reversed(vecino[i:j + 1])

        elif tipo == "2opt":
            vecino[i:j + 1] = reversed(vecino[i:j + 1])

        return vecino

    def fitness_desde_distancia(distancia):
        """
        Convierte una distancia en fitness positivo.
        Las rutas inválidas reciben fitness 0.
        """
        if not np.isfinite(distancia):
            return 0.0

        return 1.0 / (1.0 + distancia)

    # ------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------
    fuentes = [crear_camino_greedy_aleatorio(dist_matrix, rng) for _ in range(n_fuentes)]
    distancias = np.array([evaluar(fuente) for fuente in fuentes], dtype=float)
    intentos_sin_mejora = np.zeros(n_fuentes, dtype=int)

    idx_mejor = np.argmin(distancias)
    mejor_camino = fuentes[idx_mejor].copy()
    mejor_distancia = distancias[idx_mejor]

    # ------------------------------------------------------------
    # Bucle principal ABC
    # ------------------------------------------------------------
    for it in range(n_iteraciones):

        # ========================================================
        # 1. FASE DE ABEJAS EMPLEADAS
        # ========================================================
        for i in range(n_fuentes):
            candidata = generar_vecino(fuentes[i])
            distancia_candidata = evaluar(candidata)

            if distancia_candidata < distancias[i]:
                fuentes[i] = candidata
                distancias[i] = distancia_candidata
                intentos_sin_mejora[i] = 0
            else:
                intentos_sin_mejora[i] += 1

        # ========================================================
        # 2. FASE DE ABEJAS OBSERVADORAS
        # ========================================================
        fitness = np.array(
            [fitness_desde_distancia(d) for d in distancias],
            dtype=float
        )

        if fitness.sum() == 0:
            probabilidades = np.ones(n_fuentes) / n_fuentes
        else:
            probabilidades = fitness / fitness.sum()

        for _ in range(n_fuentes):
            i = rng.choice(np.arange(n_fuentes), p=probabilidades)

            candidata = generar_vecino(fuentes[i])
            distancia_candidata = evaluar(candidata)

            if distancia_candidata < distancias[i]:
                fuentes[i] = candidata
                distancias[i] = distancia_candidata
                intentos_sin_mejora[i] = 0
            else:
                intentos_sin_mejora[i] += 1

        # ========================================================
        # 3. FASE DE ABEJAS EXPLORADORAS
        # ========================================================
        for i in range(n_fuentes):
            if intentos_sin_mejora[i] >= limite:
                fuentes[i] = crear_camino_greedy_aleatorio(dist_matrix, rng)
                distancias[i] = evaluar(fuentes[i])
                intentos_sin_mejora[i] = 0

        # ========================================================
        # Actualizar mejor solución global
        # ========================================================
        idx_mejor_iteracion = np.argmin(distancias)

        if distancias[idx_mejor_iteracion] < mejor_distancia:
            mejor_distancia = distancias[idx_mejor_iteracion]
            mejor_camino = fuentes[idx_mejor_iteracion].copy()

        if verbose and (it + 1) % 10 == 0:
            validas = np.sum(np.isfinite(distancias))

            print(
                f"Iteración {it + 1}/{n_iteraciones} | "
                f"Mejor distancia: {mejor_distancia:.4f}"# | "
                # f"Rutas válidas: {validas}/{n_fuentes}"
            )

    return mejor_camino, mejor_distancia