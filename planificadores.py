import numpy as np

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