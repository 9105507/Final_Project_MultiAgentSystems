import numpy as np
import time

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

def aplicar_simulated_annealing(
    dist_matrix,
    n_iteraciones=10000,
    temperatura_inicial=None,
    temperatura_final=1e-4,
    enfriamiento=0.995,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,
    max_intentos_inicial=500,
    max_intentos_vecino=30
):
    """
    Aplica Simulated Annealing para resolver una variante abierta o cerrada
    del TSP sobre un grafo representado mediante matriz de distancias.

    Cada solución es una permutación de nodos. En cada iteración se genera
    una solución vecina mediante operadores discretos como swap, insert o
    reverse. Si el vecino mejora, se acepta. Si empeora, puede aceptarse con
    una probabilidad decreciente según la temperatura.

    Parámetros
    ----------
    dist_matrix : np.ndarray
        Matriz de distancias del grafo. Las aristas inexistentes deben estar
        representadas con np.inf.

    n_iteraciones : int
        Número máximo de iteraciones del algoritmo.

    temperatura_inicial : float o None
        Temperatura inicial. Si es None, se estima automáticamente a partir
        de distancias finitas del grafo.

    temperatura_final : float
        Temperatura mínima. Si se alcanza, el algoritmo se detiene.

    enfriamiento : float
        Factor de enfriamiento multiplicativo. Debe estar entre 0 y 1.

    ciclo_cerrado : bool
        Si es True, la ruta vuelve al nodo inicial. Si es False, se resuelve
        como TSP abierto.

    seed : int
        Semilla para reproducibilidad.

    verbose : bool
        Si es True, muestra progreso cada cierto número de iteraciones.

    max_intentos_inicial : int
        Número máximo de intentos para generar una ruta inicial válida.

    max_intentos_vecino : int
        Número máximo de intentos para generar un vecino válido antes de
        conservar la solución actual.

    Retorna
    -------
    mejor_camino : list de int
        Mejor camino encontrado.

    mejor_distancia : float
        Distancia total del mejor camino.
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

    def crear_camino_greedy_valido():
        """
        Intenta construir una ruta válida siguiendo conexiones existentes.
        Si no encuentra una ruta completa válida tras varios intentos,
        devuelve None.
        """
        for _ in range(max_intentos_inicial):
            inicio = int(rng.integers(0, n))

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

                distancias = np.array(
                    [dist_matrix[actual, j] for j in candidatos],
                    dtype=float
                )

                pesos = 1.0 / (distancias + 1e-12)
                probabilidades = pesos / pesos.sum()

                siguiente = int(rng.choice(candidatos, p=probabilidades))

                camino.append(siguiente)
                no_visitados.remove(siguiente)
                actual = siguiente

            if valido:
                distancia = evaluar(camino)

                if np.isfinite(distancia):
                    return camino, distancia

        return None, np.inf

    def generar_vecino(camino):
        """
        Genera un vecino mediante un movimiento discreto.
        """
        vecino = camino.copy()

        if ciclo_cerrado and len(vecino) > 1 and vecino[0] == vecino[-1]:
            base = vecino[:-1]
        else:
            base = vecino.copy()

        if len(base) < 3:
            return vecino

        tipo = rng.choice(["swap", "insert", "reverse"])

        i, j = sorted(rng.choice(len(base), size=2, replace=False))

        if tipo == "swap":
            base[i], base[j] = base[j], base[i]

        elif tipo == "insert":
            nodo = base.pop(j)
            base.insert(i, nodo)

        elif tipo == "reverse":
            base[i:j + 1] = reversed(base[i:j + 1])

        if ciclo_cerrado:
            return base + [base[0]]

        return base

    # ------------------------------------------------------------
    # Solución inicial
    # ------------------------------------------------------------
    camino_actual, distancia_actual = crear_camino_greedy_valido()

    if camino_actual is None or not np.isfinite(distancia_actual):
        raise ValueError(
            "No se ha podido generar una ruta inicial válida para Simulated Annealing. "
            "Prueba a aumentar k_vecinos o max_intentos_inicial."
        )

    mejor_camino = camino_actual.copy()
    mejor_distancia = distancia_actual

    # ------------------------------------------------------------
    # Temperatura inicial automática
    # ------------------------------------------------------------
    if temperatura_inicial is None:
        valores_finitos = dist_matrix[
            np.isfinite(dist_matrix) & (dist_matrix > 0)
        ]

        if len(valores_finitos) == 0:
            raise ValueError("No hay aristas válidas en la matriz de distancias.")

        temperatura = float(np.mean(valores_finitos) * n)
    else:
        temperatura = float(temperatura_inicial)

    if temperatura <= 0:
        raise ValueError("La temperatura inicial debe ser positiva.")

    if not (0 < enfriamiento < 1):
        raise ValueError("El enfriamiento debe estar entre 0 y 1.")

    # ------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------
    aceptadas = 0

    historial = []
    tiempo_inicio_total = time.perf_counter()

    for it in range(n_iteraciones):
        tiempo_inicio_iteracion = time.perf_counter()

        vecino = None
        distancia_vecino = np.inf
        aceptada_iteracion = False

        # Intentamos generar un vecino válido
        for _ in range(max_intentos_vecino):
            candidato = generar_vecino(camino_actual)
            d = evaluar(candidato)

            if np.isfinite(d):
                vecino = candidato
                distancia_vecino = d
                break

        # Si no se ha conseguido vecino válido, enfriamos y seguimos
        if vecino is None:
            temperatura *= enfriamiento

            if temperatura < temperatura_final:
                break

            continue

        diferencia = distancia_vecino - distancia_actual

        aceptar = False

        if diferencia < 0:
            aceptar = True
        else:
            probabilidad = np.exp(-diferencia / temperatura)
            aceptar = rng.random() < probabilidad

        if aceptar:
            camino_actual = vecino
            distancia_actual = distancia_vecino
            aceptadas += 1

            if distancia_actual < mejor_distancia:
                mejor_distancia = distancia_actual
                mejor_camino = camino_actual.copy()

        temperatura *= enfriamiento

        tiempo_iteracion = time.perf_counter() - tiempo_inicio_iteracion
        tiempo_total = time.perf_counter() - tiempo_inicio_total

        historial.append({
            "algoritmo": "SA",
            "iteracion": it + 1,
            "mejor_distancia_global": float(mejor_distancia),
            "distancia_actual": float(distancia_actual),
            "distancia_vecino": float(distancia_vecino),
            "temperatura": float(temperatura),
            "aceptadas": int(aceptadas),
            "aceptada_iteracion": bool(aceptada_iteracion),
            "tiempo_iteracion": float(tiempo_iteracion),
            "tiempo_total": float(tiempo_total)
        })

        if verbose and (it + 1) % 500 == 0:
            print(
                f"Iteración {it + 1}/{n_iteraciones} | "
                f"Mejor distancia: {mejor_distancia:.4f} | "
                f"Actual: {distancia_actual:.4f} | "
                f"Temperatura: {temperatura:.6f} | "
                f"Aceptadas: {aceptadas} | "
                f"Tiempo iteración: {tiempo_iteracion:.4f}s"
            )

        if temperatura < temperatura_final:
            if verbose:
                print(
                    f"Parada por temperatura mínima en iteración {it + 1}. "
                    f"Mejor distancia: {mejor_distancia:.4f}"
                )
            break

    if mejor_camino is None or not np.isfinite(mejor_distancia):
        raise ValueError(
            "Simulated Annealing no ha encontrado ninguna ruta válida."
        )

    return mejor_camino, mejor_distancia, historial

def aplicar_aco_variante(
    dist_matrix,
    variante="AS",
    n_hormigas=None,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=None,
    q=1.0,
    tau0=None,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,

    # Parámetros específicos EAS
    peso_elite=None,

    # Parámetros específicos ASrank
    w_rank=6,

    # Parámetros específicos MMAS
    tau_min=None,
    tau_max=None,
    mmas_usar_mejor_global=True,

    # Parámetros específicos ACS
    phi=0.1,
    q0=0.9,
    acs_usar_mejor_global=True,

    # Búsqueda local opcional
    aplicar_2opt=False,
    n_intentos_2opt=80
):
    """
    Aplica diferentes variantes de Ant Colony Optimization para resolver una
    variante abierta o cerrada del TSP sobre un grafo representado mediante una
    matriz de distancias.

    Variantes disponibles
    ---------------------
    "AS"     : Ant System.
    "EAS"    : Elitist Ant System.
    "ASRANK" : Rank-Based Ant System.
    "MMAS"   : MAX-MIN Ant System.
    "ACS"    : Ant Colony System.

    Parámetros
    ----------
    dist_matrix : np.ndarray de shape (N, N)
        Matriz de distancias. Las aristas inexistentes deben estar representadas
        con np.inf.

    variante : str
        Variante de ACO a ejecutar: "AS", "EAS", "ASRANK", "MMAS" o "ACS".

    n_hormigas : int o None
        Número de hormigas por iteración. Si es None, se usa N para AS/EAS/
        ASRANK/MMAS y 10 para ACS.

    n_iteraciones : int
        Número total de iteraciones.

    alpha : float
        Peso de la feromona.

    beta : float
        Peso de la heurística 1 / distancia.

    evaporacion : float o None
        Tasa de evaporación de feromonas. Si es None, se usan valores típicos:
        AS/EAS: 0.5, ASRANK: 0.1, MMAS: 0.02, ACS: 0.1.

    q : float
        Factor de depósito de feromona.

    tau0 : float o None
        Valor inicial de feromona. Si es None, se calcula automáticamente.

    ciclo_cerrado : bool
        Si es True, se fuerza regreso al nodo inicial. Si es False, se resuelve
        como TSP abierto.

    seed : int
        Semilla aleatoria.

    verbose : bool
        Si es True, muestra progreso cada 10 iteraciones.

    peso_elite : int o None
        Peso elitista para EAS. Si es None, se usa N.

    w_rank : int
        Número de hormigas que participan en ASrank. La mejor global también
        deposita feromona con peso w_rank.

    tau_min, tau_max : float o None
        Límites de feromona para MMAS. Si son None, se calculan dinámicamente.

    mmas_usar_mejor_global : bool
        Si es True, MMAS actualiza con la mejor global. Si es False, actualiza
        con la mejor de la iteración.

    phi : float
        Coeficiente de actualización local en ACS.

    q0 : float
        Factor de explotación en ACS. Si q < q0, se elige el mejor candidato;
        si no, se usa selección probabilística.

    acs_usar_mejor_global : bool
        Si es True, ACS actualiza con la mejor global. Si es False, con la mejor
        de la iteración.

    aplicar_2opt : bool
        Si es True, aplica una búsqueda local 2-opt limitada a las rutas válidas.

    n_intentos_2opt : int
        Número de intentos aleatorios de 2-opt por ruta válida.

    Retorna
    -------
    mejor_camino : list de int
        Mejor camino encontrado.

    mejor_distancia : float
        Distancia total del mejor camino.
    """

    rng = np.random.default_rng(seed)

    variante = variante.upper()

    if variante not in {"AS", "EAS", "ASRANK", "MMAS", "ACS"}:
        raise ValueError(
            "variante debe ser una de: 'AS', 'EAS', 'ASRANK', 'MMAS', 'ACS'"
        )

    n = dist_matrix.shape[0]

    conectado, visitados = comprobar_conectividad(dist_matrix)

    if not conectado:
        raise ValueError(
            f"El grafo no es conexo. Solo se alcanzan {len(visitados)} de {n} nodos. "
            f"Prueba a aumentar k_vecinos o a generar más puntos intermedios."
        )

    if n_hormigas is None:
        if variante == "ACS":
            n_hormigas = min(10, n)
        else:
            n_hormigas = n

    if evaporacion is None:
        if variante in {"AS", "EAS"}:
            evaporacion = 0.5
        elif variante == "ASRANK":
            evaporacion = 0.1
        elif variante == "MMAS":
            evaporacion = 0.02
        elif variante == "ACS":
            evaporacion = 0.1

    if peso_elite is None:
        peso_elite = n

    mascara_aristas = np.isfinite(dist_matrix) & (dist_matrix > 0)

    heuristica = np.zeros_like(dist_matrix, dtype=float)
    heuristica[mascara_aristas] = 1.0 / dist_matrix[mascara_aristas]

    def evaluar(camino):
        return calcular_distancia_camino(
            dist_matrix,
            camino,
            ciclo_cerrado=ciclo_cerrado
        )

    def estimar_distancia_nearest_neighbor():
        """
        Calcula una distancia inicial aproximada mediante nearest neighbor.
        Si no logra una ruta válida, devuelve una estimación basada en la media
        de aristas finitas.
        """
        mejor = np.inf

        for inicio in range(n):
            camino = [inicio]
            no_visitados = set(range(n))
            no_visitados.remove(inicio)
            actual = inicio

            while no_visitados:
                candidatos = [
                    j for j in no_visitados
                    if np.isfinite(dist_matrix[actual, j])
                ]

                if not candidatos:
                    break

                siguiente = min(
                    candidatos,
                    key=lambda j: dist_matrix[actual, j]
                )

                camino.append(siguiente)
                no_visitados.remove(siguiente)
                actual = siguiente

            if len(camino) == n:
                d = evaluar(camino)
                if d < mejor:
                    mejor = d

        if np.isfinite(mejor):
            return mejor

        valores_finitos = dist_matrix[mascara_aristas]

        if len(valores_finitos) == 0:
            raise ValueError("No hay aristas válidas en la matriz de distancias.")

        return float(np.mean(valores_finitos) * max(1, n - 1))

    c_nn = estimar_distancia_nearest_neighbor()

    if tau0 is None:
        if variante == "AS":
            tau0 = n_hormigas / c_nn
        elif variante == "EAS":
            tau0 = (peso_elite + n_hormigas) / c_nn
        elif variante == "ASRANK":
            tau0 = 0.5 * w_rank * (w_rank - 1) / c_nn
        elif variante == "MMAS":
            tau0 = 1.0 / (evaporacion * c_nn)
        elif variante == "ACS":
            tau0 = 1.0 / (n * c_nn)

    feromonas = np.zeros_like(dist_matrix, dtype=float)
    feromonas[mascara_aristas] = tau0

    def calcular_limites_mmas(mejor_distancia_actual):
        """
        Calcula tau_max y tau_min para MMAS.
        """
        if np.isfinite(mejor_distancia_actual):
            tmax = 1.0 / (evaporacion * mejor_distancia_actual)
        else:
            tmax = tau0

        grados = np.sum(mascara_aristas, axis=1)
        avg = float(np.mean(grados))

        if avg <= 1:
            tmin = tmax * 0.01
        else:
            p_dec = 0.05 ** (1.0 / n)
            tmin = tmax * (1.0 - p_dec) / ((avg - 1.0) * p_dec)

        if not np.isfinite(tmin) or tmin <= 0:
            tmin = tmax * 0.01

        return tmin, tmax

    def seleccionar_siguiente(actual, no_visitados):
        """
        Selección de siguiente nodo para AS, EAS, ASrank y MMAS.
        En ACS se usa también como parte probabilística cuando q >= q0.
        """
        candidatos = [
            j for j in no_visitados
            if np.isfinite(dist_matrix[actual, j])
        ]

        if len(candidatos) == 0:
            return None

        valores = []

        for j in candidatos:
            tau = feromonas[actual, j] ** alpha
            eta = heuristica[actual, j] ** beta
            valores.append(tau * eta)

        valores = np.asarray(valores, dtype=float)

        if valores.sum() == 0 or not np.isfinite(valores.sum()):
            probabilidades = np.ones(len(candidatos)) / len(candidatos)
        else:
            probabilidades = valores / valores.sum()

        return int(rng.choice(candidatos, p=probabilidades))

    def seleccionar_siguiente_acs(actual, no_visitados):
        """
        Regla pseudoaleatoria proporcional de ACS.
        """
        candidatos = [
            j for j in no_visitados
            if np.isfinite(dist_matrix[actual, j])
        ]

        if len(candidatos) == 0:
            return None

        valores = []

        for j in candidatos:
            tau = feromonas[actual, j] ** alpha
            eta = heuristica[actual, j] ** beta
            valores.append(tau * eta)

        valores = np.asarray(valores, dtype=float)

        if rng.random() < q0:
            return int(candidatos[np.argmax(valores)])

        if valores.sum() == 0 or not np.isfinite(valores.sum()):
            probabilidades = np.ones(len(candidatos)) / len(candidatos)
        else:
            probabilidades = valores / valores.sum()

        return int(rng.choice(candidatos, p=probabilidades))

    def actualizar_local_acs(a, b):
        """
        Actualización local de ACS:
        tau_ij = (1 - phi) * tau_ij + phi * tau0
        """
        feromonas[a, b] = (1.0 - phi) * feromonas[a, b] + phi * tau0
        feromonas[b, a] = feromonas[a, b]

    def construir_camino_hormiga():
        """
        Construye una solución con una hormiga.
        """
        inicio = int(rng.integers(0, n))

        camino = [inicio]
        no_visitados = set(range(n))
        no_visitados.remove(inicio)

        actual = inicio
        distancia_total = 0.0
        valido = True

        while no_visitados:
            if variante == "ACS":
                siguiente = seleccionar_siguiente_acs(actual, no_visitados)
            else:
                siguiente = seleccionar_siguiente(actual, no_visitados)

            if siguiente is None:
                valido = False
                break

            distancia_total += dist_matrix[actual, siguiente]
            camino.append(siguiente)
            no_visitados.remove(siguiente)

            if variante == "ACS":
                actualizar_local_acs(actual, siguiente)

            actual = siguiente

        if valido and ciclo_cerrado:
            inicio = camino[0]

            if np.isfinite(dist_matrix[actual, inicio]):
                distancia_total += dist_matrix[actual, inicio]
                camino.append(inicio)

                if variante == "ACS":
                    actualizar_local_acs(actual, inicio)
            else:
                valido = False

        if not valido:
            return camino, np.inf

        return camino, distancia_total

    def aplicar_2opt_limitado(camino, distancia_actual):
        """
        2-opt limitado y compatible con TSP abierto/cerrado.
        Solo acepta cambios si la nueva ruta sigue siendo válida y mejora.
        """
        if not np.isfinite(distancia_actual):
            return camino, distancia_actual

        mejor_camino_local = camino.copy()
        mejor_distancia_local = distancia_actual

        longitud = len(camino)

        if ciclo_cerrado and camino[0] == camino[-1]:
            base = camino[:-1]
        else:
            base = camino.copy()

        if len(base) < 4:
            return camino, distancia_actual

        for _ in range(n_intentos_2opt):
            i, j = sorted(rng.choice(len(base), size=2, replace=False))

            if j - i < 2:
                continue

            candidato = base.copy()
            candidato[i:j + 1] = reversed(candidato[i:j + 1])

            if ciclo_cerrado:
                candidato = candidato + [candidato[0]]

            d = evaluar(candidato)

            if d < mejor_distancia_local:
                mejor_camino_local = candidato
                mejor_distancia_local = d

                if ciclo_cerrado:
                    base = candidato[:-1]
                else:
                    base = candidato.copy()

        return mejor_camino_local, mejor_distancia_local

    def depositar_camino(camino, cantidad):
        """
        Deposita feromona en todas las aristas del camino.
        """
        for a, b in zip(camino[:-1], camino[1:]):
            if np.isfinite(dist_matrix[a, b]):
                feromonas[a, b] += cantidad
                feromonas[b, a] += cantidad

    def actualizar_feromonas_as(caminos_iteracion):
        """
        Ant System: todas las hormigas válidas depositan feromona.
        """
        feromonas[mascara_aristas] *= (1.0 - evaporacion)

        for camino, distancia in caminos_iteracion:
            if np.isfinite(distancia) and distancia > 0:
                depositar_camino(camino, q / distancia)

    def actualizar_feromonas_eas(caminos_iteracion, mejor_camino, mejor_distancia):
        """
        Elitist Ant System: AS + refuerzo extra sobre la mejor global.
        """
        actualizar_feromonas_as(caminos_iteracion)

        if mejor_camino is not None and np.isfinite(mejor_distancia):
            depositar_camino(mejor_camino, peso_elite * q / mejor_distancia)

    def actualizar_feromonas_asrank(caminos_iteracion, mejor_camino, mejor_distancia):
        """
        ASrank: solo las mejores hormigas de la iteración depositan feromona
        con peso según ranking. Además, la mejor global deposita con peso w.
        """
        feromonas[mascara_aristas] *= (1.0 - evaporacion)

        validos = [
            (camino, distancia)
            for camino, distancia in caminos_iteracion
            if np.isfinite(distancia) and distancia > 0
        ]

        validos.sort(key=lambda x: x[1])

        limite_rank = min(w_rank - 1, len(validos))

        for r in range(limite_rank):
            camino, distancia = validos[r]
            peso = w_rank - (r + 1)
            depositar_camino(camino, peso * q / distancia)

        if mejor_camino is not None and np.isfinite(mejor_distancia):
            depositar_camino(mejor_camino, w_rank * q / mejor_distancia)

    def actualizar_feromonas_mmas(caminos_iteracion, mejor_camino, mejor_distancia):
        """
        MMAS: solo actualiza la mejor hormiga de la iteración o la mejor global,
        aplicando límites tau_min y tau_max.
        """
        nonlocal tau_min, tau_max

        feromonas[mascara_aristas] *= (1.0 - evaporacion)

        validos = [
            (camino, distancia)
            for camino, distancia in caminos_iteracion
            if np.isfinite(distancia) and distancia > 0
        ]

        if len(validos) == 0:
            feromonas[mascara_aristas] = np.clip(
                feromonas[mascara_aristas],
                tau_min if tau_min is not None else 0.0,
                tau_max if tau_max is not None else np.inf
            )
            return

        mejor_iter_camino, mejor_iter_distancia = min(
            validos,
            key=lambda x: x[1]
        )

        if mmas_usar_mejor_global and mejor_camino is not None and np.isfinite(mejor_distancia):
            camino_deposito = mejor_camino
            distancia_deposito = mejor_distancia
        else:
            camino_deposito = mejor_iter_camino
            distancia_deposito = mejor_iter_distancia

        if tau_min is None or tau_max is None:
            tau_min_calc, tau_max_calc = calcular_limites_mmas(distancia_deposito)

            if tau_min is None:
                tau_min = tau_min_calc

            if tau_max is None:
                tau_max = tau_max_calc

        depositar_camino(camino_deposito, q / distancia_deposito)

        feromonas[mascara_aristas] = np.clip(
            feromonas[mascara_aristas],
            tau_min,
            tau_max
        )

    def actualizar_feromonas_acs(caminos_iteracion, mejor_camino, mejor_distancia):
        """
        ACS: actualización global solo en las aristas de la mejor solución.
        Las aristas que no pertenecen a la mejor ruta no se modifican aquí.
        """
        validos = [
            (camino, distancia)
            for camino, distancia in caminos_iteracion
            if np.isfinite(distancia) and distancia > 0
        ]

        if acs_usar_mejor_global and mejor_camino is not None and np.isfinite(mejor_distancia):
            camino_deposito = mejor_camino
            distancia_deposito = mejor_distancia
        else:
            if len(validos) == 0:
                return

            camino_deposito, distancia_deposito = min(
                validos,
                key=lambda x: x[1]
            )

        deposito = 1.0 / distancia_deposito

        for a, b in zip(camino_deposito[:-1], camino_deposito[1:]):
            if np.isfinite(dist_matrix[a, b]):
                feromonas[a, b] = (
                    (1.0 - evaporacion) * feromonas[a, b]
                    + evaporacion * deposito
                )
                feromonas[b, a] = feromonas[a, b]

    # ------------------------------------------------------------
    # Inicialización de mejor solución
    # ------------------------------------------------------------
    mejor_camino = None
    mejor_distancia = np.inf

    # Inicializar límites MMAS si procede
    if variante == "MMAS":
        tau_min_ini, tau_max_ini = calcular_limites_mmas(c_nn)

        if tau_min is None:
            tau_min = tau_min_ini

        if tau_max is None:
            tau_max = tau_max_ini

        feromonas[mascara_aristas] = tau_max

    # ------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------
    historial = []
    tiempo_inicio_total = time.perf_counter()

    for it in range(n_iteraciones):
        tiempo_inicio_iteracion = time.perf_counter()

        caminos_iteracion = []

        for _ in range(n_hormigas):
            camino, distancia = construir_camino_hormiga()

            if aplicar_2opt and np.isfinite(distancia):
                camino, distancia = aplicar_2opt_limitado(camino, distancia)

            caminos_iteracion.append((camino, distancia))

            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_camino = camino.copy()

        if variante == "AS":
            actualizar_feromonas_as(caminos_iteracion)

        elif variante == "EAS":
            actualizar_feromonas_eas(
                caminos_iteracion,
                mejor_camino,
                mejor_distancia
            )

        elif variante == "ASRANK":
            actualizar_feromonas_asrank(
                caminos_iteracion,
                mejor_camino,
                mejor_distancia
            )

        elif variante == "MMAS":
            actualizar_feromonas_mmas(
                caminos_iteracion,
                mejor_camino,
                mejor_distancia
            )

        elif variante == "ACS":
            actualizar_feromonas_acs(
                caminos_iteracion,
                mejor_camino,
                mejor_distancia
            )
        
        distancias_validas = [
            d for _, d in caminos_iteracion
            if np.isfinite(d)
        ]

        validas = len(distancias_validas)

        if validas > 0:
            mejor_distancia_iteracion = float(np.min(distancias_validas))
            media_validas = float(np.mean(distancias_validas))
            peor_valida = float(np.max(distancias_validas))
        else:
            mejor_distancia_iteracion = np.inf
            media_validas = np.inf
            peor_valida = np.inf

        tiempo_iteracion = time.perf_counter() - tiempo_inicio_iteracion
        tiempo_total = time.perf_counter() - tiempo_inicio_total

        tau_valores = feromonas[mascara_aristas]

        if len(tau_valores) > 0:
            tau_media_actual = float(np.mean(tau_valores))
            tau_min_actual = float(np.min(tau_valores))
            tau_max_actual = float(np.max(tau_valores))
        else:
            tau_media_actual = np.nan
            tau_min_actual = np.nan
            tau_max_actual = np.nan

        historial.append({
            "algoritmo": variante,
            "iteracion": it + 1,
            "mejor_distancia_global": float(mejor_distancia),
            "mejor_distancia_iteracion": float(mejor_distancia_iteracion),
            "media_distancias_validas": float(media_validas),
            "peor_distancia_valida": float(peor_valida),
            "rutas_validas": int(validas),
            "n_hormigas": int(n_hormigas),
            "tiempo_iteracion": float(tiempo_iteracion),
            "tiempo_total": float(tiempo_total),
            "tau_media": float(tau_media_actual),
            "tau_min": float(tau_min_actual),
            "tau_max": float(tau_max_actual)
        })

        if verbose and (it + 1) % 10 == 0:
            validas = sum(
                1 for _, d in caminos_iteracion
                if np.isfinite(d)
            )

            print(
                f"Iteración {it + 1}/{n_iteraciones} | "
                f"Variante: {variante} | "
                f"Mejor distancia: {mejor_distancia:.4f} | "
                f"Mejor iteración: {mejor_distancia_iteracion:.4f} | "
                f"Rutas válidas: {validas}/{n_hormigas} | "
                f"Tiempo iteración: {tiempo_iteracion:.4f}s"
            )

    if mejor_camino is None or not np.isfinite(mejor_distancia):
        raise ValueError(
            "No se ha encontrado ninguna ruta válida. "
            "Prueba a aumentar k_vecinos, n_hormigas, n_iteraciones "
            "o a generar más viewpoints."
        )

    return mejor_camino, mejor_distancia, historial

def aplicar_ant_system(
    dist_matrix,
    n_hormigas=None,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.5,
    q=1.0,
    tau0=None,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,
    aplicar_2opt=False
):
    return aplicar_aco_variante(
        dist_matrix=dist_matrix,
        variante="AS",
        n_hormigas=n_hormigas,
        n_iteraciones=n_iteraciones,
        alpha=alpha,
        beta=beta,
        evaporacion=evaporacion,
        q=q,
        tau0=tau0,
        ciclo_cerrado=ciclo_cerrado,
        seed=seed,
        verbose=verbose,
        aplicar_2opt=aplicar_2opt
    )


def aplicar_elitist_ant_system(
    dist_matrix,
    n_hormigas=None,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.5,
    q=1.0,
    tau0=None,
    peso_elite=None,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,
    aplicar_2opt=False
):
    return aplicar_aco_variante(
        dist_matrix=dist_matrix,
        variante="EAS",
        n_hormigas=n_hormigas,
        n_iteraciones=n_iteraciones,
        alpha=alpha,
        beta=beta,
        evaporacion=evaporacion,
        q=q,
        tau0=tau0,
        peso_elite=peso_elite,
        ciclo_cerrado=ciclo_cerrado,
        seed=seed,
        verbose=verbose,
        aplicar_2opt=aplicar_2opt
    )


def aplicar_as_ranked(
    dist_matrix,
    n_hormigas=None,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.1,
    q=1.0,
    tau0=None,
    w_rank=6,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,
    aplicar_2opt=False
):
    return aplicar_aco_variante(
        dist_matrix=dist_matrix,
        variante="ASRANK",
        n_hormigas=n_hormigas,
        n_iteraciones=n_iteraciones,
        alpha=alpha,
        beta=beta,
        evaporacion=evaporacion,
        q=q,
        tau0=tau0,
        w_rank=w_rank,
        ciclo_cerrado=ciclo_cerrado,
        seed=seed,
        verbose=verbose,
        aplicar_2opt=aplicar_2opt
    )


def aplicar_min_max_ant_system(
    dist_matrix,
    n_hormigas=None,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.02,
    q=1.0,
    tau0=None,
    tau_min=None,
    tau_max=None,
    mmas_usar_mejor_global=True,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,
    aplicar_2opt=False
):
    return aplicar_aco_variante(
        dist_matrix=dist_matrix,
        variante="MMAS",
        n_hormigas=n_hormigas,
        n_iteraciones=n_iteraciones,
        alpha=alpha,
        beta=beta,
        evaporacion=evaporacion,
        q=q,
        tau0=tau0,
        tau_min=tau_min,
        tau_max=tau_max,
        mmas_usar_mejor_global=mmas_usar_mejor_global,
        ciclo_cerrado=ciclo_cerrado,
        seed=seed,
        verbose=verbose,
        aplicar_2opt=aplicar_2opt
    )


def aplicar_ant_colony_system(
    dist_matrix,
    n_hormigas=10,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.1,
    q=1.0,
    tau0=None,
    phi=0.1,
    q0=0.9,
    acs_usar_mejor_global=True,
    ciclo_cerrado=False,
    seed=17,
    verbose=True,
    aplicar_2opt=False
):
    return aplicar_aco_variante(
        dist_matrix=dist_matrix,
        variante="ACS",
        n_hormigas=n_hormigas,
        n_iteraciones=n_iteraciones,
        alpha=alpha,
        beta=beta,
        evaporacion=evaporacion,
        q=q,
        tau0=tau0,
        phi=phi,
        q0=q0,
        acs_usar_mejor_global=acs_usar_mejor_global,
        ciclo_cerrado=ciclo_cerrado,
        seed=seed,
        verbose=verbose,
        aplicar_2opt=aplicar_2opt
    )