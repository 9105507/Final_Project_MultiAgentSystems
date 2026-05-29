import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def cargar_escena_raycasting_open3d(mesh_path):
    """
    Carga la malla con Open3D y crea una escena de raycasting.
    """
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)

    if mesh_legacy.is_empty():
        raise ValueError(f"No se pudo cargar la malla: {mesh_path}")

    mesh_legacy.compute_vertex_normals()

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_t)

    return scene, mesh_id, mesh_legacy


def segmento_atraviesa_malla_open3d(scene, p1, p2, eps=1e-6):
    """
    Devuelve True si el segmento entre p1 y p2 intersecta la malla.

    Usa Open3D RaycastingScene.
    """
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < eps:
        return False

    direction = direction / length

    # Evitamos que el rayo detecte una intersección justo en el origen
    origin = p1 + direction * eps
    max_distance = length - 2 * eps

    if max_distance <= 0:
        return False

    # Open3D espera rayos con formato [ox, oy, oz, dx, dy, dz]
    ray = np.hstack([origin, direction]).astype(np.float32)
    rays = o3d.core.Tensor([ray], dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)

    # t_hit es la distancia desde el origen del rayo hasta la primera intersección
    t_hit = ans["t_hit"].numpy()[0]

    if np.isinf(t_hit):
        return False

    return eps < t_hit < max_distance


def construir_grafo_viewpoints_open3d(
    mesh_path,
    viewpoints,
    k_vecinos=10,
    verbose=True
):
    """
    Construye una matriz de distancias entre viewpoints.
    Solo añade aristas si el segmento no atraviesa la malla.
    """
    scene, mesh_id, mesh_legacy = cargar_escena_raycasting_open3d(mesh_path)

    viewpoints = np.asarray(viewpoints, dtype=np.float32)
    n = len(viewpoints)

    dist_matrix = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)

    if k_vecinos is None:
        pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        tree = cKDTree(viewpoints)
        _, indices = tree.query(viewpoints, k=min(k_vecinos + 1, n))

        pares_set = set()

        for i in range(n):
            for j in indices[i]:
                if i != j:
                    a, b = min(i, j), max(i, j)
                    pares_set.add((a, b))

        pares = list(pares_set)

    total = len(pares)
    validas = 0
    bloqueadas = 0

    for idx, (i, j) in enumerate(pares):
        p1 = viewpoints[i]
        p2 = viewpoints[j]

        atraviesa = segmento_atraviesa_malla_open3d(scene, p1, p2)

        if not atraviesa:
            d = np.linalg.norm(p2 - p1)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            validas += 1
        else:
            bloqueadas += 1

        if verbose and idx % 50 == 0:
            print(f"Procesadas {idx}/{total} aristas...")

    if verbose:
        print("Grafo construido con Open3D")
        print("Nodos:", n)
        print("Aristas válidas:", validas)
        print("Aristas bloqueadas por malla:", bloqueadas)

    return dist_matrix


def comprobar_conectividad(dist_matrix):
    """
    Comprueba si el grafo es conexo mediante BFS.
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
    seed=33,
    verbose=True
):
    """
    Aplica Ant Colony Optimization sobre el grafo de viewpoints.
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
    Convierte un camino de índices en coordenadas 3D.
    """
    return np.asarray([viewpoints[i] for i in camino])