import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def cargar_escena_raycasting_open3d(mesh_path):
    """
    Carga una malla 3D y construye una escena de raycasting con Open3D.

    La función lee una malla triangular desde disco, comprueba que se haya
    cargado correctamente, calcula sus normales y la convierte al formato tensor
    de Open3D. Posteriormente, crea una escena de raycasting que permite lanzar
    rayos contra la geometría para comprobar intersecciones.

    Parámetros
    ----------
    mesh_path : str o pathlib.Path
        Ruta del archivo de la malla 3D a cargar. El archivo debe estar en un
        formato compatible con Open3D, como PLY.

    Retorna
    -------
    scene : open3d.t.geometry.RaycastingScene
        Escena de raycasting que contiene la malla cargada.

    mesh_id : int
        Identificador interno de la malla dentro de la escena de raycasting.

    mesh_legacy : open3d.geometry.TriangleMesh
        Malla cargada en formato clásico de Open3D. Se devuelve por si es
        necesario reutilizarla posteriormente para visualización, análisis o
        procesamiento adicional.

    Notas
    -----
    Open3D utiliza dos representaciones distintas de geometría: la versión
    clásica, open3d.geometry, y la versión tensorial, open3d.t.geometry.
    RaycastingScene trabaja con la segunda, por eso es necesario convertir la
    malla antes de añadirla a la escena.
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
    Comprueba si el segmento definido por dos puntos intersecta una malla.

    La función lanza un rayo desde p1 en dirección a p2 utilizando la escena de
    raycasting de Open3D. Si el primer impacto del rayo se encuentra dentro de
    la longitud real del segmento, se considera que el segmento atraviesa la
    malla.

    Para evitar falsos positivos en el propio punto de origen o en el punto
    final del segmento, se aplica un pequeño margen numérico mediante eps.

    Parámetros
    ----------
    scene : open3d.t.geometry.RaycastingScene
        Escena de raycasting que contiene la malla contra la que se comprobará
        la intersección.

    p1 : array-like de shape (3,)
        Punto inicial del segmento, con coordenadas (x, y, z).

    p2 : array-like de shape (3,)
        Punto final del segmento, con coordenadas (x, y, z).

    eps : float, opcional
        Margen numérico utilizado para evitar detectar intersecciones justo en
        los extremos del segmento. Por defecto es 1e-6.

    Retorna
    -------
    atraviesa : bool
        True si el segmento entre p1 y p2 intersecta la malla.
        False si no existe intersección dentro del tramo considerado.

    Notas
    -----
    Open3D representa cada rayo con seis valores:
    [origen_x, origen_y, origen_z, direccion_x, direccion_y, direccion_z].

    El valor t_hit devuelto por cast_rays indica la distancia desde el origen
    del rayo hasta la primera intersección encontrada. Si t_hit es infinito,
    significa que el rayo no ha impactado contra ninguna geometría.
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

def construir_grafo_viewpoints_open3d(mesh_path, viewpoints, k_vecinos=10, verbose=True):
    """
    Construye una matriz de distancias entre viewpoints evitando obstáculos.

    La función genera un grafo no dirigido a partir de un conjunto de puntos de
    vista. Cada viewpoint se interpreta como un nodo del grafo. Entre dos nodos
    se añade una arista únicamente si el segmento recto que los une no atraviesa
    la malla 3D cargada.

    La comprobación de visibilidad entre puntos se realiza mediante raycasting
    con Open3D. Las distancias euclídeas entre puntos conectados se almacenan en
    una matriz de distancias. Cuando dos puntos no son conectables, la distancia
    se mantiene como infinito.

    Parámetros
    ----------
    mesh_path : str o pathlib.Path
        Ruta de la malla 3D utilizada como obstáculo o geometría de referencia.

    viewpoints : np.ndarray de shape (N, 3)
        Conjunto de puntos de vista que actuarán como nodos del grafo. Cada fila
        representa un punto con coordenadas (x, y, z).

    k_vecinos : int o None, opcional
        Número de vecinos más cercanos que se evaluarán para cada viewpoint.
        Si se indica un entero, se construye un grafo local basado en los
        k vecinos más próximos de cada punto.
        Si es None, se evalúan todos los pares posibles de viewpoints.
        Por defecto es 10.

    verbose : bool, opcional
        Si es True, muestra por consola el progreso de construcción del grafo y
        un resumen final con el número de nodos, aristas válidas y aristas
        bloqueadas por la malla. Por defecto es True.

    Retorna
    -------
    dist_matrix : np.ndarray de shape (N, N)
        Matriz de distancias del grafo. La posición dist_matrix[i, j] contiene
        la distancia euclídea entre los viewpoints i y j si existe conexión
        directa sin atravesar la malla. En caso contrario, contiene np.inf.
        La diagonal principal contiene ceros.

    Notas
    -----
    El uso de k_vecinos permite reducir el coste computacional, ya que evita
    comprobar todos los pares posibles de puntos. Esto resulta especialmente
    útil cuando el número de viewpoints es elevado.

    Si k_vecinos es None, el número de comprobaciones crece de forma cuadrática
    con el número de viewpoints, ya que se analizan todas las combinaciones de
    pares posibles.
    """
    scene, _, _ = cargar_escena_raycasting_open3d(mesh_path)

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