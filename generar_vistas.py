import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from pathlib import Path

def export_point_cloud(points, output_path, color=[1.0, 0.0, 0.0], verbose=False):
    """
    Exporta una nube de puntos a PLY para visualizar en MeshLab.
    """
    points = np.asarray(points)

    if points.shape[0] == 0:
        print(f"No se exporta {output_path}: no hay puntos.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)

    o3d.io.write_point_cloud(
        str(output_path),
        pcd,
        write_ascii=True
    )

    if verbose:
        print(f"Exportado: {output_path} | puntos: {points.shape[0]}")

def filter_viewpoints_outside_and_above_ground(points, mesh, vertices, ground_margin=0.08):
    """
    Filtra un conjunto de puntos:
    1. Elimina los puntos que están dentro de la malla.
    2. Elimina los puntos que están por debajo del suelo + margen.

    Parámetros
    ----------
    points : np.ndarray de shape (N, 3)
        Puntos a filtrar.
    mesh : open3d.geometry.TriangleMesh
        Malla original.
    vertices : np.ndarray de shape (M, 3)
        Vértices de la malla.
    ground_margin : float
        Margen por encima del suelo para aceptar puntos.

    Retorna
    -------
    filtered_points : np.ndarray de shape (K, 3)
        Puntos filtrados.
    """

    # ------------------------------------------------------------
    # 1. FILTRAR PUNTOS INTERIORES
    # ------------------------------------------------------------
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    points_tensor = o3d.core.Tensor(
        points.astype(np.float32),
        dtype=o3d.core.Dtype.Float32
    )

    signed_distances = scene.compute_signed_distance(points_tensor).numpy()

    # En tu caso:
    # signed_distance > 0  --> punto fuera
    # signed_distance < 0  --> punto dentro
    mask_outside = signed_distances > 0

    filtered_points = points[mask_outside]

    # ------------------------------------------------------------
    # 2. FILTRAR PUNTOS POR DEBAJO DEL SUELO
    # ------------------------------------------------------------
    z_floor = vertices[:, 2].min()

    mask_above_floor = filtered_points[:, 2] > z_floor + ground_margin

    filtered_points = filtered_points[mask_above_floor]

    return filtered_points

def generar_puntos_de_vista(
    mesh_path,
    d_f=0.15,
    R_1=0.02,
    d_min=0.1,
    R_2=0.06,
    min_samples=1,
    ground_margin=0.08,
    export_intermediate=False,
    output_dir=None,
    verbose=False
):
    """
    Genera puntos de vista a partir de una malla triangular.

    Parámetros
    ----------
    mesh_path : str
        Ruta de la malla .ply o .obj.

    d_f : float
        Distancia de desplazamiento desde el centro de cada triángulo.

    R_1 : float
        Radio del primer DBSCAN.

    d_min : float
        Distancia mínima a la malla para conservar centroides.

    R_2 : float
        Radio del segundo DBSCAN.

    min_samples : int
        Mínimo de puntos por clúster en DBSCAN.

    ground_margin : float
        Margen respecto al suelo para eliminar puntos bajos.

    export_intermediate : bool
        Si True, exporta nubes de puntos intermedias.

    output_dir : str
        Carpeta donde guardar resultados intermedios y finales.

    Retorna
    -------
    final_viewpoints : np.ndarray
        Puntos de vista finales, shape (N, 3).
    """

    # ============================================================
    # 1. LEER MALLA
    # ============================================================

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if mesh.is_empty():
        raise ValueError(f"No se ha podido leer la malla: {mesh_path}")

    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_degenerate_triangles()

    # mesh.orient_triangles()
    mesh.compute_triangle_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    if verbose:
        print("Vértices:", vertices.shape)
        print("Triángulos:", triangles.shape)

    # ============================================================
    # 2. CENTROS DE TRIÁNGULOS
    # ============================================================

    triangle_vertices = vertices[triangles]
    triangle_centers = triangle_vertices.mean(axis=1)

    # ============================================================
    # CORREGIR ORIENTACIÓN GLOBAL DE NORMALES
    # ============================================================

    # mesh_center = vertices.mean(axis=0)

    # directions_out = triangle_centers - mesh_center
    # dots = np.sum(triangle_normals * directions_out, axis=1)

    # triangle_normals[dots < 0] *= -1

    # print("Normales invertidas:", np.sum(dots < 0))

    # ============================================================
    # 3. GENERAR PUNTOS DESPLAZADOS D
    # ============================================================

    D = triangle_centers + d_f * triangle_normals

    if verbose:
        print("D inicial:", D.shape)

    D = filter_viewpoints_outside_and_above_ground(
        D,
        mesh,
        vertices,
        ground_margin=ground_margin
    )

    if verbose:
        print("D tras filtrar interior/suelo:", D.shape)

    if export_intermediate and output_dir is not None:
        export_point_cloud(
            D,
            Path(output_dir) / "01_D_puntos_desplazados_filtrados.ply",
            color=[1.0, 1.0, 0.0],
            verbose=verbose
        )

    # ============================================================
    # 4. PRIMER DBSCAN
    # ============================================================

    dbscan_1 = DBSCAN(
        eps=R_1,
        min_samples=min_samples,
        algorithm="kd_tree",
        n_jobs=-1
    )

    labels_1 = dbscan_1.fit_predict(D)
    unique_labels_1 = np.unique(labels_1)

    if verbose:
        print("Número de clústeres primera fase:", len(unique_labels_1))

    # ============================================================
    # 5. CENTROIDES PRIMERA FASE
    # ============================================================

    centroids_1 = []

    for label in unique_labels_1:
        cluster_points = D[labels_1 == label]
        centroid = cluster_points.mean(axis=0)
        centroids_1.append(centroid)

    centroids_1 = np.asarray(centroids_1)

    centroids_1 = filter_viewpoints_outside_and_above_ground(
        centroids_1,
        mesh,
        vertices,
        ground_margin=ground_margin
    )

    if verbose:
        print("Centroides primera fase:", centroids_1.shape)

    if export_intermediate and output_dir is not None:
        export_point_cloud(
            centroids_1,
            Path(output_dir) / "02_centroides_primera_fase.ply",
            color=[0.0, 1.0, 0.0],
            verbose=verbose
        )

    # ============================================================
    # 6. FILTRAR CENTROIDES DEMASIADO CERCANOS A LA MALLA
    # ============================================================

    tree = cKDTree(vertices)

    distances_to_mesh, _ = tree.query(centroids_1, k=1)

    mask = distances_to_mesh > d_min

    filtered_centroids = centroids_1[mask]

    filtered_centroids = filter_viewpoints_outside_and_above_ground(
        filtered_centroids,
        mesh,
        vertices,
        ground_margin=ground_margin
    )

    if verbose:
        print("Centroides antes del filtrado distancia:", centroids_1.shape[0])
        print("Centroides después del filtrado distancia:", filtered_centroids.shape[0])

    if filtered_centroids.shape[0] == 0:
        raise ValueError("Todos los centroides han sido descartados. Reduce d_min.")

    if export_intermediate and output_dir is not None:
        export_point_cloud(
            filtered_centroids,
            Path(output_dir) / "03_centroides_filtrados_distancia.ply",
            color=[1.0, 0.0, 1.0],
            verbose=verbose
        )

    # ============================================================
    # 7. SEGUNDO DBSCAN
    # ============================================================

    dbscan_2 = DBSCAN(
        eps=R_2,
        min_samples=min_samples,
        algorithm="kd_tree",
        n_jobs=-1
    )

    labels_2 = dbscan_2.fit_predict(filtered_centroids)
    unique_labels_2 = np.unique(labels_2)

    if verbose:
        print("Número de clústeres segunda fase:", len(unique_labels_2))

    # ============================================================
    # 8. CENTROIDES FINALES
    # ============================================================

    final_viewpoints = []

    for label in unique_labels_2:
        cluster_points = filtered_centroids[labels_2 == label]
        centroid = cluster_points.mean(axis=0)
        final_viewpoints.append(centroid)

    final_viewpoints = np.asarray(final_viewpoints)

    final_viewpoints = filter_viewpoints_outside_and_above_ground(
        final_viewpoints,
        mesh,
        vertices,
        ground_margin=ground_margin
    )

    if verbose:
        print("Puntos de vista finales:", final_viewpoints.shape)

    if export_intermediate and output_dir is not None:
        export_point_cloud(
            final_viewpoints,
            Path(output_dir) / "04_viewpoints_finales.ply",
            color=[1.0, 0.0, 0.0],
            verbose=verbose
        )

    return final_viewpoints