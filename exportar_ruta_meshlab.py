import numpy as np
import open3d as o3d
import os


def crear_cilindro_entre_puntos(p1, p2, radio=0.006, color=(0.0, 0.0, 1.0)):
    """
    Crea un cilindro entre dos puntos 3D.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    vector = p2 - p1
    longitud = np.linalg.norm(vector)

    if longitud < 1e-12:
        return None

    direccion = vector / longitud

    cilindro = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radio,
        height=longitud,
        resolution=20
    )

    # El cilindro en Open3D nace alineado con Z
    eje_z = np.array([0.0, 0.0, 1.0])
    v = np.cross(eje_z, direccion)
    c = np.dot(eje_z, direccion)

    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            R = np.eye(3)
        else:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0]) * np.pi
            )
    else:
        vx = np.array([
            [0.0, -v[2],  v[1]],
            [v[2],  0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

    cilindro.rotate(R, center=(0.0, 0.0, 0.0))

    punto_medio = (p1 + p2) / 2.0
    cilindro.translate(punto_medio)
    cilindro.paint_uniform_color(color)

    return cilindro


def exportar_ruta_meshlab(
    viewpoints,
    camino,
    output_dir,
    nombre_ruta="ruta_aco_meshlab.ply",
    radio_ruta=0.006,
    color_ruta=(0.0, 0.0, 1.0)
):
    """
    Exporta solo la ruta como un PLY formado por cilindros.
    viewpoints: array Nx3
    camino: lista de índices, por ejemplo [4, 2, 7, 1, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    viewpoints = np.asarray(viewpoints, dtype=float)
    ruta = np.asarray([viewpoints[i] for i in camino], dtype=float)

    mesh_ruta = o3d.geometry.TriangleMesh()

    for p1, p2 in zip(ruta[:-1], ruta[1:]):
        cilindro = crear_cilindro_entre_puntos(
            p1, p2,
            radio=radio_ruta,
            color=color_ruta
        )
        if cilindro is not None:
            mesh_ruta += cilindro

    path_ruta = os.path.join(output_dir, nombre_ruta)
    o3d.io.write_triangle_mesh(path_ruta, mesh_ruta)

    return path_ruta