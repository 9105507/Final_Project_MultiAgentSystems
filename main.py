from generar_vistas import generar_puntos_de_vista

from planificador_aco import (
    construir_grafo_viewpoints_open3d,
    aplicar_aco,
    obtener_ruta_coordenadas
)

from exportar_ruta_meshlab import exportar_ruta_meshlab

mesh_path = "escenarios/escenarios_test/escenario2_ply/escenario2.ply"
output_dir = "escenarios/escenarios_test/escenario2_ply"

viewpoints = generar_puntos_de_vista(
    mesh_path=mesh_path,
    d_f=0.15,                       # distancia de desplazamiento desde cada triángulo
    R_1=0.02,                       # radio primer DBSCAN
    d_min=0.1,                      # distancia mínima a la malla
    R_2=0.06,                       # radio segundo DBSCAN
    min_samples=1,                  # Mínimo de puntos por clúster
    ground_margin=0.08,
    export_intermediate=True,
    output_dir=output_dir,
    verbose=False
)

print("Viewpoints obtenidos en main:", viewpoints.shape)
# print(viewpoints)

dist_matrix = construir_grafo_viewpoints_open3d(
    mesh_path=mesh_path,
    viewpoints=viewpoints,
    k_vecinos=15, # con escenario1 k = 10 vale
    verbose=True
)

# print(dist_matrix)
mejor_camino, mejor_distancia = aplicar_aco(
    dist_matrix=dist_matrix,
    n_hormigas=50,
    n_iteraciones=200,
    alpha=1.0,
    beta=4.0,
    evaporacion=0.4,
    q=1.0,
    ciclo_cerrado=False,
    seed=33,
    verbose=True
)

ruta_3d = obtener_ruta_coordenadas(viewpoints, mejor_camino)

print("Mejor camino de índices:")
print(mejor_camino)

print("Mejor distancia total:")
print(mejor_distancia)

print("Ruta 3D:")
print(ruta_3d.shape)

camino_cerrado = mejor_camino + [mejor_camino[0]]

path_ruta = exportar_ruta_meshlab(
    viewpoints=viewpoints,
    camino=camino_cerrado,
    output_dir=output_dir,
    nombre_ruta="ruta_aco_meshlab.ply",
    radio_ruta=0.006
)