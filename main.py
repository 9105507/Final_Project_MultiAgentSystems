import os
from pathlib import Path
from generar_vistas import generar_puntos_de_vista
from construir_grafo import construir_grafo_viewpoints_open3d
from planificadores import aplicar_aco, obtener_ruta_coordenadas
from utilidades import exportar_ruta_meshlab, visualizar_ruta_3d_con_mesh

escenario = 2
mesh_path = f"escenarios/escenarios_test/escenario{escenario}_ply/escenario{escenario}.ply"
output_dir = f"escenarios/escenarios_test/escenario{escenario}_ply"
os.makedirs(output_dir, exist_ok=True)

algoritmo = "ACO"

print("================== FASE 1: GENERACIÓN DE VISTAS ==================\n")

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
    verbose=True
)

print("\n================== FASE 2: CREACIÓN DEL GRAFO ==================\n")

dist_matrix = construir_grafo_viewpoints_open3d(
    mesh_path=mesh_path,
    viewpoints=viewpoints,
    # k_vecinos=10, # escenario 1
    k_vecinos=15, # escenario 2
    verbose=True
)

print("\n================== FASE 3: PATH PLANNING ==================\n")

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
    seed=17,
    verbose=True
)

ruta_3d = obtener_ruta_coordenadas(viewpoints, mejor_camino)

mejor_camino_limpio = [int(i) for i in mejor_camino]

print("\nMejor camino de índices:")
print(mejor_camino_limpio)

print("\nMejor distancia total:")
print("└──", round(float(mejor_distancia), 4))

print("\nRuta 3D (Shape):")
print("└──", ruta_3d.shape)

path_ruta = exportar_ruta_meshlab(
    viewpoints=viewpoints,
    camino=mejor_camino,
    output_dir=output_dir,
    nombre_ruta=f"ruta_{algoritmo.lower()}_meshlab.ply",
    radio_ruta=0.001
)

path_imagen = Path(output_dir) / f"ruta_{algoritmo.lower()}_sobre_escenario.png"

# Guarda la imagen
visualizar_ruta_3d_con_mesh(
    mesh_path=mesh_path,
    ruta_3d=ruta_3d,
    titulo=f"Ruta {algoritmo} sobre escenario {escenario}",
    algoritmo = algoritmo,
    save_path=path_imagen,
    max_triangulos=30000,
    color_mesh="#d9d6cf",
    color_ruta="crimson",
    grosor_ruta=0.008,
    radio_puntos=0.015,
    radio_inicio_fin=0.02,
    mostrar_ventana=False,
    zoom=0.85
)

# Abre ventana para visualizar
visualizar_ruta_3d_con_mesh(
    mesh_path=mesh_path,
    ruta_3d=ruta_3d,
    titulo=f"Ruta {algoritmo} sobre escenario {escenario}",
    algoritmo = algoritmo,
    save_path=None,
    max_triangulos=30000,
    color_mesh="#c0b7a0",
    color_ruta="crimson",
    grosor_ruta=0.008,
    radio_puntos=0.015,
    radio_inicio_fin=0.02,
    mostrar_ventana=True,
    ventana=(950, 750),
    posicion_ventana=(268, 20),
    zoom=0.9,
    elev=-8
)