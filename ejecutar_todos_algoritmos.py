import os
from pathlib import Path
import pandas as pd
from generar_vistas import generar_puntos_de_vista
from construir_grafo import construir_grafo_viewpoints_open3d
from ejecucion_individual import ejecutar_algoritmo, obtener_parametros_escenario
from utilidades import exportar_ruta_meshlab, visualizar_ruta_3d_con_mesh
from planificadores import obtener_ruta_coordenadas

def execute_all_algorithms(escenario):
    """
    Ejecuta todos los algoritmos en un escenario proporcionado con semilla = 17.
    """
    mesh_path = f"escenarios/escenarios_test/escenario{escenario}_ply/escenario{escenario}.ply"
    output_dir = f"escenarios/escenarios_test/escenario{escenario}_ply"
    os.makedirs(output_dir, exist_ok=True)

    parametros_escenario = obtener_parametros_escenario(escenario)

    print("================== FASE 1: GENERACIÓN DE VISTAS ==================\n")

    viewpoints = generar_puntos_de_vista(
        mesh_path=mesh_path,
        d_f=parametros_escenario["d_f"],                       # distancia de desplazamiento desde cada triángulo
        R_1=parametros_escenario["R_1"],                       # radio primer DBSCAN
        d_min=0.1,                                             # distancia mínima a la malla
        R_2=parametros_escenario["R_2"],                       # radio segundo DBSCAN
        min_samples=1,                                         # Mínimo de puntos por clúster
        ground_margin=0.08,
        export_intermediate=True,
        output_dir=output_dir,
        verbose=True
    )

    print("\n================== FASE 2: CREACIÓN DEL GRAFO ==================\n")

    dist_matrix = construir_grafo_viewpoints_open3d(
        mesh_path=mesh_path,
        viewpoints=viewpoints,
        k_vecinos=parametros_escenario["k_vecinos"],
        verbose=True
    )

    print(f"\n================== FASE 3: PATH PLANNING ==================")

    ciclo_cerrado = False
    algoritmos = [
        "ACO_AS",
        "ACO_EAS",
        "ACO_AS_RANK",
        "ACO_MMAS",
        "ACO_ACS",
        "SA"
    ]

    for algoritmo in algoritmos:
        print(f"\n└──Algoritmo: {algoritmo}!\n")
        try:
            mejor_camino, mejor_distancia, historial = ejecutar_algoritmo(algoritmo, dist_matrix, seed=17, verbose=True, ciclo_cerrado=ciclo_cerrado)

            df_historial = pd.DataFrame(historial)

            path_historial = Path(output_dir) / f"historial_{algoritmo.lower()}.csv"
            df_historial.to_csv(path_historial, index=False)

            print(f"\n└──Historial guardado en: {path_historial}")

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

            if ciclo_cerrado:
                path_imagen = Path(output_dir) / f"ruta_{algoritmo.lower()}_TSP_Cerrado.png"
            else:
                path_imagen = Path(output_dir) / f"ruta_{algoritmo.lower()}_TSP_Abierto.png"

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
        except Exception as e:
            print(f"└──[ERROR] Falló {algoritmo} en escenario {escenario}: {e}")
            continue

if __name__ == "__main__":
    """
    Itera sobre los dos escenarios, ejecutando todos los algoritmos en cada escenario.
    """
    escenarios = [1, 2]

    for escenario in escenarios:
        print("\n" + "=" * 80)
        print(f"Ejecutando algoritmos en escenario: {escenario}")
        print("=" * 80)

        execute_all_algorithms(escenario=escenario)