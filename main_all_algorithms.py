import os
from pathlib import Path
import pandas as pd
from generar_vistas import generar_puntos_de_vista
from construir_grafo import construir_grafo_viewpoints_open3d
from planificadores import (
    aplicar_ant_system,
    aplicar_elitist_ant_system,
    aplicar_as_ranked,
    aplicar_min_max_ant_system,
    aplicar_ant_colony_system,
    aplicar_simulated_annealing,
    obtener_ruta_coordenadas
)
from utilidades import exportar_ruta_meshlab, visualizar_ruta_3d_con_mesh

def main(algoritmo, escenario):
    mesh_path = f"escenarios/escenarios_test/escenario{escenario}_ply/escenario{escenario}.ply"
    output_dir = f"escenarios/escenarios_test/escenario{escenario}_ply"
    os.makedirs(output_dir, exist_ok=True)

    if escenario == 1:
        # ESCENARIO 1
        parametros_escenario = {
            "d_f": 0.15,
            "R_1": 0.02,
            "R_2": 0.05,
            "k_vecinos": 20
        }
    else:
        # ESCENARIO 2
        parametros_escenario = {
            "d_f": 0.15,
            "R_1": 0.02,
            "R_2": 0.07,
            "k_vecinos": 20
        }

    print("================== FASE 1: GENERACIÓN DE VISTAS ==================\n")

    viewpoints = generar_puntos_de_vista(
        mesh_path=mesh_path,
        d_f=parametros_escenario["d_f"],                       # distancia de desplazamiento desde cada triángulo
        R_1=parametros_escenario["R_1"],                       # radio primer DBSCAN
        d_min=0.1,                      # distancia mínima a la malla
        R_2=parametros_escenario["R_2"],                       # radio segundo DBSCAN
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
        k_vecinos=parametros_escenario["k_vecinos"],
        verbose=True
    )

    print(f"\n================== FASE 3: PATH PLANNING CON {algoritmo} ==================\n")

    ciclo_cerrado = False
    n_nodos = dist_matrix.shape[0]

    if algoritmo == "ACO_AS":
        mejor_camino, mejor_distancia, historial = aplicar_ant_system(
            dist_matrix=dist_matrix,
            n_hormigas=n_nodos,
            n_iteraciones=100,
            alpha=1.0,
            beta=4.0,
            evaporacion=0.5,
            q=1.0,
            tau0=None,
            ciclo_cerrado=ciclo_cerrado,
            seed=17,
            verbose=True,
            aplicar_2opt=False
        )
    elif algoritmo == "ACO_EAS":
        mejor_camino, mejor_distancia, historial = aplicar_elitist_ant_system(
            dist_matrix=dist_matrix,
            n_hormigas=n_nodos,
            n_iteraciones=100,
            alpha=1.0,
            beta=4.0,
            evaporacion=0.5,
            q=1.0,
            tau0=None,
            peso_elite=n_nodos,
            ciclo_cerrado=ciclo_cerrado,
            seed=17,
            verbose=True,
            aplicar_2opt=False
        )
    elif algoritmo == "ACO_AS_RANK":
        mejor_camino, mejor_distancia, historial = aplicar_as_ranked(
            dist_matrix=dist_matrix,
            n_hormigas=n_nodos,
            n_iteraciones=100,
            alpha=1.0,
            beta=4.0,
            evaporacion=0.1,
            q=1.0,
            tau0=None,
            w_rank=6,
            ciclo_cerrado=ciclo_cerrado,
            seed=17,
            verbose=True,
            aplicar_2opt=False
        )
    elif algoritmo == "ACO_MMAS":
        mejor_camino, mejor_distancia, historial = aplicar_min_max_ant_system(
            dist_matrix=dist_matrix,
            n_hormigas=n_nodos,
            n_iteraciones=100,
            alpha=1.0,
            beta=4.0,
            evaporacion=0.02,
            q=1.0,
            tau0=None,
            tau_min=None,
            tau_max=None,
            mmas_usar_mejor_global=True,
            ciclo_cerrado=ciclo_cerrado,
            seed=17,
            verbose=True,
            aplicar_2opt=False
        )
    elif algoritmo == "ACO_ACS":
        mejor_camino, mejor_distancia, historial = aplicar_ant_colony_system(
            dist_matrix=dist_matrix,
            n_hormigas=10,
            n_iteraciones=100,
            alpha=1.0,
            beta=4.0,
            evaporacion=0.1,
            q=1.0,
            tau0=None,
            phi=0.1,
            q0=0.9,
            acs_usar_mejor_global=True,
            ciclo_cerrado=ciclo_cerrado,
            seed=17,
            verbose=True,
            aplicar_2opt=False
        )
    elif algoritmo == "SA":
        mejor_camino, mejor_distancia, historial = aplicar_simulated_annealing(
            dist_matrix=dist_matrix,
            n_iteraciones=10000,
            temperatura_inicial=None,
            temperatura_final=1e-4,
            enfriamiento=0.995,
            ciclo_cerrado=ciclo_cerrado,
            seed=17,
            verbose=True,
            max_intentos_inicial=1000,
            max_intentos_vecino=50
        )

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

if __name__ == "__main__":

    algoritmos = [
        "ACO_AS",
        "ACO_EAS",
        "ACO_AS_RANK",
        "ACO_MMAS",
        "ACO_ACS",
        "SA"
    ]

    escenarios = [1, 2]

    for escenario in escenarios:
        for algoritmo in algoritmos:
            print("\n" + "=" * 80)
            print(f"Ejecutando algoritmo: {algoritmo} | Escenario: {escenario}")
            print("=" * 80)

            try:
                main(
                    algoritmo=algoritmo,
                    escenario=escenario
                )

            except Exception as e:
                print(
                    f"\n[ERROR] Falló la ejecución de {algoritmo} "
                    f"en escenario {escenario}: {e}"
                )