import os
from pathlib import Path
import pandas as pd
import numpy as np

from generar_vistas import generar_puntos_de_vista
from construir_grafo import construir_grafo_viewpoints_open3d
from ejecucion_individual import ejecutar_algoritmo, obtener_parametros_escenario

def main():
    """
    Para cada combinación (escenario, algoritmo), ejecuta 30 semillas distintas (del 1 al 30),
    y guarda el historial de cada ejecución (info de todas las iteraciones), un resumen de cada
    ejecución, y un resumen estadístico de las 30 ejecuciones de cada combinación. 
    """
    algoritmos = [
        "ACO_AS",
        "ACO_EAS",
        "ACO_AS_RANK",
        "ACO_MMAS",
        "ACO_ACS",
        "SA"
    ]

    escenarios = [1, 2]

    semillas = list(range(1, 31))
    ciclo_cerrado = False

    resultados = []
    historiales_aco = []
    historiales_sa = []

    output_experimentos = Path("resultados_experimentos")
    output_experimentos.mkdir(parents=True, exist_ok=True)

    for escenario in escenarios:
        print("\n" + "=" * 90)
        print(f"PREPARANDO ESCENARIO {escenario}")
        print("=" * 90)

        mesh_path = f"escenarios/escenarios_test/escenario{escenario}_ply/escenario{escenario}.ply"
        output_dir = f"escenarios/escenarios_test/escenario{escenario}_ply"
        os.makedirs(output_dir, exist_ok=True)

        parametros = obtener_parametros_escenario(escenario)

        print("\nFASE 1: GENERACIÓN DE VIEWPOINTS")

        viewpoints = generar_puntos_de_vista(
            mesh_path=mesh_path,
            d_f=parametros["d_f"],
            R_1=parametros["R_1"],
            d_min=0.1,
            R_2=parametros["R_2"],
            min_samples=1,
            ground_margin=0.08,
            export_intermediate=False,
            output_dir=output_dir,
            verbose=False
        )

        print(f"Viewpoints generados: {len(viewpoints)}")

        print("\nFASE 2: CONSTRUCCIÓN DEL GRAFO")

        dist_matrix = construir_grafo_viewpoints_open3d(
            mesh_path=mesh_path,
            viewpoints=viewpoints,
            k_vecinos=parametros["k_vecinos"],
            verbose=False
        )

        n_nodos = dist_matrix.shape[0]

        print(f"Grafo construido. Nodos: {n_nodos}")

        for algoritmo in algoritmos:
            for seed in semillas:
                print(
                    f"Ejecutando {algoritmo} | "
                    f"Escenario {escenario} | "
                    f"Seed {seed}"
                )

                try:
                    mejor_camino, mejor_distancia, historial = ejecutar_algoritmo(
                        algoritmo=algoritmo,
                        dist_matrix=dist_matrix,
                        seed=seed,
                        verbose=False,
                        ciclo_cerrado=ciclo_cerrado
                    )

                    df_historial = pd.DataFrame(historial)

                    tiempo_total = float(df_historial["tiempo_total"].max())
                    iteraciones = int(len(df_historial))

                    resultado = {
                        "escenario": escenario,
                        "algoritmo": algoritmo,
                        "seed": seed,
                        "n_viewpoints": int(n_nodos),
                        "mejor_distancia": float(mejor_distancia),
                        "tiempo_total": tiempo_total,
                        "iteraciones": iteraciones,
                        "estado": "OK"
                    }

                    resultados.append(resultado)

                    df_historial["escenario"] = escenario
                    df_historial["seed"] = seed

                    if algoritmo == "SA":
                        historiales_sa.append(df_historial)
                    else:
                        historiales_aco.append(df_historial)

                except Exception as e:
                    resultado = {
                        "escenario": escenario,
                        "algoritmo": algoritmo,
                        "seed": seed,
                        "n_viewpoints": int(n_nodos),
                        "mejor_distancia": np.inf,
                        "tiempo_total": np.nan,
                        "iteraciones": 0,
                        "estado": f"ERROR: {e}"
                    }

                    resultados.append(resultado)

                    print(f"[ERROR] {algoritmo} | Escenario {escenario} | Seed {seed}: {e}")

    df_resultados = pd.DataFrame(resultados)

    path_resultados = output_experimentos / "resultados_ejecuciones.csv"
    df_resultados.to_csv(path_resultados, index=False)

    print(f"\nResultados por ejecución guardados en: {path_resultados}")

    # Historiales separados porque ACO y SA registran métricas distintas
    if len(historiales_aco) > 0:
        df_historiales_aco = pd.concat(historiales_aco, ignore_index=True)

        path_historiales_aco = output_experimentos / "historiales_aco.csv"
        df_historiales_aco.to_csv(path_historiales_aco, index=False)

        print(f"\nHistoriales ACO guardados en: {path_historiales_aco}")

    if len(historiales_sa) > 0:
        df_historiales_sa = pd.concat(historiales_sa, ignore_index=True)

        path_historiales_sa = output_experimentos / "historiales_sa.csv"
        df_historiales_sa.to_csv(path_historiales_sa, index=False)

        print(f"\nHistoriales SA guardados en: {path_historiales_sa}")

    # ------------------------------------------------------------
    # Estadísticos descriptivos
    # ------------------------------------------------------------
    df_ok = df_resultados[df_resultados["estado"] == "OK"].copy()

    resumen = (
        df_ok
        .groupby(["escenario", "algoritmo"])
        .agg(
            ejecuciones=("mejor_distancia", "count"),
            media_distancia=("mejor_distancia", "mean"),
            std_distancia=("mejor_distancia", "std"),
            mediana_distancia=("mejor_distancia", "median"),
            mejor_distancia=("mejor_distancia", "min"),
            peor_distancia=("mejor_distancia", "max"),
            media_tiempo=("tiempo_total", "mean"),
            std_tiempo=("tiempo_total", "std"),
            mediana_tiempo=("tiempo_total", "median")
        )
        .reset_index()
    )

    resumen["cv_distancia"] = (
        resumen["std_distancia"] / resumen["media_distancia"]
    )

    path_resumen = output_experimentos / "resumen_estadistico.csv"
    resumen.to_csv(path_resumen, index=False)

    print(f"Resumen estadístico guardado en: {path_resumen}")

    print("\nResumen estadístico:")
    print(resumen)


if __name__ == "__main__":
    main()