# Planificación de rutas de inspección 3D mediante algoritmos multiagente y metaheurísticas

## 1. Descripción del proyecto

Este proyecto implementa un sistema de planificación de rutas sobre escenarios tridimensionales con el objetivo de generar trayectorias de inspección a partir de un conjunto de puntos de vista o *viewpoints*. El flujo de trabajo parte de una malla 3D, genera automáticamente puntos de observación alrededor del escenario, construye un grafo de visibilidad entre dichos puntos y aplica distintos algoritmos de optimización para obtener una ruta de recorrido eficiente.

El problema se modela como una variante abierta del problema del viajante (*Travelling Salesman Problem*, TSP), donde cada nodo representa un viewpoint, las aristas representan conexiones válidas entre puntos siempre que el segmento que los une no atraviese la geometría del escenario, el peso de cada arista es la distancia euclidiana entre nodos. TSP Abierto hace referencia a que la ruta no tiene que volver obligatoriamente al nodo inicial.

El proyecto compara distintas variantes de Ant Colony Optimization (ACO) y utiliza Simulated Annealing (SA) como algoritmo de referencia externo.

## 2. Objetivo

El objetivo principal es comparar el comportamiento de varias técnicas de optimización aplicadas a la planificación de rutas en grafos 3D generados a partir de mallas.

Los objetivos específicos son:

* Generar viewpoints automáticamente a partir de una malla triangular.
* Filtrar puntos interiores, puntos demasiado próximos a la malla y puntos cercanos al suelo.
* Construir un grafo de visibilidad entre viewpoints mediante raycasting.
* Resolver una variante abierta del TSP sobre dicho grafo.
* Comparar distintas variantes de ACO.
* Comparar los resultados de ACO con Simulated Annealing.
* Exportar rutas, grafos e historiales de ejecución para su análisis posterior.
* Visualizar las rutas generadas sobre la malla 3D.

## 3. Algoritmos implementados

El proyecto implementa las siguientes variantes de Ant Colony Optimization:

* Ant System (AS)
* Elitist Ant System (EAS)
* Rank-Based Ant System (ASrank)
* MAX-MIN Ant System (MMAS)
* Ant Colony System (ACS)

Además, se incluye:

* Simulated Annealing (SA)

Las variantes ACO se utilizan como familia principal de algoritmos multiagente, mientras que Simulated Annealing se emplea como metaheurística comparativa basada en búsqueda local estocástica.

## 4. Estructura del proyecto

La estructura principal del proyecto es la siguiente:

```text
.
├── ejecucion_individual.py
├── ejecutar_todos_algoritmos.py
├── experimentos_estadisticos.py
├── generar_vistas.py
├── construir_grafo.py
├── planificadores.py
├── utilidades.py
├── escenarios/
│   └── escenarios_test/
│       ├── escenario1_ply/
│       │   └── escenario1.ply
│       └── escenario2_ply/
│           └── escenario2.ply
└── resultados_experimentos/
```

## 5. Descripción de los archivos principales

### `generar_vistas.py`

Contiene las funciones necesarias para generar los viewpoints a partir de una malla 3D.

El proceso incluye:

1. Lectura de la malla.
2. Cálculo de centros y normales de triángulos.
3. Generación de puntos desplazados desde la superficie.
4. Filtrado de puntos interiores y puntos cercanos al suelo.
5. Agrupamiento mediante DBSCAN.
6. Filtrado por distancia mínima a la malla.
7. Segundo agrupamiento mediante DBSCAN.
8. Obtención de los viewpoints finales.

También permite exportar nubes de puntos intermedias en formato `.ply`.

### `construir_grafo.py`

Construye el grafo de visibilidad entre viewpoints.

Cada viewpoint se representa como un nodo. Para cada par de puntos cercanos, se comprueba mediante raycasting si el segmento que los une atraviesa la malla. Si no la atraviesa, se añade una arista con peso igual a la distancia euclídea entre ambos puntos.

El resultado es una matriz de distancias `dist_matrix`, donde:

* Los valores finitos representan conexiones válidas.
* Los valores `np.inf` representan conexiones no válidas.
* La diagonal contiene ceros.

### `planificadores.py`

Contiene la implementación de los algoritmos de planificación y optimización.

Incluye:

* Cálculo de distancia de una ruta.
* Comprobación de conectividad del grafo.
* Simulated Annealing.
* Variantes de Ant Colony Optimization:

  * AS
  * EAS
  * ASrank
  * MMAS
  * ACS

Cada algoritmo devuelve:

```python
mejor_camino, mejor_distancia, historial
```

donde:

* `mejor_camino` es la secuencia de nodos visitados.
* `mejor_distancia` es la distancia total de la mejor ruta encontrada.
* `historial` contiene la evolución iteración a iteración.

### `utilidades.py`

Incluye funciones auxiliares para exportación y visualización.

Permite:

* Convertir una ruta en cilindros 3D.
* Exportar rutas en formato `.ply` para MeshLab.
* Exportar el grafo de viewpoints.
* Visualizar rutas sobre la malla usando PyVista.
* Guardar capturas de las rutas generadas.

### `ejecucion_individual.py`

Ejecuta un único algoritmo sobre un único escenario.

Este archivo permite probar rápidamente una combinación concreta de escenario y algoritmo. Genera los viewpoints, construye el grafo, ejecuta el algoritmo seleccionado, exporta la ruta, guarda el historial y genera una imagen de visualización.

Los algoritmos disponibles son:

```python
"ACO_AS"
"ACO_EAS"
"ACO_AS_RANK"
"ACO_MMAS"
"ACO_ACS"
"SA"
```

### `ejecutar_todos_algoritmos.py`

Ejecuta todos los algoritmos sobre los escenarios definidos.

Este script está pensado para obtener una primera comparativa directa entre algoritmos utilizando una semilla fija. Para cada escenario, genera los viewpoints y el grafo una sola vez, y posteriormente ejecuta todos los algoritmos sobre el mismo grafo.

### `experimentos_estadisticos.py`

Ejecuta experimentos estadísticos con múltiples semillas.

Para cada combinación de escenario y algoritmo, se realizan varias ejecuciones independientes con semillas distintas. El objetivo es analizar no solo el mejor resultado obtenido, sino también la estabilidad de cada algoritmo.

Este script genera:

* Resultados por ejecución.
* Historiales de ACO.
* Historiales de SA.
* Resumen estadístico agregado.

## 6. Escenarios utilizados

El proyecto utiliza dos escenarios 3D almacenados en formato `.ply`.

Los escenarios fueron normalizados previamente para quedar contenidos dentro de una caja unitaria de dimensiones:

```text
1 × 1 × 1
```

Por tanto, las coordenadas y distancias utilizadas por los algoritmos no representan metros reales, sino unidades normalizadas del sistema de coordenadas de la malla.

En consecuencia, las distancias obtenidas deben interpretarse como distancias relativas dentro del espacio normalizado.

Ejemplo:

```text
mejor_distancia = 10.7047
```

significa:

```text
10.7047 unidades normalizadas acumuladas
```

No significa 10.7047 metros.

Para convertir estas distancias a unidades físicas sería necesario conocer el factor de escala aplicado durante la normalización del modelo original (Trabajo Futuro).

## 7. Parámetros de los escenarios

Los parámetros utilizados para cada escenario se definen en `obtener_parametros_escenario`.

### Escenario 1

```python
{
    "d_f": 0.15,
    "R_1": 0.02,
    "R_2": 0.05,
    "k_vecinos": 20
}
```

### Escenario 2

```python
{
    "d_f": 0.15,
    "R_1": 0.02,
    "R_2": 0.07,
    "k_vecinos": 20
}
```

Donde:

* `d_f`: distancia de desplazamiento desde la superficie de la malla.
* `R_1`: radio del primer agrupamiento DBSCAN.
* `R_2`: radio del segundo agrupamiento DBSCAN.
* `k_vecinos`: número de vecinos próximos evaluados en la construcción del grafo.

Todos estos parámetros están expresados en unidades normalizadas.

## 8. Ejecución individual

Para ejecutar un único algoritmo sobre un escenario, modificar en `ejecucion_individual.py` las variables:

```python
algoritmo = "SA"
escenario = 1
```

Ejemplo de algoritmos válidos:

```python
algoritmo = "ACO_AS"
algoritmo = "ACO_EAS"
algoritmo = "ACO_AS_RANK"
algoritmo = "ACO_MMAS"
algoritmo = "ACO_ACS"
algoritmo = "SA"
```

Después ejecutar:

```bash
python ejecucion_individual.py
```

El script genera:

* Viewpoints finales.
* Grafo de visibilidad.
* Historial del algoritmo.
* Ruta exportada en `.ply`.
* Imagen de la ruta sobre la malla.
* Visualización interactiva de la ruta.

## 9. Ejecución de todos los algoritmos

Para ejecutar todos los algoritmos sobre los escenarios definidos:

```bash
python ejecutar_todos_algoritmos.py
```

Este script ejecuta:

* ACO_AS
* ACO_EAS
* ACO_AS_RANK
* ACO_MMAS
* ACO_ACS
* SA

sobre los escenarios 1 y 2.

## 10. Experimentos estadísticos

Para ejecutar la comparativa estadística con múltiples semillas:

```bash
python experimentos_estadisticos.py
```

El script ejecuta cada combinación de escenario y algoritmo con distintas semillas, permitiendo analizar la estabilidad de los métodos.

Los resultados se guardan en la carpeta:

```text
resultados_experimentos/
```

Los archivos generados son:

```text
resultados_ejecuciones.csv
historiales_aco.csv
historiales_sa.csv
resumen_estadistico.csv
```

### `resultados_ejecuciones.csv`

Contiene una fila por ejecución.

Columnas principales:

* `escenario`
* `algoritmo`
* `seed`
* `n_viewpoints`
* `mejor_distancia`
* `tiempo_total`
* `iteraciones`
* `estado`

### `historiales_aco.csv`

Contiene la evolución iteración a iteración de las variantes ACO.

Incluye métricas como:

* Mejor distancia global.
* Mejor distancia de la iteración.
* Media de distancias válidas.
* Peor distancia válida.
* Número de rutas válidas.
* Número de hormigas.
* Tiempo por iteración.
* Tiempo total acumulado.
* Estadísticos de feromonas.

### `historiales_sa.csv`

Contiene la evolución iteración a iteración de Simulated Annealing.

Incluye métricas como:

* Mejor distancia global.
* Distancia actual.
* Distancia del vecino generado.
* Temperatura.
* Número de soluciones aceptadas.
* Si el vecino fue aceptado en esa iteración.
* Tiempo por iteración.
* Tiempo total acumulado.

### `resumen_estadistico.csv`

Contiene la comparativa agregada por escenario y algoritmo.

Incluye:

* Número de ejecuciones.
* Media de la distancia.
* Desviación típica.
* Mediana.
* Mejor distancia.
* Peor distancia.
* Tiempo medio.
* Desviación típica del tiempo.
* Mediana del tiempo.
* Coeficiente de variación de la distancia.

## 11. Interpretación de resultados

La métrica principal de calidad es la distancia total de la ruta.

Como los escenarios están normalizados en una caja unitaria, esta distancia se interpreta como:

```text
distancia total normalizada
```

Un menor valor indica una ruta más corta dentro del grafo generado.

Además de la distancia, se analiza el tiempo de ejecución. Los tiempos están expresados en segundos, ya que se calculan mediante `time.perf_counter()`.

En la comparación estadística se consideran especialmente:

* La media de distancia.
* La mejor distancia obtenida.
* La desviación típica.
* El coeficiente de variación.
* El tiempo medio de ejecución.

Un algoritmo será más estable si presenta baja desviación típica y bajo coeficiente de variación.

## 12. Visualización

Las rutas generadas pueden visualizarse de dos formas:

1. Mediante imágenes `.png` exportadas automáticamente.
2. Mediante archivos `.ply` abiertos en MeshLab, Open3D o CloudCompare.

El archivo de ruta se exporta como una malla formada por cilindros entre puntos consecutivos.

El grafo también puede exportarse en formato `.ply` para visualizar las conexiones válidas entre viewpoints.

## 13. Dependencias

El proyecto utiliza las siguientes librerías principales:

```text
numpy
pandas
scipy
scikit-learn
open3d
pyvista
```

Instalación orientativa:

```bash
pip install numpy pandas scipy scikit-learn open3d pyvista
```

## 14. Flujo general del sistema

El flujo de ejecución completo es:

```text
Malla 3D
   ↓
Generación de viewpoints
   ↓
Filtrado geométrico
   ↓
Agrupamiento DBSCAN
   ↓
Construcción del grafo de visibilidad
   ↓
Aplicación de algoritmos de planificación
   ↓
Obtención de la mejor ruta
   ↓
Exportación de resultados
   ↓
Análisis estadístico y visualización
```

## 15. Notas metodológicas

El problema se resuelve como un TSP abierto, por lo que la ruta no tiene que volver obligatoriamente al nodo inicial.

La matriz de distancias representa un grafo no completamente conectado por visibilidad directa, aunque se comprueba que el grafo sea conexo antes de ejecutar los algoritmos. Si el grafo no es conexo, se recomienda aumentar el número de vecinos `k_vecinos` o revisar la generación de viewpoints.

Las variantes ACO construyen soluciones mediante agentes artificiales que recorren el grafo utilizando información de feromonas y una heurística basada en la inversa de la distancia.

Simulated Annealing genera soluciones vecinas mediante operadores discretos sobre la permutación de nodos y acepta o rechaza cambios en función de la mejora obtenida y de una temperatura decreciente.

## 16. Salidas generadas

Durante la ejecución pueden generarse archivos como:

```text
01_D_puntos_desplazados_filtrados.ply
02_centroides_primera_fase.ply
03_centroides_filtrados_distancia.ply
04_viewpoints_finales.ply
grafo_escenarioX_k20.ply
ruta_algoritmo_meshlab.ply
historial_algoritmo.csv
ruta_algoritmo_TSP_Abierto.png
```

En los experimentos estadísticos se generan:

```text
resultados_ejecuciones.csv
historiales_aco.csv
historiales_sa.csv
resumen_estadistico.csv
```

## 17. Consideraciones finales

El sistema permite evaluar diferentes algoritmos de planificación sobre escenarios 3D normalizados. La comparación entre algoritmos es válida porque todos se ejecutan sobre los mismos grafos y bajo las mismas condiciones experimentales.

No obstante, las distancias obtenidas deben interpretarse como distancias normalizadas, no como unidades físicas absolutas. Para obtener distancias reales sería necesario conocer la escala original de cada escenario antes de su normalización.

El proyecto proporciona tanto resultados numéricos como salidas visuales, facilitando el análisis cuantitativo y cualitativo de las rutas generadas.