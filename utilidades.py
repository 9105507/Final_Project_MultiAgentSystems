import numpy as np
import open3d as o3d
import os
from pathlib import Path
import tkinter as tk

def crear_cilindro_entre_puntos(p1, p2, radio=0.006, color=(0.0, 0.0, 1.0)):
    """
    Crea un cilindro 3D que une dos puntos del espacio.

    La función genera un cilindro de Open3D alineado entre los puntos p1 y p2.
    Por defecto, Open3D crea los cilindros alineados con el eje Z, por lo que
    la función calcula la rotación necesaria para orientar el cilindro en la
    dirección del segmento definido por ambos puntos.

    Este tipo de geometría resulta útil para visualizar aristas, trayectorias,
    rutas o conexiones entre viewpoints en herramientas como MeshLab, Open3D o
    CloudCompare.

    Parámetros
    ----------
    p1 : array-like de shape (3,)
        Punto inicial del segmento. Debe contener las coordenadas (x, y, z).

    p2 : array-like de shape (3,)
        Punto final del segmento. Debe contener las coordenadas (x, y, z).

    radio : float, opcional
        Radio del cilindro generado. Controla el grosor visual de la conexión.
        Por defecto es 0.006.

    color : tuple de float, opcional
        Color uniforme asignado al cilindro en formato RGB normalizado, con
        valores entre 0.0 y 1.0. Por defecto es azul: (0.0, 0.0, 1.0).

    Retorna
    -------
    cilindro : open3d.geometry.TriangleMesh o None
        Malla triangular correspondiente al cilindro entre p1 y p2.
        Devuelve None si ambos puntos están demasiado cerca y no es posible
        crear un cilindro con longitud válida.

    Notas
    -----
    El cilindro se crea inicialmente centrado y alineado con el eje Z. Después
    se rota para hacerlo coincidir con la dirección del segmento p1-p2 y se
    traslada hasta el punto medio entre ambos puntos.

    Si p1 y p2 son prácticamente iguales, la longitud del segmento es nula y la
    función devuelve None para evitar errores geométricos.
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

def exportar_ruta_meshlab(viewpoints, camino, output_dir, nombre_ruta="ruta_aco_meshlab.ply", radio_ruta=0.006, color_ruta=(0.0, 0.0, 1.0)):
    """
    Exporta una ruta 3D como una malla PLY formada por cilindros.

    La función recibe un conjunto de viewpoints y una secuencia de índices que
    define el orden de visita de una ruta. A partir de esos puntos, construye
    una geometría formada por cilindros entre cada par de puntos consecutivos
    del camino y la guarda en formato PLY.

    El archivo generado puede abrirse en MeshLab, Open3D o CloudCompare para
    visualizar la trayectoria calculada.

    Parámetros
    ----------
    viewpoints : np.ndarray de shape (N, 3)
        Conjunto completo de puntos de vista disponibles. Cada fila representa
        un viewpoint con coordenadas (x, y, z).

    camino : list de int
        Lista con los índices de los viewpoints que forman la ruta. El orden de
        la lista determina el orden de conexión entre puntos.
        Por ejemplo: [4, 2, 7, 1].

    output_dir : str
        Carpeta donde se guardará el archivo PLY generado. Si no existe, se crea
        automáticamente.

    nombre_ruta : str, opcional
        Nombre del archivo PLY de salida. Por defecto es
        "ruta_aco_meshlab.ply".

    radio_ruta : float, opcional
        Radio de los cilindros que representan los tramos de la ruta. Controla
        el grosor visual de la trayectoria exportada. Por defecto es 0.006.

    color_ruta : tuple de float, opcional
        Color uniforme de la ruta en formato RGB normalizado, con valores entre
        0.0 y 1.0. Por defecto es azul: (0.0, 0.0, 1.0).

    Retorna
    -------
    path_ruta : str
        Ruta completa del archivo PLY generado.
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

    path_ruta = Path(output_dir) / nombre_ruta
    o3d.io.write_triangle_mesh(path_ruta, mesh_ruta)
    print(f"\n└──Ruta guardada en: {path_ruta}")

    return path_ruta

def centrar_ventana(ancho_ventana=1000, alto_ventana=800):
    """
    Calcula la posición necesaria para centrar una ventana en la pantalla.

    La función obtiene la resolución actual de la pantalla mediante Tkinter y
    calcula las coordenadas superiores izquierdas que debe tener una ventana
    para aparecer centrada.

    Parámetros
    ----------
    ancho_ventana : int, opcional
        Ancho de la ventana que se desea centrar, expresado en píxeles.
        Por defecto es 1000.

    alto_ventana : int, opcional
        Alto de la ventana que se desea centrar, expresado en píxeles.
        Por defecto es 800.

    Retorna
    -------
    x : int
        Coordenada horizontal de la esquina superior izquierda de la ventana.

    y : int
        Coordenada vertical de la esquina superior izquierda de la ventana.

    Notas
    -----
    El sistema de coordenadas de la pantalla tiene el origen en la esquina
    superior izquierda. Por tanto, valores mayores de x desplazan la ventana
    hacia la derecha y valores mayores de y la desplazan hacia abajo.
    """
    root = tk.Tk()
    root.withdraw()

    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()

    root.destroy()

    x = (ancho_pantalla - ancho_ventana) // 2
    y = (alto_pantalla - alto_ventana) // 2

    return x, y

def visualizar_ruta_3d_con_mesh(
    mesh_path,
    ruta_3d,
    titulo="Ruta óptima sobre malla 3D",
    algoritmo = "ACO",
    save_path=None,
    max_triangulos=30000,
    color_mesh="#d9d6cf",
    color_ruta="crimson",
    grosor_ruta=0.008,
    radio_puntos=0.015,
    radio_inicio_fin=0.035,
    mostrar_ventana=True,
    ventana=(1200, 1000),
    posicion_ventana=None,
    zoom = 0.85,
    elev=-10
):
    """
    Visualiza una ruta 3D sobre una malla usando PyVista.

    La función carga una malla 3D desde disco, la limpia, la simplifica si supera
    un número máximo de triángulos y la convierte a un formato compatible con
    PyVista. Sobre esta malla se representa una ruta 3D mediante un tubo continuo,
    puntos intermedios y marcadores diferenciados para el inicio y el final.

    La visualización utiliza el z-buffer real de PyVista, por lo que la malla se
    comporta como un objeto opaco: las partes de la ruta que quedan por detrás
    de la geometría no se muestran, mientras que las partes visibles aparecen
    correctamente delante de la malla.

    La función puede utilizarse en modo interactivo, mostrando una ventana de
    visualización, o en modo no interactivo, guardando una captura de la escena
    en una imagen.

    Parámetros
    ----------
    mesh_path : str o pathlib.Path
        Ruta del archivo de la malla 3D que se desea visualizar. Debe ser un
        formato compatible con Open3D, como PLY, OBJ, STL u OFF.

    ruta_3d : np.ndarray de shape (N, 3)
        Secuencia de puntos 3D que define la ruta a representar. Cada fila
        contiene las coordenadas (x, y, z) de un punto de la trayectoria.

    titulo : str, opcional
        Título que se mostrará en la ventana o imagen generada.
        Por defecto es "Ruta óptima sobre malla 3D".

    algoritmo : str, opcional
        Nombre del algoritmo utilizado para calcular la ruta. Se usa en la
        leyenda de la visualización. Por defecto es "ACO".

    save_path : str o None, opcional
        Ruta donde se guardará la imagen generada. Si es None, no se guarda
        ninguna captura. Por defecto es None.

    max_triangulos : int, opcional
        Número máximo de triángulos permitidos en la malla antes de aplicar una
        simplificación. Si la malla original supera este valor, se reduce su
        complejidad mediante decimación cuadrática. Por defecto es 30000.

    color_mesh : str o tuple, opcional
        Color de la malla. Puede indicarse como nombre de color, código
        hexadecimal o tupla RGB compatible con PyVista. Por defecto es
        "#d9d6cf".

    color_ruta : str o tuple, opcional
        Color utilizado para representar la ruta y los puntos intermedios.
        Por defecto es "crimson".

    grosor_ruta : float, opcional
        Radio del tubo que representa la ruta. Controla el grosor visual de la
        trayectoria. Por defecto es 0.008.

    radio_puntos : float, opcional
        Radio de las esferas que representan los puntos intermedios de la ruta.
        Por defecto es 0.015.

    radio_inicio_fin : float, opcional
        Radio de las esferas que representan el punto inicial y el punto final.
        Por defecto es 0.035.

    mostrar_ventana : bool, opcional
        Si es True, abre una ventana interactiva con la visualización 3D.
        Si es False, la visualización se realiza en modo off-screen, útil para
        guardar imágenes sin mostrar la ventana. Por defecto es True.

    ventana : tuple de int, opcional
        Tamaño de la ventana de visualización en píxeles, con formato
        (ancho, alto). Por defecto es (1200, 1000).

    posicion_ventana : tuple de int o None, opcional
        Posición de la esquina superior izquierda de la ventana, con formato
        (x, y). Si es None, la ventana se centra automáticamente en la pantalla.
        Por defecto es None.

    zoom : float, opcional
        Factor de zoom aplicado a la cámara tras ajustar la vista. Valores
        mayores a 1 acercan la cámara y valores menores a 1 la alejan. Por defecto es
        0.85.

    elev : float, opcional
        Ángulo de elevación aplicado a la cámara después de colocarla en vista
        isométrica. Permite inclinar la vista verticalmente. Por defecto es -10.

    Retorna
    -------
    save_path : str o None
        Si mostrar_ventana es False y save_path no es None, devuelve la ruta de
        la imagen guardada. En el resto de casos devuelve None.

    Lanza
    -----
    ValueError
        Si ruta_3d no tiene forma (N, 3).

    ValueError
        Si la malla no puede cargarse correctamente desde mesh_path.

    ValueError
        Si la malla cargada no contiene vértices o triángulos válidos.

    Notas
    -----
    La función utiliza Open3D para cargar, limpiar y simplificar la malla, y
    PyVista para la visualización final.

    La ruta se representa como un tubo generado a partir de líneas entre puntos
    consecutivos. Esto mejora la visibilidad frente a una línea simple,
    especialmente en escenas 3D con perspectiva.

    El punto inicial se representa en verde y el punto final en azul. Los puntos
    intermedios y el tubo de la ruta usan el color definido en color_ruta.

    Los ejes de coordenadas se muestran en la esquina inferior derecha mediante
    el parámetro viewport de plotter.add_axes().

    La leyenda se muestra en la esquina inferior izquierda e incluye la ruta, el
    punto inicial y el punto final.
    """

    import pyvista as pv

    ruta_3d = np.asarray(ruta_3d, dtype=float)

    if ruta_3d.ndim != 2 or ruta_3d.shape[1] != 3:
        raise ValueError("ruta_3d debe tener forma (N, 3)")

    # Cargar mesh con Open3D
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)

    if mesh_o3d.is_empty():
        raise ValueError(f"No se pudo cargar la mesh desde: {mesh_path}")

    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_non_manifold_edges()

    # Simplificar si hace falta
    if len(mesh_o3d.triangles) > max_triangulos:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(
            target_number_of_triangles=max_triangulos
        )
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_non_manifold_edges()

    vertices = np.asarray(mesh_o3d.vertices, dtype=float)
    triangles = np.asarray(mesh_o3d.triangles, dtype=int)

    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("La mesh no tiene vértices o triángulos válidos.")

    # Convertir mesh a formato PyVista
    faces = np.hstack([
        np.full((triangles.shape[0], 1), 3),
        triangles
    ]).astype(np.int64)

    mesh_pv = pv.PolyData(vertices, faces)

    # Crear línea de ruta
    puntos_ruta = pv.PolyData(ruta_3d)

    lineas = []
    for i in range(len(ruta_3d) - 1):
        lineas.extend([2, i, i + 1])

    puntos_ruta.lines = np.array(lineas)

    # Convertir ruta en tubo para que se vea bien
    tubo_ruta = puntos_ruta.tube(radius=grosor_ruta)

    # Plotter
    plotter = pv.Plotter(
        off_screen=not mostrar_ventana,
        window_size=ventana
    )

    # Centrar ventana automáticamente si no se indica posición
    if mostrar_ventana:
        if posicion_ventana is None:
            posicion_ventana = centrar_ventana(*ventana)
            print(posicion_ventana)

        plotter.ren_win.SetPosition(
            posicion_ventana[0],
            posicion_ventana[1]
        )

    # Mesh opaca
    plotter.add_mesh(
        mesh_pv,
        color=color_mesh,
        opacity=1.0,
        show_edges=False,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.75,
        specular=0.08
    )

    # Ruta como tubo rojo
    plotter.add_mesh(
        tubo_ruta,
        color=color_ruta,
        opacity=1.0,
        smooth_shading=True,
        label=f"Ruta {algoritmo}"
    )

    # Puntos de la ruta
    for p in ruta_3d:
        esfera = pv.Sphere(radius=radio_puntos, center=p)
        plotter.add_mesh(
            esfera,
            color=color_ruta,
            smooth_shading=True
        )

    # Inicio
    esfera_inicio = pv.Sphere(radius=radio_inicio_fin, center=ruta_3d[0])
    plotter.add_mesh(
        esfera_inicio,
        color="limegreen",
        smooth_shading=True,
        label="Inicio"
    )

    # Fin
    esfera_fin = pv.Sphere(radius=radio_inicio_fin, center=ruta_3d[-1])
    plotter.add_mesh(
        esfera_fin,
        color="blue",
        smooth_shading=True,
        label="Fin"
    )

    plotter.add_title(titulo, font_size=14)
    # plotter.add_axes()
    plotter.add_axes(
        viewport=(0.80, 0.00, 1.00, 0.20)
    )
    # plotter.add_legend()
    plotter.add_legend(
        loc="lower left",
        border=True
    )

    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.camera.Elevation(elev)
    plotter.camera.zoom(zoom)

    # Crear carpeta si se va a guardar imagen
    if save_path is not None:
        carpeta = os.path.dirname(save_path)

        if carpeta != "":
            os.makedirs(carpeta, exist_ok=True)

    # Opción 1: mostrar ventana interactiva
    if mostrar_ventana:
        plotter.show()
        return None

    # Opción 2: guardar imagen sin mostrar ventana
    if save_path is not None:
        plotter.show(auto_close=False, interactive=False)
        plotter.render()
        plotter.screenshot(save_path)
        plotter.close()

        print(f"\n└──Imagen guardada en: {save_path}")
        return save_path

    # Si no se muestra ventana ni se guarda imagen
    plotter.close()
    return None