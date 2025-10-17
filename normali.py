import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def compute_face_normal(vertices, face_indices):
    """
    Вычисляет нормаль к грани через векторное произведение
    """
    # Берем первые три вершины грани для вычисления нормали
    v0 = vertices[face_indices[0]]
    v1 = vertices[face_indices[1]]
    v2 = vertices[face_indices[2]]
    
    # Векторы в плоскости грани
    vec1 = np.array(v1) - np.array(v0)
    vec2 = np.array(v2) - np.array(v0)
    
    # Векторное произведение
    normal = np.cross(vec1, vec2)
    
    # Нормализация
    length = np.linalg.norm(normal)
    if length > 0:
        normal = normal / length
    
    return normal

def compute_face_center(vertices, face_indices):
    """
    Вычисляет центр грани как среднее арифметическое её вершин
    """
    face_vertices = [vertices[i] for i in face_indices]
    center = np.mean(face_vertices, axis=0)
    return center

def compute_vertex_normals(vertices, faces):
    """
    Вычисляет вершинные нормали как среднее нормалей всех граней, содержащих вершину
    """
    vertex_normals = np.zeros((len(vertices), 3))
    face_normals = []
    
    # Сначала вычисляем нормали для всех граней
    for face in faces:
        normal = compute_face_normal(vertices, face[:3])
        face_normals.append(normal)
    
    # Для каждой вершины находим все грани, которые её содержат, и усредняем их нормали
    for i, vertex in enumerate(vertices):
        contributing_normals = []
        for j, face in enumerate(faces):
            if i in face:
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                face_normal = np.cross(v1 - v0, v2 - v0)
                face_normal = face_normal / np.linalg.norm(face_normal)
                contributing_normals.append(face_normal)
        
        if contributing_normals:
            # Усредняем все нормали граней, содержащих эту вершину
            contributing_normals_array = np.array(contributing_normals)
            avg_normal = np.mean(contributing_normals_array, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
        else:
            avg_normal = np.zeros(3)
        vertex_normals[i] = avg_normal
    return vertex_normals

def create_cube():
    """
    Создает вершины и грани куба
    """
    # 8 вершин куба со стороной 2, центрированного в начале координат
    vertices = np.array([
        [-1, -1, -1],  # 0: нижняя-задняя-левая
        [ 1, -1, -1],  # 1: нижняя-задняя-правая  
        [ 1,  1, -1],  # 2: нижняя-передняя-правая
        [-1,  1, -1],  # 3: нижняя-передняя-левая
        [-1, -1,  1],  # 4: верхняя-задняя-левая
        [ 1, -1,  1],  # 5: верхняя-задняя-правая
        [ 1,  1,  1],  # 6: верхняя-передняя-правая
        [-1,  1,  1]   # 7: верхняя-передняя-левая
    ])
    
    # Грани куба (каждая грань - 4 вершины)
    # Для визуализации каждая квадратная грань разбивается на 2 треугольника
    faces = [
        [0, 3, 2, 1],  # нижняя грань
        [4, 5, 6, 7],  # верхняя грань
        [0, 1, 5, 4],  # задняя грань
        [2, 3, 7, 6],  # передняя грань
        [0, 4, 7, 3],  # левая грань
        [1, 2, 6, 5]   # правая грань
    ]
    
    # Преобразуем квадратные грани в треугольники для лучшей визуализации
    triangle_faces = []
    for face in faces:
        # Разбиваем квадрат на два треугольника
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])
    
    return vertices, faces, triangle_faces

def visualize_vertex_normals():
    """
    Визуализирует куб с вершинными нормалями
    """
    # Создаем куб
    vertices, faces, triangle_faces = create_cube()
    
    # Вычисляем вершинные нормали
    vertex_normals = compute_vertex_normals(vertices, faces)
    
    # Создаем график
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Визуализируем куб
    cube_polygons = []
    for face in faces:
        polygon = [vertices[i] for i in face]
        cube_polygons.append(polygon)
    
    cube_collection = Poly3DCollection(cube_polygons, 
                                     alpha=0.3, 
                                     facecolors='lightgreen', 
                                     edgecolors='green',
                                     linewidths=2)
    ax.add_collection3d(cube_collection)
    
    # Визуализируем вершинные нормали
    normal_length = 0.8  # Длина нормалей для визуализации
    
    for i, (vertex, normal) in enumerate(zip(vertices, vertex_normals)):
        # Визуализируем вершинную нормаль
        ax.quiver(vertex[0], vertex[1], vertex[2],
                 normal[0], normal[1], normal[2],
                 length=normal_length, color='purple', 
                 arrow_length_ratio=0.15, linewidth=2,
                 label='Вершинная нормаль' if i == 0 else "")
        
        # Отмечаем вершины точками
        ax.scatter(vertex[0], vertex[1], vertex[2], color='black', s=50)
    
    # Настраиваем график
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('3D-геометрия: Визуализация куба с ВЕРШИННЫМИ нормалями', fontsize=14, fontweight='bold')
    
    # Устанавливаем равные масштабы по осям
    max_range = 3
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Добавляем легенду
    ax.legend()
    
    # Добавляем сетку для лучшего восприятия 3D-пространства
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем угол обзора
    ax.view_init(elev=20, azim=45)
    
    # Добавляем информационную панель
    info_text = "Куб с вершинными нормалями:\n• Зеленый: грани куба\n• Фиолетовый: вершинные нормали\n• Черные точки: вершины\n• Вершинные нормали усреднены из соседних граней"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
              verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    return vertices, vertex_normals

def visualize_cube_with_normals():
    """
    Визуализирует куб с нормалями граней
    """
    # Создаем куб
    vertices, faces, triangle_faces = create_cube()
    
    # Создаем график
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Визуализируем куб
    cube_polygons = []
    for face in faces:
        polygon = [vertices[i] for i in face]
        cube_polygons.append(polygon)
    
    cube_collection = Poly3DCollection(cube_polygons, 
                                     alpha=0.3, 
                                     facecolors='cyan', 
                                     edgecolors='blue',
                                     linewidths=2)
    ax.add_collection3d(cube_collection)
    
    # Вычисляем и визуализируем нормали для каждой грани
    normal_length = 1.5  # Длина нормалей для визуализации
    
    for face in faces:
        # Вычисляем центр грани
        center = compute_face_center(vertices, face)
        
        # Вычисляем нормаль грани (используем первые 3 вершины)
        normal = compute_face_normal(vertices, face[:3])
        
        # Визуализируем нормаль
        ax.quiver(center[0], center[1], center[2],
                 normal[0], normal[1], normal[2],
                 length=normal_length, color='red', 
                 arrow_length_ratio=0.2, linewidth=2,
                 label='Нормаль грани' if face == faces[0] else "")
    
    # Настраиваем график
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('3D-геометрия: Визуализация куба с нормалями ГРАНЕЙ', fontsize=14, fontweight='bold')
    
    # Устанавливаем равные масштабы по осям
    max_range = 3
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Добавляем легенду
    ax.legend()
    
    # Добавляем сетку для лучшего восприятия 3D-пространства
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем угол обзора
    ax.view_init(elev=20, azim=45)
    
    # Добавляем информационную панель
    info_text = "Куб с нормалями граней:\n• Синий: грани куба\n• Красный: нормали граней\n• Нормали перпендикулярны граням"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
              verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def compare_face_vs_vertex_normals():
    """
    Сравнивает нормали граней и вершинные нормали на одном графике
    """
    vertices, faces, triangle_faces = create_cube()
    vertex_normals = compute_vertex_normals(vertices, faces)
    
    fig = plt.figure(figsize=(14, 6))
    
    # График с нормалями граней
    ax1 = fig.add_subplot(121, projection='3d')
    cube_polygons = []
    for face in faces:
        polygon = [vertices[i] for i in face]
        cube_polygons.append(polygon)
    
    cube_collection = Poly3DCollection(cube_polygons, 
                                     alpha=0.2, 
                                     facecolors='cyan', 
                                     edgecolors='blue',
                                     linewidths=1)
    ax1.add_collection3d(cube_collection)
    
    # Нормали граней
    for face in faces:
        center = compute_face_center(vertices, face)
        normal = compute_face_normal(vertices, face[:3])
        ax1.quiver(center[0], center[1], center[2],
                  normal[0], normal[1], normal[2],
                  length=1.2, color='red', arrow_length_ratio=0.2, linewidth=2)
    
    ax1.set_title('Нормали граней (Face Normals)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # График с вершинными нормалями
    ax2 = fig.add_subplot(122, projection='3d')
    cube_collection2 = Poly3DCollection(cube_polygons, 
                                      alpha=0.2, 
                                      facecolors='lightgreen', 
                                      edgecolors='green',
                                      linewidths=1)
    ax2.add_collection3d(cube_collection2)
    
    # Вершинные нормали
    for i, (vertex, normal) in enumerate(zip(vertices, vertex_normals)):
        ax2.quiver(vertex[0], vertex[1], vertex[2],
                  normal[0], normal[1], normal[2],
                  length=0.8, color='purple', arrow_length_ratio=0.15, linewidth=2)
        ax2.scatter(vertex[0], vertex[1], vertex[2], color='black', s=30)
    
    ax2.set_title('Вершинные нормали (Vertex Normals)', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Общие настройки
    for ax in [ax1, ax2]:
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_zlim([-2.5, 2.5])
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

def transformation_demo():
    """
    Демонстрация трансформаций куба
    """
    # Создаем оригинальный куб
    vertices, faces, triangle_faces = create_cube()
    
    # Матрицы трансформации
    # 1. Перемещение
    translation_matrix = np.array([
        [1, 0, 0, 2],  # перемещение на 2 по X
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 2. Масштабирование
    scaling_matrix = np.array([
        [1.5, 0, 0, 0],  # масштаб 1.5 по X
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 3. Поворот вокруг оси Z на 45 градусов
    angle = np.pi / 4  # 45 градусов
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Применяем трансформации к вершинам
    def transform_vertices(vertices, matrix):
        homogeneous_vertices = np.hstack([vertices, np.ones((len(vertices), 1))])
        transformed = (matrix @ homogeneous_vertices.T).T
        return transformed[:, :3]
    
    translated_vertices = transform_vertices(vertices, translation_matrix)
    scaled_vertices = transform_vertices(vertices, scaling_matrix)
    rotated_vertices = transform_vertices(vertices, rotation_matrix)
    
    # Визуализируем все варианты
    fig = plt.figure(figsize=(15, 5))
    
    # Оригинальный куб
    ax1 = fig.add_subplot(141, projection='3d')
    plot_cube(ax1, vertices, faces, 'Исходный куб', 'blue')
    
    # Перемещенный куб
    ax2 = fig.add_subplot(142, projection='3d')
    plot_cube(ax2, translated_vertices, faces, 'Перемещение', 'green')
    
    # Масштабированный куб  
    ax3 = fig.add_subplot(143, projection='3d')
    plot_cube(ax3, scaled_vertices, faces, 'Масштабирование', 'orange')
    
    # Повернутый куб
    ax4 = fig.add_subplot(144, projection='3d')
    plot_cube(ax4, rotated_vertices, faces, 'Поворот', 'red')
    
    plt.tight_layout()
    plt.show()

def plot_cube(ax, vertices, faces, title, color):
    """Вспомогательная функция для отрисовки куба"""
    cube_polygons = []
    for face in faces:
        polygon = [vertices[i] for i in face]
        cube_polygons.append(polygon)
    
    cube_collection = Poly3DCollection(cube_polygons, 
                                     alpha=0.5, 
                                     facecolors=color, 
                                     edgecolors='dark' + color,
                                     linewidths=1)
    ax.add_collection3d(cube_collection)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    max_range = 4
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

def calculate_normals_demo():
    """
    Демонстрация расчета нормалей для всех граней
    """
    vertices, faces, triangle_faces = create_cube()
    
    print("=" * 60)
    print("РАСЧЕТ НОРМАЛЕЙ ДЛЯ КУБА")
    print("=" * 60)
    
    face_names = ["Нижняя", "Верхняя", "Задняя", "Передняя", "Левая", "Правая"]
    
    for i, face in enumerate(faces):
        normal = compute_face_normal(vertices, face[:3])
        center = compute_face_center(vertices, face)
        
        print(f"\n{face_names[i]} грань (вершины {face}):")
        print(f"  Центр: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Нормаль: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})")
        
        # Проверяем перпендикулярность (скалярное произведение с векторами грани должно быть ~0)
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        edge_vector = np.array(v1) - np.array(v0)
        dot_product = np.dot(normal, edge_vector)
        print(f"  Проверка перпендикулярности (скалярное произведение): {dot_product:.2e}")

# Главная функция
if __name__ == "__main__":
    print("3D-геометрия: Модели, нормали, трансформация и визуализация куба")
    print("=" * 70)
    
    # Демонстрация расчета нормалей
    calculate_normals_demo()
    
    # Визуализация куба с нормалями граней
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ КУБА С НОРМАЛЯМИ ГРАНЕЙ")
    print("=" * 70)
    visualize_cube_with_normals()
    
    # Визуализация куба с вершинными нормалями
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ КУБА С ВЕРШИННЫМИ НОРМАЛЯМИ")
    print("=" * 70)
    vertices, vertex_normals = visualize_vertex_normals()
    
    # Выводим информацию о вершинных нормалях
    print("\nВершинные нормали:")
    for i, (vertex, normal) in enumerate(zip(vertices, vertex_normals)):
        print(f"Вершина {i}: ({vertex[0]:.1f}, {vertex[1]:.1f}, {vertex[2]:.1f}) -> "
              f"Нормаль: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})")
    
    # Сравнение двух типов нормалей
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ НОРМАЛЕЙ ГРАНЕЙ И ВЕРШИННЫХ НОРМАЛЕЙ")
    print("=" * 70)
    compare_face_vs_vertex_normals()
    
    # Демонстрация трансформаций
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 3D-ТРАНСФОРМАЦИЙ")
    print("=" * 70)
    transformation_demo()
    
    print("\nПрограмма завершена! Все визуализации были показаны.")