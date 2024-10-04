import numpy as np

def generate_obj_file(filename, height, width):
    # Create a 3D grid of vertices
    vertices = np.linspace(0, height, width + 1)[:, np.newaxis, np.newaxis]
    vertices = np.repeat(vertices, width + 1, axis=2)
    vertices = np.tile(vertices, (1, width + 1, 1))

    # Create faces
    faces = []
    for i in range(width):
        for j in range(width):
            face = [
                j * (width + 1) + i,
                j * (width + 1) + i + 1,
                (j + 1) * (width + 1) + i + 1,
                (j + 1) * (width + 1) + i,
            ]
            faces.append(face)

    # Write the OBJ file
    with open(filename, 'w') as f:
        f.write('o cube\n')

        for vertex in vertices:
            f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

        for face in faces:
            f.write('f {} {} {} {}\n'.format(*face))

# Generate the OBJ file
filename = 'fantasyA1.fbx'
height = 100
width = 10

generate_obj_file(filename, height, width)
