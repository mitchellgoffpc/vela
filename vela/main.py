import sys
import math
import numpy as np
from OpenGL import GL
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QSurfaceFormat, QMatrix4x4
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram

from vela.stl import parse_stl

vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform float ambientStrength;

void main() {
    // Ambient light
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse light
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

# Helper functions

def compute_vertex_normals(vertices):
    normals = np.zeros_like(vertices)
    for i in range(len(vertices)):
        v0, v1, v2 = vertices[i]
        normal = np.cross(v1 - v0, v2 - v0)
        normals[i] = normal[None].repeat(3, axis=0)
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    return normals.astype(np.float32)

def look_at(eye, target, up):
    # Compute the forward (z), right (x), and up (y) vectors
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Create the rotation matrix
    rotation = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Create the translation matrix
    translation = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Combine rotation and translation
    return np.dot(rotation, translation)

def projection_matrix(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def create_buffer(data, buffer_type):
    buffer = QOpenGLBuffer(buffer_type)
    buffer.create()
    buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
    buffer.bind()
    buffer.allocate(data.tobytes(), data.nbytes)
    return buffer


# Main OpenGL Widget

class OpenGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vela")
        self.resize(800, 600)
        self.last_pos = QPoint()
        self.rotation = [0, 0]
        self.camera_radius = 3.0

    def mousePressEvent(self, event):
        self.last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        dx = event.position().toPoint().x() - self.last_pos.x()
        dy = event.position().toPoint().y() - self.last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            self.rotation[0] -= dx * 0.2
            self.rotation[1] += dy * 0.2
            self.rotation[1] = max(-90, min(90, self.rotation[1]))  # Clamp vertical rotation
            self.update()

        self.last_pos = event.position().toPoint()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.camera_radius -= delta * 0.5
        self.camera_radius = max(1.0, min(50.0, self.camera_radius))  # Clamp zoom level
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def initializeGL(self):
        self.shader = QOpenGLShaderProgram(self)
        self.shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader)
        self.shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader)
        self.shader.link()
        self.shader.bind()

        vertices, normals = parse_stl("/Users/mitchell/Downloads/so-100 follower/STS3215.stl")
        normals = compute_vertex_normals(vertices)  # These normals look less weird than the ones in the STL
        vertices = vertices * 0.1  # This model is pretty big, so scale it down
        self.n_elements = len(vertices) * 3

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        vbo = create_buffer(vertices.flatten(), QOpenGLBuffer.Type.VertexBuffer)
        self.shader.setAttributeBuffer("aPos", GL.GL_FLOAT, 0, 3)
        self.shader.enableAttributeArray("aPos")

        nbo = create_buffer(normals.flatten(), QOpenGLBuffer.Type.VertexBuffer)
        self.shader.setAttributeBuffer("aNormal", GL.GL_FLOAT, 0, 3)
        self.shader.enableAttributeArray("aNormal")

        self.shader.setUniformValue("lightPos", 0.0, 10.0, 5.0)  # Light position
        self.shader.setUniformValue("lightColor", 1.0, 1.0, 1.0)  # White light
        self.shader.setUniformValue("objectColor", 1.0, 0.5, 0.2)  # Object color (orange)
        self.shader.setUniformValue("ambientStrength", 0.2)  # Ambient light strength

        # Create projection matrix
        fov, aspect, near, far = 45.0, 800.0 / 600.0, 0.1, 100.0
        projection = projection_matrix(fov, aspect, near, far)
        self.shader.setUniformValue("projection", QMatrix4x4(*projection.flatten()))

        # Create model matrix
        model = np.eye(4)
        self.shader.setUniformValue("model", QMatrix4x4(model.flatten()))

        vbo.release()
        nbo.release()
        self.vao.release()
        self.shader.release()

        GL.glClearColor(0.0, 0.0, 0.4, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.bind()
        self.vao.bind()

        # Create view matrix
        camera_x = self.camera_radius * math.sin(math.radians(self.rotation[0])) * math.cos(math.radians(self.rotation[1]))
        camera_y = self.camera_radius * math.sin(math.radians(self.rotation[1]))
        camera_z = self.camera_radius * math.cos(math.radians(self.rotation[0])) * math.cos(math.radians(self.rotation[1]))
        camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = look_at(camera_pos, camera_target, camera_up)
        self.shader.setUniformValue("view", QMatrix4x4(view.flatten()))

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_elements)

        self.vao.release()
        self.shader.release()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = OpenGLWidget()
    w.show()
    sys.exit(app.exec())
