import sys
import math
import numpy as np
from dataclasses import dataclass
from OpenGL import GL
from PyQt6 import sip
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent, QSurfaceFormat, QMatrix4x4
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram

from vela.urdf import load_urdf, LoadedMesh, Link, Joint, Origin

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

@dataclass
class MeshObject:
    vao: QOpenGLVertexArrayObject
    n_elements: int
    model_matrix: np.ndarray

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    rotation = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    translation = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return np.dot(rotation, translation)

def projection_matrix(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def rot_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx

def create_transform_matrix(origin: Origin) -> np.ndarray:
    translation = np.eye(4, dtype=np.float32)
    rotation = np.eye(4, dtype=np.float32)

    tx, ty, tz = origin.xyz
    translation = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    roll, pitch, yaw = origin.rpy
    rotation = rot_from_euler(roll, pitch, yaw)

    return translation @ rotation

def build_transformations(links: list[Link], joints: list[Joint]) -> dict[str, np.ndarray]:
    transformations: dict[str, np.ndarray] = {}
    joint_dict = {joint.child: joint for joint in joints}

    def compute_transform(link_name: str) -> np.ndarray:
        if link_name in transformations:
            return transformations[link_name]
        elif link_name not in joint_dict:
            transformations[link_name] = np.eye(4, dtype=np.float32)
        else:
            joint = joint_dict[link_name]
            parent_transform = compute_transform(joint.parent)
            joint_transform = create_transform_matrix(joint.origin)
            transformations[link_name] = parent_transform @ joint_transform

        return transformations[link_name]

    for link in links:
        compute_transform(link.name)

    return transformations

def create_buffer(data: np.ndarray, buffer_type: QOpenGLBuffer.Type) -> QOpenGLBuffer:
    buffer = QOpenGLBuffer(buffer_type)
    buffer.create()
    buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
    buffer.bind()
    buffer.allocate(sip.voidptr(data.tobytes()), data.nbytes)
    return buffer

def create_vao(shader: QOpenGLShaderProgram, vertices: np.ndarray, normals: np.ndarray) -> QOpenGLVertexArrayObject:
    vao = QOpenGLVertexArrayObject()
    vao.create()
    vao.bind()

    vbo = create_buffer(vertices.flatten(), QOpenGLBuffer.Type.VertexBuffer)
    shader.setAttributeBuffer("aPos", GL.GL_FLOAT, 0, 3)
    shader.enableAttributeArray("aPos")
    vbo.release()

    nbo = create_buffer(normals.flatten(), QOpenGLBuffer.Type.VertexBuffer)
    shader.setAttributeBuffer("aNormal", GL.GL_FLOAT, 0, 3)
    shader.enableAttributeArray("aNormal")
    nbo.release()

    vao.release()
    return vao


# Main OpenGL Widget

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, urdf_path: str):
        super().__init__()
        self.setWindowTitle("Vela")
        self.resize(800, 600)
        self.last_pos = QPoint()
        self.rotation: list[float] = [0, 0]
        self.camera_radius: float = 1.0
        self.urdf_path: str = urdf_path
        self.meshes: list[MeshObject] = []

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event:
            self.last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event:
            dx = event.position().toPoint().x() - self.last_pos.x()
            dy = event.position().toPoint().y() - self.last_pos.y()

            if event.buttons() & Qt.MouseButton.LeftButton:
                self.rotation[0] -= dx * 0.2
                self.rotation[1] += dy * 0.2
                self.rotation[1] = max(-90, min(90, self.rotation[1]))
                self.update()

            self.last_pos = event.position().toPoint()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event:
            delta = event.angleDelta().y() / 120
            self.camera_radius -= delta * 0.5
            self.camera_radius = max(0.1, min(5.0, self.camera_radius))
            self.update()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event and event.key() == Qt.Key.Key_Escape:
            self.close()

    def initializeGL(self) -> None:
        self.shader = QOpenGLShaderProgram(self)
        self.shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader)
        self.shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader)
        self.shader.link()
        self.shader.bind()

        # Load URDF file
        links, joints = load_urdf(self.urdf_path)
        if not links:
            raise ValueError("No links found in the URDF file")

        # Load meshes
        transformations = build_transformations(links, joints)
        for link in links:
            for visual in link.visual:
                if isinstance(visual.geometry, LoadedMesh):
                    visual_transform = create_transform_matrix(visual.origin)
                    model_matrix = transformations[link.name] @ visual_transform
                    vao = create_vao(self.shader, visual.geometry.vertices, visual.geometry.normals)
                    self.meshes.append(MeshObject(vao, len(visual.geometry.vertices) * 3, model_matrix))

        # Set up projection and lighting uniforms
        fov, aspect, near, far = 45.0, 800.0 / 600.0, 0.01, 100.0
        projection = projection_matrix(fov, aspect, near, far)
        self.shader.setUniformValue("projection", QMatrix4x4(*projection.flatten()))
        self.shader.setUniformValue("lightPos", 0.0, 10.0, 5.0)
        self.shader.setUniformValue("lightColor", 1.0, 1.0, 1.0)
        self.shader.setUniformValue("objectColor", 1.0, 0.5, 0.2)
        self.shader.setUniformValue("ambientStrength", 0.2)
        self.shader.release()

        GL.glClearColor(0.0, 0.0, 0.4, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.bind()

        camera_x = self.camera_radius * math.sin(math.radians(self.rotation[0])) * math.cos(math.radians(self.rotation[1]))
        camera_y = self.camera_radius * math.sin(math.radians(self.rotation[1]))
        camera_z = self.camera_radius * math.cos(math.radians(self.rotation[0])) * math.cos(math.radians(self.rotation[1]))
        camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = look_at(camera_pos, camera_target, camera_up)
        self.shader.setUniformValue("view", QMatrix4x4(*view.flatten()))

        for mesh in self.meshes:
            self.shader.setUniformValue("model", QMatrix4x4(*mesh.model_matrix.flatten()))
            mesh.vao.bind()
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh.n_elements)
            mesh.vao.release()

        self.shader.release()


# Entry point

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <urdf_file>")
        sys.exit(1)

    urdf_path = sys.argv[1]

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = OpenGLWidget(urdf_path)
    w.show()
    sys.exit(app.exec())
