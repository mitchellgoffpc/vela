import sys
import numpy as np
from dataclasses import dataclass
from OpenGL import GL
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent, QSurfaceFormat, QMatrix4x4
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVertexArrayObject

from vela.helpers.math import rot_from_euler, look_at, projection_matrix
from vela.helpers.urdf import load_urdf, LoadedMesh, Link, Joint, Origin
from vela.helpers.shaders import create_shader_program, create_vao

@dataclass
class MeshObject:
    link_name: str
    n_elements: int
    vao: QOpenGLVertexArrayObject
    model_matrix: np.ndarray


def create_transform_matrix(origin: Origin) -> np.ndarray:
    tx, ty, tz = origin.xyz
    r, p, y = origin.rpy
    translation = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    rotation = rot_from_euler(r, p, y)
    return translation @ rotation

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos = np.cos(angle)
    sin = np.sin(angle)
    c = 1 - cos
    return np.array([
        [cos + x*x*c,   x*y*c - z*sin, x*z*c + y*sin, 0],  # noqa: E226, E241
        [y*x*c + z*sin, cos + y*y*c,   y*z*c - x*sin, 0],  # noqa: E226, E241
        [z*x*c - y*sin, z*y*c + x*sin, cos + z*z*c,   0],  # noqa: E226, E241
        [0,             0,             0,             1]   # noqa: E226, E241
    ], dtype=np.float32)

def build_transformations(links: list[Link], joints: list[Joint], joint_angles: dict[str, float]) -> dict[str, np.ndarray]:
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
            if joint.type in ['revolute', 'continuous', 'prismatic']:
                angle = joint_angles.get(joint.name, 0.0)
                if joint.type in ['revolute', 'continuous']:
                    assert joint.axis
                    axis = np.array(joint.axis.xyz, dtype=np.float32)
                    rot_matrix = rotation_matrix(axis, angle)
                    joint_transform = joint_transform @ rot_matrix
                elif joint.type == 'prismatic':
                    assert joint.axis
                    axis = np.array(joint.axis.xyz, dtype=np.float32)
                    translation = np.eye(4, dtype=np.float32)
                    translation[:3, 3] = axis * angle
                    joint_transform = joint_transform @ translation
            transformations[link_name] = parent_transform @ joint_transform
        return transformations[link_name]

    for link in links:
        compute_transform(link.name)

    return transformations


# Main OpenGL Widget

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, urdf_path: str):
        super().__init__()
        self.setWindowTitle("Vela")
        self.resize(800, 600)
        self.last_pos = QPoint()
        self.camera_rotation: list[float] = [0, 0]
        self.camera_radius: float = 1.0
        self.meshes: list[MeshObject] = []

        self.links, self.joints = load_urdf(urdf_path)
        self.link_dict = {link.name: link for link in self.links}
        self.joint_dict = {joint.name: joint for joint in self.joints}
        self.joint_angles = {joint.name: 0.0 for joint in self.joints}

        revolute_joints = [joint.name for joint in self.joints if joint.type == 'continuous']
        self.first_revolute_joint = revolute_joints[0]

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

    def update_animation(self):
        if self.first_revolute_joint:
            self.joint_angles[self.first_revolute_joint] += 0.01
            self.joint_angles[self.first_revolute_joint] %= 2 * np.pi
            self.update_transformations()
            self.update()

    def update_transformations(self):
        self.transformations = build_transformations(self.links, self.joints, self.joint_angles)
        for mesh in self.meshes:
            link_name = mesh.link_name
            link = self.link_dict[link_name]
            for visual in link.visual:
                if isinstance(visual.geometry, LoadedMesh):
                    visual_transform = create_transform_matrix(visual.origin)
                    model_matrix = self.transformations[link_name] @ visual_transform
                    mesh.model_matrix = model_matrix
                    break

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event:
            self.last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event:
            dx = event.position().toPoint().x() - self.last_pos.x()
            dy = event.position().toPoint().y() - self.last_pos.y()
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.camera_rotation[0] -= dx * 0.2
                self.camera_rotation[1] += dy * 0.2
                self.camera_rotation[1] = max(-90, min(90, self.camera_rotation[1]))
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
        self.shader = create_shader_program(self)
        self.transformations = build_transformations(self.links, self.joints, self.joint_angles)
        for link in self.links:
            for visual in link.visual:
                if isinstance(visual.geometry, LoadedMesh):
                    visual_transform = create_transform_matrix(visual.origin)
                    model_matrix = self.transformations[link.name] @ visual_transform
                    vao = create_vao(self.shader, visual.geometry.vertices, visual.geometry.normals)
                    self.meshes.append(MeshObject(link.name, len(visual.geometry.vertices) * 3, vao, model_matrix))

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

        camera_x = self.camera_radius * np.sin(np.radians(self.camera_rotation[0])) * np.cos(np.radians(self.camera_rotation[1]))
        camera_y = self.camera_radius * np.sin(np.radians(self.camera_rotation[1]))
        camera_z = self.camera_radius * np.cos(np.radians(self.camera_rotation[0])) * np.cos(np.radians(self.camera_rotation[1]))
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

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = OpenGLWidget(sys.argv[1])
    w.show()
    sys.exit(app.exec())
