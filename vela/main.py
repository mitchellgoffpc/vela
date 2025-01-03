import sys
import numpy as np
from dataclasses import dataclass
from OpenGL import GL
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent, QResizeEvent, QSurfaceFormat, QMatrix4x4
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QSlider, QLabel, QVBoxLayout
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVertexArrayObject

from vela.ui.transforms import rot_from_euler, look_at, projection_matrix
from vela.ui.shaders import create_shader_program, create_vao
from vela.geometry.urdf import load_urdf, LoadedMesh, Link, Joint, Origin

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

def build_transforms(links: dict[str, Link], joints: dict[str, Joint], joint_angles: dict[str, float]) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}
    parent_joints = {joint.child: joint for joint in joints.values()}

    def compute_transform(link_name: str) -> np.ndarray:
        if link_name in transforms:
            return transforms[link_name]
        elif link_name not in parent_joints:  # This is the root link
            transforms[link_name] = np.eye(4, dtype=np.float32)
        else:
            joint = parent_joints[link_name]
            world_to_parent = compute_transform(joint.parent)
            parent_to_joint = create_transform_matrix(joint.origin)
            joint_to_child = np.eye(4, dtype=np.float32)
            angle = joint_angles.get(joint.name, 0.0)
            if joint.type in ['revolute', 'continuous']:
                axis = np.array(joint.axis.xyz, dtype=np.float32)
                joint_to_child = rotation_matrix(axis, angle)
            elif joint.type == 'prismatic':
                axis = np.array(joint.axis.xyz, dtype=np.float32)
                joint_to_child[:3, 3] = axis * angle
            transforms[link_name] = world_to_parent @ parent_to_joint @ joint_to_child
        return transforms[link_name]

    for link in links.values():
        compute_transform(link.name)

    return transforms


# Main OpenGL Widget

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, links: list[Link], joints: list[Joint]):
        super().__init__()
        self.last_pos = QPoint()
        self.camera_rotation: list[float] = [0, 0]
        self.camera_radius: float = 1.0
        self.meshes: list[MeshObject] = []

        self.links = {link.name: link for link in links}
        self.joints = {joint.name: joint for joint in joints}
        self.joint_angles = {joint.name: 0.0 for joint in joints}
        self.transforms: dict[str, np.ndarray] = {}

    def update_transforms(self):
        self.transforms = build_transforms(self.links, self.joints, self.joint_angles)

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
        self.update_transforms()
        for link in self.links.values():
            for visual in link.visual:
                if isinstance(visual.geometry, LoadedMesh):
                    model_matrix = create_transform_matrix(visual.origin)
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
            model_matrix = self.transforms[mesh.link_name] @ mesh.model_matrix
            self.shader.setUniformValue("model", QMatrix4x4(*model_matrix.flatten()))
            mesh.vao.bind()
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh.n_elements)
            mesh.vao.release()

        self.shader.release()


# Main application window

class MainWindow(QMainWindow):
    def __init__(self, urdf_path: str):
        super().__init__()
        self.setWindowTitle("Vela")
        self.resize(800, 600)

        # Create opengl widget
        links, joints = load_urdf(urdf_path)
        self.opengl_widget = OpenGLWidget(links, joints)
        self.setCentralWidget(self.opengl_widget)

        # Create sliders widget
        sliders_layout = QVBoxLayout()
        for joint in joints:
            if joint.type in ['revolute', 'continuous']:
                label = QLabel(joint.name)
                label.setStyleSheet("background-color: none; color: white;")
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setStyleSheet("background-color: none;")
                slider.setRange(-360, 360)
                slider.setValue(0)
                slider.valueChanged.connect(lambda value, joint_name=joint.name: self.on_slider_value_changed(joint_name, value))
                sliders_layout.addWidget(label)
                sliders_layout.addWidget(slider)

        self.sliders_widget = QWidget(self)
        self.sliders_widget.setLayout(sliders_layout)
        self.sliders_widget.setStyleSheet("background-color: rgba(0, 0, 0, 100);")
        self.sliders_widget.setGeometry(10, self.height() - 350, 300, 340)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        self.sliders_widget.setGeometry(10, self.height() - 350, 300, 340)

    def on_slider_value_changed(self, joint_name: str, value: int):
        self.opengl_widget.joint_angles[joint_name] = np.radians(value)
        self.opengl_widget.update_transforms()
        self.opengl_widget.update()


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
    main_window = MainWindow(sys.argv[1])
    main_window.show()
    sys.exit(app.exec())
