import sys
from OpenGL import GL
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram

from vela.stl import parse_stl

vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
void main()
{
    gl_Position = vec4(position, 1.0);
}
"""

fragment_shader = """
#version 330 core
layout(location = 0, index = 0) out vec4 out_color;
void main()
{
    out_color = vec4(0.5, 0.2, 0.9, 1.0);
}
"""


class OpenGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Triangle, PyQt5, OpenGL 3.3")
        self.resize(400, 400)

    def initializeGL(self):
        self.shader = QOpenGLShaderProgram(self)
        self.shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader)
        self.shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader)
        self.shader.link()
        self.shader.bind()

        vertices, normals = parse_stl('/Users/mitchell/Downloads/so-100 leader/STS3215.stl')
        vertices[:, :, 0] -= 100
        self.n_triangles = vertices.shape[0]

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        self.vbo = QOpenGLBuffer()
        self.vbo.create()
        self.vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self.vbo.bind()
        self.vbo.allocate(vertices.tobytes(), vertices.nbytes)

        self.shader.setAttributeBuffer("position", GL.GL_FLOAT, 0, 3);
        self.shader.enableAttributeArray("position");

        self.vao.release()
        self.vbo.release()
        self.shader.release()

        GL.glClearColor(0.2, 0.2, 0.2, 1)

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        self.shader.bind()
        self.vao.bind()

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_triangles)

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
