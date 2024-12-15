import sys
import numpy as np
from OpenGL import GL as gl
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt

vertex_shader = """
#version 330 core
in vec3 position;
void main()
{
    gl_Position = vec4(position, 1.0);
}
"""
fragment_shader = """
#version 330 core
void main()
{
    gl_FragColor = vec4(0.5, 0.2, 0.9, 1.0);
}
"""

class OpenGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Triangle, PyQt5, OpenGL 3.3")
        self.resize(400, 400)

    def initializeGL(self):
        gl.glClearColor(0.5, 0.8, 0.7, 1.0)

        program = QOpenGLShaderProgram(self)
        program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader)
        program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader)
        program.link()
        program.bind()

        vertices = np.array([
            -0.5, -0.5, 0.0,
            0.5, -0.5, 0.0,
            0.0, 0.5, 0.0], dtype=np.float32)

        self.vertPosBuffer = QOpenGLBuffer()
        self.vertPosBuffer.create()
        self.vertPosBuffer.bind()
        self.vertPosBuffer.allocate(vertices, len(vertices) * 4)

        program.bindAttributeLocation("position", 0)
        program.setAttributeBuffer(0, gl.GL_FLOAT, 0, 3)
        program.enableAttributeArray(0)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    a = QApplication(sys.argv)
    w = OpenGLWidget()
    w.show()
    sys.exit(a.exec())

if __name__ == "__main__":
    main()
