import numpy as np
from OpenGL import GL
from PyQt6 import sip
from PyQt6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject, QOpenGLBuffer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

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


def create_shader_program(parent: QOpenGLWidget) -> QOpenGLShaderProgram:
    shader = QOpenGLShaderProgram(parent)
    shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader)
    shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader)
    shader.link()
    shader.bind()
    return shader

def create_buffer(data: np.ndarray, buffer_type: QOpenGLBuffer.Type) -> QOpenGLBuffer:
    buffer = QOpenGLBuffer(buffer_type)
    buffer.create()
    buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
    buffer.bind()
    buffer.allocate(sip.voidptr(data.data), data.nbytes)
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
