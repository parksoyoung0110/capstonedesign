from __future__ import print_function

# pylint: disable=unused-import,g-import-not-at-top,g-statement-before-imports

try:
    import OpenGL
except:
    print('This module depends on PyOpenGL.')
    print('Please run "\033[1m!pip install -q pyopengl\033[0m" '
          'prior importing this module.')
    raise

import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import os

# GLFW 환경 설정
os.environ['PYOPENGL_PLATFORM'] = 'glfw'


def create_opengl_context(surface_size=(640, 480), version=(4, 6), debug=False):
    # GLFW 초기화
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    # OpenGL 컨텍스트를 숨기기 위해 윈도우를 보이지 않게 설정
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    # 윈도우 생성 (Headless 모드)
    window = glfw.create_window(surface_size[0], surface_size[1], "OpenGL Window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    # OpenGL 컨텍스트 활성화
    glfw.make_context_current(window)

    return window



def destroy_opengl_context(window):

    glfw.destroy_window(window)
    
    # GLFW를 종료합니다.
    glfw.terminate()