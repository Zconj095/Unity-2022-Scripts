import OpenGL.GL as gl

def swap_buffers():
    gl.glFlush()
    gl.glSwapBuffers()