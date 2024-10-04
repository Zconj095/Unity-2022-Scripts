import subprocess
import sys

def install_qiskit():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])

# Install Qiskit
install_qiskit()