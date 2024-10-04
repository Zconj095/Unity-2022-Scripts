import os
import sys
import time

class SimpleGUI:
    def __init__(self):
        self.width = 400
        self.height = 300
        self.title = "Simple GUI"
        self.is_running = True

    def clear_screen(self):
        if os.name == 'nt':
            _ = os.system('cls')
        else:
            _ = os.system('clear')

    def draw_border(self):
        print("+" + "-" * (self.width - 2) + "+")
        for _ in range(self.height - 2):
            print("|" + " " * (self.width - 2) + "|")
        print("+" + "-" * (self.width - 2) + "+")

    def draw_button(self, x, y, text):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            print("\033[%d;%dH%s" % (y + i, x, line))

    def update(self):
        while self.is_running:
            self.clear_screen()
            self.draw_border()
            self.draw_button(10, 5, "[ Click Me ]")
            time.sleep(0.1)
            self.handle_input()

    def handle_input(self):
        import termios
        import fcntl
        import sys
        import os

        fd = sys.stdin.fileno()
        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

        try:
            while self.is_running:
                try:
                    c = sys.stdin.read(1)
                    if c == '\x1b':  # ESC key to exit
                        self.is_running = False
                    elif c == '\n':  # Enter key to simulate button click
                        self.button_click()
                except IOError:
                    pass
                time.sleep(0.1)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

    def button_click(self):
        print("\033[%d;%dH%s" % (self.height - 1, 1, "Button Clicked!"))

if __name__ == "__main__":
    gui = SimpleGUI()
    gui.update()
