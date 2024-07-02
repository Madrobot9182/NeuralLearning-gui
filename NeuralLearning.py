'''
Execute this file to create and run the application
Author: Madrobot9182 - https://github.com/Madrobot9182
'''


from PyQt6.QtWidgets import QApplication
import app


def main():
    program = QApplication([])
    gui = app.GUI()
    gui.show()

    program.exec()


if __name__ == "__main__":
    main()