import os

def find_pycharm_project_root(start_path=None):
    current = os.path.abspath(start_path or os.getcwd())
    while current != os.path.dirname(current):  # Stop at root directory
        if '.idea' in os.listdir(current):
            return current
        current = os.path.dirname(current)
    return None