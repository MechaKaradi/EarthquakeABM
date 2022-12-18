import os
print(os.getcwd())
sys.path.append(os.getcwd() + "\\rich_model")

from rich_model.server import server


server.launch()