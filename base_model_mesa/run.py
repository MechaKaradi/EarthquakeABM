import os
print(os.getcwd())


from rich_model.server import server


server.launch()