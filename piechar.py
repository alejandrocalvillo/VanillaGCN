import matplotlib.pyplot as plt

# Definimos los datos
tareas = ['Tarea 1', 'Tarea 2', 'Tarea 3', 'Tarea 4']
tiempo = [20, 30, 15, 35] # tiempo en minutos

# Definimos los colores para cada tarea
colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Creamos el diagrama de quesos
plt.pie(tiempo, labels=tareas, colors=colores, autopct='%1.1f%%', startangle=90)

# Mostramos el diagrama
plt.axis('equal')
plt.show()
