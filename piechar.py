import matplotlib.pyplot as plt
import torch
# # Definimos los datos
# tareas = ['Tarea 1', 'Tarea 2', 'Tarea 3', 'Tarea 4']
# tiempo = [20, 30, 15, 35] # tiempo en minutos

# # Definimos los colores para cada tarea
# colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# # Creamos el diagrama de quesos
# plt.pie(tiempo, labels=tareas, colors=colores, autopct='%1.1f%%', startangle=90)

# # Mostramos el diagrama
# plt.axis('equal')
# plt.show()

# Generar un tensor aleatorio de forma (20, 9, 1) con valores entre 0 y 1
tensor = torch.rand(20, 9, 1)

# Concatenar los datos a lo largo de la primera dimensión
concatenated_data = torch.cat(torch.unbind(tensor), dim=0)

# Calcular la suma acumulada de los datos
cumulative_sum = torch.cumsum(concatenated_data, dim=0)

# Calcular la CDF
cdf = cumulative_sum / torch.sum(concatenated_data)

# Visualizar la CDF
plt.plot(cdf, label='CDF conjunta')
plt.title('Función de distribución acumulativa del delay')
plt.xlabel('Valor del delay')
plt.ylabel('Valor acumulado')

plt.legend()
plt.show()