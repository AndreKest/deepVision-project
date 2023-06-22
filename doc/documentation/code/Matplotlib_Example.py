# Erstellt x-Werte, berechnet den Sinus und stellt das Ergebnis dar
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, num=100)
y = np.sin(x)

plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y , color="blue")
plt.show()