import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loading files
csv_x1 = r"./pressure_8734.csv"
csv_x2 = r"./pressure_8606.csv"

df1 = pd.read_csv(csv_x1)
df2 = pd.read_csv(csv_x2)

# segnali di pressione
x1 = df1["pressure_value"].values
x2 = df2["pressure_value"].values

# --- CODICE ---

p34 = open(csv_x1, "r") # RICORDA DI CHIUDERE IL FILE CON p34.close()
p34.readline()
p34.readline()

ore = list()
pressioni = list()
for riga in p34.readlines():
    r = riga.strip("\n").split(",")
    ore.append(int(r[0].split(":")[0]))
    pressioni.append(float(r[1]))

ore = np.array(ore)
pressioni = np.array(pressioni)

p34.close()


#funzioni utili
def rect(n):
    return np.where(np.abs(n) <= 0.5, 1, 0)

def tri(n):
    return np.where(np.abs(n) <= 1, 1 - np.abs(n), 0)

def valore_medio(sig):
    return np.mean(sig)

def energia(sig):
    return np.sum(np.dot(sig, sig))

def sinc_filter(n):
    return np.sin(n)/(n)



#Esercizio 2

x_spostato = np.subtract(ore, (len(ore)-1)/2)
h_x = np.sinc(x_spostato)

N = len(pressioni)
n = np.arange(N)
n_centered = n - (N - 1) / 2

B = 0.1

h_x_reale = np.sinc(B * n_centered)
h_x_reale = h_x_reale / np.sum(h_x_reale)

y_n = np.convolve(pressioni, h_x_reale, mode="same")



#Esercizio 2 punto b
#autocorrelazione segnali x e y
x_corr = int(np.correlate(pressioni, pressioni))
y_corr = int(np.correlate(y_n, y_n))


###grafici in una sola finestra###

#finestra unica
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Grafico 1
axs[0, 0].plot(ore, pressioni, label='funzione', color='#1f77b4', linewidth=1)
axs[0, 0].axhline(valore_medio(pressioni), linestyle='--', color='darkred', label=f'Valore Medio: {valore_medio(pressioni):.2f}')
#axs[0, 0].axhline(energia(pressioni), linestyle='--', color='darkred', label=f'Energia: {energia(pressioni):.2f}')
axs[0, 0].set_xlim(min(ore), max(ore))
axs[0, 0].set_ylim(min(pressioni), max(pressioni))
axs[0, 0].set_xlabel("Orario")
axs[0, 0].set_ylabel("Pressioni")
axs[0, 0].set_title("Esercizio 1: Segnale Pressioni")
axs[0, 0].grid(True, linestyle=':', alpha=0.5)
axs[0, 0].legend()


# Esercizio 2a
axs[0, 1].plot(x_spostato, y_n, label='segnale filtrato y_n', color="#060002", linewidth=1)
axs[0, 1].set_ylim(31,38)
axs[0, 1].set_title("Esercizio 2a: Segnale filtrato (y_n) e segnale originale")
axs[0, 1].set_xlabel("Campioni (spostati)")
axs[0, 1].set_ylabel("Ampiezza")
axs[0, 1].grid(True, linestyle=':', alpha=0.5)
axs[0, 1].legend()

axs[0, 1].plot(x_spostato, pressioni, label='funzione originale', color="#17C2D2", linewidth=0.5)
axs[0, 1].set_xlabel("Campioni (spostati)")
axs[0, 1].set_ylabel("Pressione")
axs[0, 1].grid(True, linestyle=':', alpha=0.5)
axs[0, 1].legend()

axs[1, 0].plot(h_x_reale, label='filtro sinc', color="#FF5733", linewidth=1)
axs[1,0].set_title("Filtro Sinc Applicato")
axs[1,0].set_xlabel("Campioni")
axs[1,0].set_ylabel("Ampiezza")
axs[1,0].grid(True, linestyle=':', alpha=0.5)
axs[1,0].legend()

#es2b: Autocorrelazione X
axs[1, 1].axhline(x_corr, linestyle='--', color='darkred', label=f'autocorrelazione x: {x_corr:.2f}')
axs[1, 1].set_title("Autocorrelazione pressioni (Energia)")
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()
axs[1, 1].set_yticks([]) 


#es2b: Autocorrelazione Y
axs[2, 0].axhline(y_corr, linestyle='--', color='darkred', label=f'autocorrelazione y: {y_corr:.2f}')
axs[2, 0].set_title("Autocorrelazione filtro (Energia)")
axs[2, 0].grid(True, alpha=0.3)
axs[2, 0].legend()
axs[2, 0].set_yticks([])


#axs[2, 1].axis('off')

# Spaziatura e visualizzazione
plt.tight_layout(pad = 3.0)
plt.show()