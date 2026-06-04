import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from ProGP import ProGpMp 

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
num_demos = 9           
num_via_points = 3      
font_size = 18
observation_noise = 0.5 

# --- ALINEACIÓN TEMPORAL ---
# True: Estira/encoge las demos para que duren lo mismo (Time Warping lineal).
# False: Respeta la velocidad original (solo re-muestrea puntos).
ALIGN_TEMPORAL = True   

# --- GAP (DOWNSAMPLING) ---
# Reduce la cantidad de puntos. 
# avg_len final = (puntos dibujados promedio) / gap
# RECOMENDADO: Ajustar gap para que el resultado final esté entre 50 y 80 puntos.
# Si gap es muy bajo (ej. 1) y dibujas lento, tendrás demasiados puntos y fallará la matriz.
gap = 8

# -----------------------------------------------------------------------------
# 2. CLASES PARA INTERACCIÓN CON EL RATÓN
# -----------------------------------------------------------------------------

class DrawDemos:
    def __init__(self, n_demos):
        self.n_demos = n_demos
        self.demos = [] 
        self.current_demo = {'t': [], 'x': [], 'y': []}
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title(f"Dibuja {n_demos} demostraciones (Arrastra el ratón)")
        self.ax.set_xlim(-50, 50) 
        self.ax.set_ylim(-50, 50)
        self.ax.grid(True)
        self.drawing = False
        self.count = 0

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if self.count >= self.n_demos: return
        self.drawing = True
        self.current_demo = {'t': [], 'x': [], 'y': []}
        self.current_demo['x'].append(event.xdata)
        self.current_demo['y'].append(event.ydata)
        self.current_demo['t'].append(0.0)
        
        self.line, = self.ax.plot([], [], 'b-', lw=2)
        self.ax.plot(event.xdata, event.ydata, 'go')
        self.fig.canvas.draw()

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax: return
        dt_sim = 0.05 
        last_t = self.current_demo['t'][-1]
        self.current_demo['x'].append(event.xdata)
        self.current_demo['y'].append(event.ydata)
        self.current_demo['t'].append(last_t + dt_sim)

        self.line.set_data(self.current_demo['x'], self.current_demo['y'])
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.drawing:
            self.drawing = False
            self.demos.append(self.current_demo)
            self.count += 1
            self.ax.set_title(f"Demostración {self.count}/{self.n_demos} guardada")
            if self.count >= self.n_demos:
                self.ax.set_title("Demostraciones completadas. Cierra la ventana.")
                plt.pause(0.5)
                plt.close(self.fig)

class PickViaPoints:
    def __init__(self, demos_data, n_points):
        self.demos = demos_data
        self.n_points = n_points
        self.picked_points = [] 
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title(f"Haz clic para seleccionar {n_points} Via-Points")
        self.ax.set_xlim(-50, 50) 
        self.ax.set_ylim(-50, 50)
        self.count = 0
        
        for d in self.demos:
            self.ax.plot(d['x'], d['y'], 'b-', alpha=0.3)
            
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)

    def get_estimated_time(self, x, y):
        best_dist = float('inf')
        estimated_t = 0.0
        for d in self.demos:
            trajectory = np.column_stack((d['x'], d['y']))
            query = np.array([[x, y]])
            dists = cdist(query, trajectory)
            min_idx = np.argmin(dists)
            dist = dists[0, min_idx]
            if dist < best_dist:
                best_dist = dist
                estimated_t = d['t'][min_idx]
        return estimated_t

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if self.count >= self.n_points: return
        x, y = event.xdata, event.ydata
        t = self.get_estimated_time(x, y)
        self.picked_points.append([t, x, y])
        self.ax.plot(x, y, 'rx', markersize=12, markeredgewidth=2)
        self.ax.text(x, y, f"t={t:.2f}", color='red')
        self.fig.canvas.draw()
        self.count += 1
        self.ax.set_title(f"Via-Point {self.count}/{self.n_points} seleccionado")
        if self.count >= self.n_points:
            self.ax.set_title("Puntos seleccionados. Cierra la ventana.")
            plt.pause(0.5)
            plt.close(self.fig)

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

print("1. Dibuja las demostraciones en la ventana emergente.")
drawer = DrawDemos(num_demos)
plt.show()

demos_raw = drawer.demos
if not demos_raw:
    print("No se dibujaron demostraciones.")
    exit()

# --- PROCESAMIENTO: Gap & Alineación ---

# 1. Calcular longitud original promedio
lengths = [len(d['t']) for d in demos_raw]
mean_raw_len = int(np.mean(lengths))

# 2. Aplicar GAP para determinar la longitud final (avg_len)
# Esto define cuántos puntos tendrá cada demo interpolada.
avg_len = int(mean_raw_len / gap)

# Protección mínima (si gap es muy grande y quedan 2 puntos, interpolar falla)
if avg_len < 10: 
    avg_len = 10
    print("Aviso: Gap demasiado alto, forzando mínimo de 10 puntos.")

print(f"Longitud original media: {mean_raw_len} puntos.")
print(f"Aplicando Gap {gap} -> Nueva longitud fija: {avg_len} puntos.")

# 3. Calcular duración promedio (para alineación temporal)
durations = [d['t'][-1] for d in demos_raw]
avg_duration = np.mean(durations)

if ALIGN_TEMPORAL:
    print(f"ALINEACIÓN ACTIVADA: Todas las demos durarán {avg_duration:.2f}s")
else:
    print("ALINEACIÓN DESACTIVADA: Se mantienen los tiempos originales.")

X = None
Y = None
all_end_t = []
interpolated_demos = [] 

for i, d in enumerate(demos_raw):
    t_orig = np.array(d['t'])
    x_orig = np.array(d['x'])
    y_orig = np.array(d['y'])

    # --- LÓGICA DE ALINEACIÓN + GAP ---
    # Usamos 'avg_len' (calculado con el gap) como el número de puntos destino
    if ALIGN_TEMPORAL:
        t_normalized_orig = t_orig / t_orig[-1] 
        t_new = np.linspace(0, avg_duration, avg_len) # <- Aquí se aplica el tamaño reducido
        t_normalized_dest = np.linspace(0, 1, avg_len)
        
        x_new = np.interp(t_normalized_dest, t_normalized_orig, x_orig)
        y_new = np.interp(t_normalized_dest, t_normalized_orig, y_orig)
        
    else:
        # Si no alineamos, usamos el tiempo final original, pero reducimos puntos según avg_len
        t_new = np.linspace(0, t_orig[-1], avg_len)
        x_new = np.interp(t_new, t_orig, x_orig)
        y_new = np.interp(t_new, t_orig, y_orig)

    interpolated_demos.append({'t': t_new, 'x': x_new, 'y': y_new})

    # Preparar ProGP
    t_col = t_new.reshape(-1, 1)
    jitter = np.random.normal(0, 1e-4, t_col.shape)
    t_col = t_col + jitter
    pos_col = np.column_stack((x_new, y_new))

    if i == 0:
        X = t_col
        Y = pos_col
    else:
        X = np.vstack((X, t_col))
        Y = np.vstack((Y, pos_col))
    
    all_end_t.append(t_new[-1])

avg_end_time = np.mean(all_end_t) 

# --- B) Seleccionar Via-Points ---
print("2. Selecciona los Via-Points.")
picker = PickViaPoints(interpolated_demos, num_via_points)
plt.show()

user_via_points = picker.picked_points 

# --- CONSTRUCCIÓN DE CONSTRAINTS ---
constraints_t = []
constraints_pos = []

user_via_points.sort(key=lambda p: p[0]) 

for p in user_via_points:
    constraints_t.append(p[0])
    constraints_pos.append(np.array([p[1], p[2]]))

if len(constraints_t) > 0:
    X_ = np.array(constraints_t).reshape(-1, 1)
    Y_ = np.array(constraints_pos)
else:
    print("Aviso: No se seleccionaron puntos, usando inicio por defecto.")
    X_ = np.array([0.0]).reshape(-1, 1)
    Y_ = np.array([interpolated_demos[0]['x'][0], interpolated_demos[0]['y'][0]]).reshape(1, 2)

# -----------------------------------------------------------------------------
# 4. ENTRENAMIENTO Y PREDICCIÓN (ProGP)
# -----------------------------------------------------------------------------
print("Entrenando GPMP...")

# Usamos el avg_len calculado con el Gap
gp_mp = ProGpMp(X, Y, X_, Y_, dim=2, demos=num_demos, size=avg_len, observation_noise=observation_noise)
gp_mp.BlendedGpMp(gp_mp.ProGP)

dt_pred = 0.05
test_x = np.arange(0.0, avg_end_time + 0.1, dt_pred)

mean_blended, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))

var_blended[0] = np.where(var_blended[0] < 0, 0, var_blended[0])
var_blended[1] = np.where(var_blended[1] < 0, 0, var_blended[1])

# -----------------------------------------------------------------------------
# 5. GRAFICAR RESULTADOS
# -----------------------------------------------------------------------------
plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)

# Panel 1: Trayectoria 2D (X vs Y)
plt1 = plt.subplot2grid((8, 16), (0, 0), rowspan=8, colspan=8)
plt1.scatter(Y_[:, 0], Y_[:, 1], s=400, c='green', marker='x', label='Via Points')
plt1.scatter(Y[:, 0], Y[:, 1], s=30, c='green', marker='o', alpha=0.1) 
plt1.plot(mean_blended[0], mean_blended[1], c='black', linewidth=4, label='$GMP$')
plt1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt1.tick_params(labelsize=font_size)
plt1.set_xlabel('$x$/mm\n(a)', fontsize=font_size)
plt1.set_ylabel('$y$/mm', fontsize=font_size)
plt1.axis('equal') 

# Panel 2: Tiempo vs X 
plt2 = plt.subplot2grid((8, 16), (0, 9), rowspan=3, colspan=8)
plt2.plot(test_x, mean_blended[0], c='red', linewidth=3, label='$x_{GMP}$')
plt2.fill_between(test_x, mean_blended[0] - 5 * np.sqrt(var_blended[0]), mean_blended[0] + 5 * np.sqrt(var_blended[0]), color='red', alpha=0.3)
plt2.scatter(X_[:, 0], Y_[:, 0], s=400, c='red', marker='x')
plt2.scatter(X[:, 0], Y[:, 0], s=15, c='red', marker='o', alpha=0.1)
plt2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt2.tick_params(labelsize=font_size)
plt2.set_xlabel('$time$(s)\n(b)', fontsize=font_size)
plt2.set_ylabel('$x$/mm', fontsize=font_size)

# Panel 3: Tiempo vs Y 
plt3 = plt.subplot2grid((8, 16), (4, 9), rowspan=3, colspan=8)
plt3.plot(test_x, mean_blended[1], c='blue', linewidth=3, label='$y_{GMP}$')
plt3.fill_between(test_x, mean_blended[1] - 5 * np.sqrt(var_blended[1]), mean_blended[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
plt3.scatter(X_[:, 0], Y_[:, 1], s=400, c='blue', marker='x')
plt3.scatter(X[:, 0], Y[:, 1], s=15, c='blue', marker='o', alpha=0.1)
plt3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt3.tick_params(labelsize=font_size)
plt3.set_xlabel('$time$(s)\n(c)', fontsize=font_size)
plt3.set_ylabel('$y$/mm', fontsize=font_size)

plt.show()