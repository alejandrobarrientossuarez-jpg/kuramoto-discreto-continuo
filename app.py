"""
======================================================================
 Sincronización en el modelo de Kuramoto discreto — App interactiva
 Acompaña la nota pedagógica sobre particiones cruzadas en K_N.
 Ejecutar:  streamlit run app.py
======================================================================
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend no-interactivo para servidor
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import itertools
import pandas as pd

# ----------------------------------------------------------------------
#  CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Kuramoto discreto · Simulador",
    page_icon="🔄",
    layout="wide",
)

# Paleta de colores para los osciladores (hasta N=6)
COLORS = ["#1f4a8c", "#0e6e56", "#ba7517", "#a01e1e", "#6a3d9a", "#188a99"]


# ----------------------------------------------------------------------
#  NÚCLEO MATEMÁTICO DEL MODELO  F_kappa^(0)  sobre  K_N
# ----------------------------------------------------------------------
def dM(a, b, M):
    """Diferencia angular con signo en Z_M (camino más corto de b hacia a)."""
    return ((a - b + M // 2) % M) - M // 2


def sigma_kappa(S, kappa):
    """Función signo umbralizada."""
    if S >= kappa:
        return 1
    if S <= -kappa:
        return -1
    return 0


def step(theta, M, kappa):
    """
    Un paso de la dinámica sobre el grafo completo K_N.
    Devuelve la nueva configuración y el detalle de cada oscilador.
    """
    N = len(theta)
    new = []
    info = []
    for i in range(N):
        contribs = [(j, dM(theta[j], theta[i], M)) for j in range(N) if j != i]
        S = sum(c for _, c in contribs)
        sig = sigma_kappa(S, kappa)
        new.append((theta[i] + sig) % M)
        info.append({"i": i, "S": S, "sigma": sig, "contribs": contribs,
                     "old": theta[i], "new": (theta[i] + sig) % M})
    return new, info


def induced_partition(theta):
    """Partición inducida: agrupa los índices que comparten fase."""
    groups = {}
    for i, v in enumerate(theta):
        groups.setdefault(v, []).append(i)
    return sorted([sorted(g) for g in groups.values()], key=lambda g: g[0])


def partition_str(theta):
    """Representación legible de la partición, p. ej. {1,3}{2}."""
    part = induced_partition(theta)
    return "".join("{" + ",".join(str(i + 1) for i in block) + "}" for block in part)


def is_crossing(theta):
    """¿La partición inducida es cruzada en el orden lineal 1<2<...<N?

    Una partición es cruzada si existen índices a<c<b tales que a,b están
    en un mismo bloque y c en un bloque distinto (c queda "atrapado" entre
    a y b pero pertenece a otro grupo). Esto cubre el caso del singleton
    intercalado, como {1,3}{2}.
    """
    part = induced_partition(theta)
    block_of = {}
    for bi, block in enumerate(part):
        for x in block:
            block_of[x] = bi
    for block in part:
        for a, b in itertools.combinations(block, 2):
            for c in range(a + 1, b):
                if block_of.get(c) != block_of[a]:
                    return True
    return False


def compute_orbit(theta0, M, kappa, max_steps=60):
    """
    Itera la dinámica hasta detectar punto fijo, ciclo o sincronización.
    Devuelve: trayectoria, tipo, y datos del ciclo si aplica.
    """
    traj = [list(theta0)]
    seen = {tuple(theta0): 0}
    cur = list(theta0)

    for t in range(max_steps):
        nxt, _ = step(cur, M, kappa)

        # ¿Sincronización completa?
        if len(set(nxt)) == 1:
            traj.append(nxt)
            return traj, "sincroniza", None

        # ¿Punto fijo?
        if nxt == cur:
            return traj, "punto_fijo", None

        key = tuple(nxt)
        if key in seen:
            # Ciclo detectado
            cycle_start = seen[key]
            period = (t + 1) - cycle_start
            traj.append(nxt)
            return traj, "ciclo", {"start": cycle_start, "period": period}

        seen[key] = t + 1
        traj.append(nxt)
        cur = nxt

    return traj, "sin_converger", None


# ----------------------------------------------------------------------
#  RETÍCULO DE PARTICIONES (estructura combinatoria)
# ----------------------------------------------------------------------
def all_partitions(n):
    """Genera todas las particiones del conjunto {0,1,...,n-1}."""
    def _gen(collection):
        if len(collection) == 1:
            yield [collection]
            return
        first = collection[0]
        for smaller in _gen(collection[1:]):
            for i, subset in enumerate(smaller):
                yield smaller[:i] + [[first] + subset] + smaller[i + 1:]
            yield [[first]] + smaller
    return list(_gen(list(range(n))))


def normalize_partition(part):
    """Ordena bloques por (tamaño, primer elemento) para comparación estable."""
    return tuple(sorted((tuple(sorted(b)) for b in part), key=lambda b: (len(b), b[0])))


def partition_is_crossing(part):
    """¿La partición (lista de bloques) es cruzada en el orden 1<2<...<N?"""
    block_of = {}
    for bi, block in enumerate(part):
        for x in block:
            block_of[x] = bi
    for block in part:
        sb = sorted(block)
        for a, b in itertools.combinations(sb, 2):
            for c in range(a + 1, b):
                if block_of.get(c) != block_of[a]:
                    return True
    return False


def partition_label(part):
    """Etiqueta legible: {1,3}{2}."""
    pn = sorted((sorted(b) for b in part), key=lambda b: b[0])
    return "".join("{" + ",".join(str(i + 1) for i in block) + "}" for block in pn)


def partition_rank(part):
    """Número de bloques (para ubicar en niveles del retículo)."""
    return len(part)


def bell_number(n):
    """Número de Bell B_n: todas las particiones de un conjunto de n elementos."""
    row = [1]
    for _ in range(n):
        new = [row[-1]]
        for x in row:
            new.append(new[-1] + x)
        row = new
    return row[0]


def catalan_number(n):
    """Número de Catalan C_n: particiones no cruzadas (sentido de Kreweras)."""
    from math import comb
    return comb(2 * n, n) // (n + 1)


# ----------------------------------------------------------------------
#  VISUALIZACIÓN: CÍRCULO DISCRETO
# ----------------------------------------------------------------------
def draw_circle(theta, M, ax, title=""):
    """Dibuja los M sitios del círculo discreto y coloca los osciladores."""
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    R = 1.0
    # Sitios del círculo (marcas)
    angles = [np.pi / 2 - 2 * np.pi * k / M for k in range(M)]
    site_x = [R * np.cos(a) for a in angles]
    site_y = [R * np.sin(a) for a in angles]

    # Círculo guía
    circ = plt.Circle((0, 0), R, fill=False, color="#cccccc",
                      linewidth=1, linestyle="--", zorder=1)
    ax.add_patch(circ)

    # Marcas y etiquetas de posición
    for k in range(M):
        ax.plot(site_x[k], site_y[k], "o", color="#dddddd",
                markersize=14, zorder=2)
        ax.text(1.28 * np.cos(angles[k]), 1.28 * np.sin(angles[k]),
                str(k), ha="center", va="center",
                fontsize=10, color="#888888")

    # Agrupar osciladores por sitio (para apilar si coinciden)
    by_site = {}
    for i, ph in enumerate(theta):
        by_site.setdefault(ph, []).append(i)

    # Dibujar osciladores
    for ph, oscs in by_site.items():
        n = len(oscs)
        for idx, i in enumerate(oscs):
            # Si varios coinciden, los separamos ligeramente en radio
            rr = R if n == 1 else R * (1 + 0.13 * (idx - (n - 1) / 2))
            x = rr * np.cos(angles[ph])
            y = rr * np.sin(angles[ph])
            ax.plot(x, y, "o", color=COLORS[i % len(COLORS)],
                    markersize=24, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.text(x, y, str(i + 1), ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold", zorder=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)


def draw_network(theta, ax, title=""):
    """Dibuja la subred sincronizada: nodos en círculo, aristas si comparten fase."""
    N = len(theta)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    angles = [np.pi / 2 - 2 * np.pi * k / N for k in range(N)]
    pos = [(np.cos(a), np.sin(a)) for a in angles]

    # Aristas: conectar i,j si theta_i == theta_j
    for i in range(N):
        for j in range(i + 1, N):
            if theta[i] == theta[j]:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color="#333333", linewidth=2.5, zorder=1)

    # Nodos
    for i in range(N):
        ax.plot(pos[i][0], pos[i][1], "o", color=COLORS[i % len(COLORS)],
                markersize=30, zorder=3,
                markeredgecolor="white", markeredgewidth=2)
        ax.text(pos[i][0], pos[i][1], str(i + 1), ha="center", va="center",
                fontsize=13, color="white", fontweight="bold", zorder=4)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)


# ----------------------------------------------------------------------
#  VISUALIZACIÓN: RETÍCULO DE PARTICIONES
# ----------------------------------------------------------------------
def draw_lattice(N, ax, visited_labels=None, current_label=None,
                 mode="discrete"):
    """
    Dibuja el retículo de particiones de {1..N} organizado por niveles
    (número de bloques). Colorea las particiones cruzadas y, opcionalmente,
    resalta las visitadas por una trayectoria.

    mode='continuous': las cruzadas se muestran tachadas (vedadas).
    mode='discrete':   las cruzadas se muestran disponibles.
    """
    visited_labels = visited_labels or set()
    parts = all_partitions(N)

    # Agrupar por nivel (número de bloques)
    levels = {}
    for p in parts:
        r = len(p)
        levels.setdefault(r, []).append(p)

    max_rank = N        # nivel del ínfimo (todos separados)
    min_rank = 1        # nivel del supremo (todos juntos)

    # Separación horizontal mínima entre nodos vecinos (en unidades de datos).
    # El ancho de cada nivel se calcula para garantizar esta separación,
    # de modo que las etiquetas no se empalmen aunque el nivel esté muy poblado.
    DX = 2.4
    # Cuántos nodos tiene el nivel más poblado (para fijar el ancho total)
    max_per_level = max(len(plist) for plist in levels.values())
    half_width = (max_per_level - 1) * DX / 2.0

    # Asignar coordenadas: y por nivel, x repartido en cada nivel
    pos = {}
    info = {}
    for r, plist in levels.items():
        # ordenar para estabilidad visual
        plist_sorted = sorted(plist, key=lambda p: partition_label(p))
        k = len(plist_sorted)
        for idx, p in enumerate(plist_sorted):
            # Centrar cada nivel y separar los nodos por DX exactos
            x = (idx - (k - 1) / 2) * DX
            # y: ínfimo arriba (rank=N), supremo abajo (rank=1).
            # Más separación vertical entre niveles para que respire.
            y = (r - min_rank) / max(max_rank - min_rank, 1) * 5.0
            lbl = partition_label(p)
            pos[lbl] = (x, y)
            info[lbl] = {
                "crossing": partition_is_crossing(p),
                "rank": r,
                "part": p,
            }

    # Dibujar aristas de cobertura (relación de refinamiento entre niveles
    # adyacentes): conectar P (nivel r) con P' (nivel r-1) si P' se obtiene
    # uniendo exactamente dos bloques de P.
    def covers(p_fine, p_coarse):
        # p_coarse cubre a p_fine si une exactamente dos bloques
        if len(p_coarse) != len(p_fine) - 1:
            return False
        set_fine = [set(b) for b in p_fine]
        set_coarse = [set(b) for b in p_coarse]
        # cada bloque coarse es unión de bloques fine
        for cb in set_coarse:
            covered = [fb for fb in set_fine if fb <= cb]
            if set().union(*covered) != cb:
                return False
        return True

    labels_by_rank = {r: [partition_label(p) for p in plist]
                      for r, plist in levels.items()}
    for r in range(max_rank, min_rank, -1):
        for lf in labels_by_rank.get(r, []):
            pf = info[lf]["part"]
            for lc in labels_by_rank.get(r - 1, []):
                pc = info[lc]["part"]
                if covers(pf, pc):
                    x1, y1 = pos[lf]
                    x2, y2 = pos[lc]
                    ax.plot([x1, x2], [y1, y2], "-",
                            color="#dddddd", linewidth=0.8, zorder=1)

    # Dibujar nodos
    for lbl, (x, y) in pos.items():
        crossing = info[lbl]["crossing"]
        visited = lbl in visited_labels
        is_current = (lbl == current_label)

        # Estilo según tipo
        if crossing:
            if mode == "continuous":
                # vedada: contorno rojo tenue, relleno hueco
                face = "#ffffff"
                edge = "#d9a0a0"
                ax.plot(x, y, "o", markersize=15, markerfacecolor=face,
                        markeredgecolor=edge, markeredgewidth=1.5, zorder=3)
                # tachado
                ax.plot([x - 0.12, x + 0.12], [y - 0.12, y + 0.12],
                        color="#c0392b", linewidth=1.3, zorder=4)
                ax.plot([x - 0.12, x + 0.12], [y + 0.12, y - 0.12],
                        color="#c0392b", linewidth=1.3, zorder=4)
            else:
                # discreta cruzada: naranja
                face = "#ba7517" if visited else "#f3d9b3"
                edge = "#7a4d0e"
                ax.plot(x, y, "o", markersize=16, markerfacecolor=face,
                        markeredgecolor=edge, markeredgewidth=1.6, zorder=3)
        else:
            # no cruzada
            face = "#1f4a8c" if visited else "#cdd8e8"
            edge = "#13315c"
            ax.plot(x, y, "o", markersize=15, markerfacecolor=face,
                    markeredgecolor=edge, markeredgewidth=1.3, zorder=3)

        # Resaltar el nodo actual con un anillo
        if is_current:
            ax.plot(x, y, "o", markersize=24, markerfacecolor="none",
                    markeredgecolor="#0e6e56", markeredgewidth=2.5, zorder=5)

        # Etiqueta (rotada si el nivel está muy poblado, para no empalmar)
        fs = 8 if N <= 4 else 7
        rot = 0 if max_per_level <= 6 else 30
        ax.text(x, y - 0.38, lbl, ha="center", va="top",
                fontsize=fs, color="#333333", zorder=6,
                rotation=rot, rotation_mode="anchor")

    # Marcas de ínfimo / supremo
    ax.text(0, 5.65, "ínfimo (todos separados)", ha="center",
            fontsize=9, style="italic", color="#888888")
    ax.text(0, -1.05, "supremo (todos sincronizados)", ha="center",
            fontsize=9, style="italic", color="#888888")

    margin = half_width + DX * 0.6
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-1.5, 6.0)
    ax.axis("off")
st.sidebar.title("⚙️ Parámetros")

N = st.sidebar.slider("Número de osciladores  N", 3, 6, 3,
                      help="Grafo completo K_N")
M = st.sidebar.slider("Tamaño del alfabeto  M", N, 12, max(4, N),
                      help="Posiciones del reloj discreto Z_M")
kappa = st.sidebar.slider("Umbral de acoplamiento  κ", 1, 2 * M, 2,
                          help="Sensibilidad de la corrección")

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración inicial  θ⁰")

# Presets útiles
preset = st.sidebar.selectbox(
    "Plantillas rápidas",
    ["Personalizada",
     "Ejemplo de la nota (K₃)",
     "P× punto fijo (K₄)",
     "P× sincroniza (K₄)",
     "P× ciclo antípoda (K₄)"],
)

# Determinar valores iniciales según preset
if preset == "Ejemplo de la nota (K₃)" and N == 3 and M >= 4:
    default = [0, 2, 3]
elif preset == "P× punto fijo (K₄)" and N == 4 and M >= 6:
    default = [0, 1, 0, 1]
elif preset == "P× sincroniza (K₄)" and N == 4 and M >= 6:
    default = [0, 2, 0, 2]
elif preset == "P× ciclo antípoda (K₄)" and N == 4 and M >= 6:
    default = [0, M // 2, 0, M // 2]
else:
    default = [(2 * i) % M for i in range(N)]

# Sliders individuales para cada fase
theta0 = []
for i in range(N):
    val = default[i] if i < len(default) else 0
    val = min(val, M - 1)
    theta0.append(
        st.sidebar.number_input(
            f"θ⁰[{i + 1}]", min_value=0, max_value=M - 1, value=val, step=1,
            key=f"theta_{i}_{N}_{M}",
        )
    )

# Aviso si el preset no encaja con N, M actuales
if preset != "Personalizada":
    needs = {"Ejemplo de la nota (K₃)": (3, 4),
             "P× punto fijo (K₄)": (4, 6),
             "P× sincroniza (K₄)": (4, 6),
             "P× ciclo antípoda (K₄)": (4, 6)}
    rn, rm = needs[preset]
    if N != rn or M < rm:
        st.sidebar.warning(
            f"Esta plantilla está pensada para N={rn}, M≥{rm}. "
            f"Ajusta los deslizadores de arriba."
        )

# ---- Botón para ejecutar la simulación ----
st.sidebar.markdown("---")
run_clicked = st.sidebar.button(
    "▶️  SIMULAR",
    type="primary",
    use_container_width=True,
)

# Si se presiona, calcular la órbita y guardarla en el estado de sesión.
# Esto hace que los resultados persistan al mover el deslizador de pasos
# o cambiar de pestaña, sin recalcular en cada interacción.
if run_clicked:
    traj_run, kind_run, cyc_run = compute_orbit(theta0, M, kappa)
    st.session_state["sim"] = {
        "traj": traj_run,
        "kind": kind_run,
        "cycle_info": cyc_run,
        "N": N,
        "M": M,
        "kappa": kappa,
        "theta0": list(theta0),
    }

# Aviso si los parámetros cambiaron respecto a la última simulación
sim = st.session_state.get("sim")
if sim is not None:
    params_changed = (
        sim["N"] != N or sim["M"] != M or sim["kappa"] != kappa
        or sim["theta0"] != list(theta0)
    )
    if params_changed:
        st.sidebar.info(
            "Cambiaste algún parámetro. Pulsa **SIMULAR** "
            "de nuevo para actualizar los resultados."
        )


# ----------------------------------------------------------------------
#  CABECERA
# ----------------------------------------------------------------------
st.title("🔄 Sincronización en el modelo de Kuramoto discreto")
st.markdown(
    "Simulador interactivo de la dinámica "
    r"$F_\kappa^{(0)}$ sobre el grafo completo $K_N$. "
    "Explora cómo los osciladores sincronizan paso a paso y descubre las "
    "**particiones cruzadas** que el modelo continuo no permite."
)

# Fórmula del modelo
with st.expander("📐 ¿Qué calcula el modelo?", expanded=False):
    st.markdown("Cada oscilador actualiza su fase según:")
    st.latex(r"\theta_i' = \theta_i + \sigma_\kappa\!\Big(\sum_{j \neq i} "
             r"d_M(\theta_j, \theta_i)\Big) \pmod{M}")
    st.markdown(
        r"""
        donde:
        - $d_M(a,b) = \big((a-b+\lfloor M/2\rfloor)\bmod M\big) - \lfloor M/2\rfloor$
          es la **diferencia angular con signo** (camino más corto de $b$ a $a$).
        - $\sigma_\kappa(S)$ vale $+1$ si $S\geq\kappa$, $-1$ si $S\leq-\kappa$, y $0$ si $|S|<\kappa$.

        La corrección es siempre $\pm 1$ o $0$: un **salto entero**, no un
        desplazamiento proporcional. Por eso las fases pueden cruzarse
        algebraicamente sin obstáculo geométrico.
        """
    )

# ----------------------------------------------------------------------
#  ¿HAY UNA SIMULACIÓN LISTA?
# ----------------------------------------------------------------------
sim = st.session_state.get("sim")

if sim is None:
    # Aún no se ha pulsado el botón: mostrar pantalla de bienvenida y parar.
    st.info(
        "👈  Ajusta los parámetros en la barra lateral y pulsa "
        "**▶️ SIMULAR** para ejecutar la dinámica."
    )
    st.markdown(
        """
        #### ¿Qué vas a ver al simular?
        - **🎨 Vista visual** — los osciladores moviéndose sobre el círculo
          discreto y la subred que van formando al sincronizar.
        - **🔬 Vista técnica** — el detalle numérico de cada paso (sumas de
          acoplamiento, correcciones) y la trayectoria completa.
        - **🔷 Retículo y comparación** — el mapa de todas las particiones,
          con las que el modelo continuo tiene **prohibidas** frente a las que
          el discreto **sí alcanza**.

        Prueba las **plantillas rápidas** de la barra lateral para cargar los
        ejemplos de la nota con un clic.
        """
    )
    st.stop()

# Recuperar los datos de la simulación guardada
traj = sim["traj"]
kind = sim["kind"]
cycle_info = sim["cycle_info"]
# Usar los parámetros con los que se simuló (no los actuales de los sliders)
M = sim["M"]
N = sim["N"]
kappa = sim["kappa"]

# Banner de resultado
labels = {
    "sincroniza": ("✅ El sistema SINCRONIZA", "success"),
    "punto_fijo": ("⏸️ El sistema queda en un PUNTO FIJO", "warning"),
    "ciclo": ("🔁 El sistema entra en un CICLO", "warning"),
    "sin_converger": ("❓ Sin converger en el límite de pasos", "info"),
}
msg, level = labels[kind]
if cycle_info:
    msg += f"  ·  período = {cycle_info['period']}"
getattr(st, level)(f"**{msg}**  ·  longitud de trayectoria: {len(traj) - 1} pasos")


# ----------------------------------------------------------------------
#  PESTAÑAS
# ----------------------------------------------------------------------
tab_visual, tab_tecnica, tab_reticulo = st.tabs(
    ["🎨 Vista visual", "🔬 Vista técnica", "🔷 Retículo y comparación"]
)

# Estado del paso actual (control deslizante compartido)
n_steps = len(traj) - 1
step_idx = st.slider("Paso de la simulación  t", 0, n_steps, 0,
                     help="Desliza para avanzar la dinámica")

theta_now = traj[step_idx]

# ======================================================================
#  PESTAÑA VISUAL
# ======================================================================
with tab_visual:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Círculo discreto · t = {step_idx}")
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        draw_circle(theta_now, M, ax1)
        st.pyplot(fig1)
        plt.close(fig1)

        # Estado textual
        st.markdown(
            f"**θ({step_idx})** = ({', '.join(str(v) for v in theta_now)})"
        )
        cross = is_crossing(theta_now)
        part = partition_str(theta_now)
        if cross:
            st.markdown(f"Partición: **{part}** &nbsp; 🔶 *cruzada*")
        else:
            st.markdown(f"Partición: **{part}** &nbsp; (no cruzada)")

    with col2:
        st.subheader(f"Subred sincronizada · t = {step_idx}")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        draw_network(theta_now, ax2)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown(
            "Dos osciladores se conectan cuando comparten exactamente la "
            "misma fase. Cuando el grafo está completo, hay sincronización total."
        )

    # Línea de tiempo de particiones
    st.markdown("---")
    st.subheader("📊 Evolución de las particiones")
    timeline = " → ".join(
        f"**[{partition_str(traj[t])}]**" if t == step_idx else f"{partition_str(traj[t])}"
        for t in range(len(traj))
    )
    st.markdown(timeline)


# ======================================================================
#  PESTAÑA TÉCNICA
# ======================================================================
with tab_tecnica:
    st.subheader(f"Detalle del paso  t = {step_idx} → {step_idx + 1}")

    if step_idx < n_steps:
        _, info = step(theta_now, M, kappa)

        # Tabla de cálculo por oscilador
        rows = []
        for d in info:
            i = d["i"]
            contrib_str = " + ".join(
                f"d({theta_now[j]},{theta_now[i]})={c:+d}"
                for j, c in d["contribs"]
            )
            rows.append({
                "Oscilador": f"{i + 1}",
                "Fase θ": theta_now[i],
                "Vecinos (cálculo dₘ)": contrib_str,
                "Sₐ": d["S"],
                "σκ(S)": f"{d['sigma']:+d}",
                "Nueva fase": d["new"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        st.caption(
            "Sₐ es la suma de acoplamiento del oscilador con sus vecinos. "
            "La corrección σκ(S) se suma a la fase (módulo M)."
        )
    else:
        st.info("Último paso de la trayectoria — no hay transición siguiente.")

    # Tabla de la trayectoria completa
    st.markdown("---")
    st.subheader("🧭 Trayectoria completa")
    traj_rows = []
    for t, th in enumerate(traj):
        traj_rows.append({
            "t": t,
            "Configuración θ": "(" + ", ".join(str(v) for v in th) + ")",
            "Partición": partition_str(th),
            "¿Cruzada?": "🔶 sí" if is_crossing(th) else "—",
            "# bloques": len(induced_partition(th)),
        })
    df_traj = pd.DataFrame(traj_rows)

    # Resaltar el paso actual
    def highlight_row(row):
        return ["background-color: #fff3cd" if row["t"] == step_idx else ""
                for _ in row]

    st.dataframe(
        df_traj.style.apply(highlight_row, axis=1),
        width="stretch", hide_index=True,
    )

    # Diagnóstico del comportamiento
    st.markdown("---")
    st.subheader("🔎 Diagnóstico")
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    diag_col1.metric("Comportamiento", {
        "sincroniza": "Sincroniza",
        "punto_fijo": "Punto fijo",
        "ciclo": "Ciclo",
        "sin_converger": "Sin converger",
    }[kind])
    diag_col2.metric("Longitud", f"{len(traj) - 1} pasos")
    if cycle_info:
        diag_col3.metric("Período del ciclo", cycle_info["period"])
    else:
        n_cross = sum(1 for th in traj if is_crossing(th))
        diag_col3.metric("Estados cruzados", n_cross)

    # Nota sobre P_x si aplica
    if N == 4 and any(is_crossing(th) for th in traj):
        st.info(
            "🔶 **Partición cruzada detectada.** Para N=4, la partición "
            r"$P_\times=\{\{1,3\},\{2,4\}\}$ es la región dinámicamente "
            "aislada de la nota: desde ahí el sistema solo puede quedar fijo, "
            "saltar a la sincronización, o ciclar (caso antípoda |a−b|=M/2). "
            "Ninguna de estas tres dinámicas existe en el modelo continuo."
        )


# ======================================================================
#  PESTAÑA RETÍCULO Y COMPARACIÓN
# ======================================================================
with tab_reticulo:
    st.subheader("El retículo de particiones de los osciladores")
    st.markdown(
        "Cada nodo es una forma posible de agrupar los osciladores según su "
        "fase. El **ínfimo** (arriba) es el estado inicial con todos separados; "
        "el **supremo** (abajo) es la sincronización total. La trayectoria de "
        "tu simulación recorre este retículo de arriba hacia abajo."
    )

    # Etiquetas visitadas por la trayectoria actual
    visited = set(partition_label(induced_partition(th)) for th in traj)
    current_lbl = partition_label(induced_partition(theta_now))

    # ---- Leyenda ----
    leg_c1, leg_c2, leg_c3, leg_c4 = st.columns(4)
    leg_c1.markdown("🔵 **Azul** — no cruzada")
    leg_c2.markdown("🟠 **Naranja** — cruzada")
    leg_c3.markdown("🟢 **Anillo verde** — estado actual")
    leg_c4.markdown("**Relleno sólido** — visitada")

    st.markdown("---")

    # ---- PANEL DE CONTEOS COMBINATORIOS ----
    st.markdown("### 🔢 Los números detrás del retículo")
    st.markdown(
        f"Para **N = {N}** osciladores, tres sucesiones clásicas describen la "
        "estructura combinatoria. Cada una cuenta algo distinto:"
    )

    # Calcular los conteos reales
    _all_p = all_partitions(N)
    B = bell_number(N)
    Cat = catalan_number(N)
    n_cross_dyn = sum(1 for p in _all_p if partition_is_crossing(p))
    n_noncross_dyn = len(_all_p) - n_cross_dyn

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(
        f"Bell  B_{N}", B,
        help="Todas las particiones posibles. Es el tamaño total del "
             "retículo, común a ambos modelos.",
    )
    cc2.metric(
        f"Catalan  C_{N}", Cat,
        help="Particiones no cruzadas en el sentido clásico de Kreweras. "
             "Ligadas a los caminos de Dyck del modelo continuo.",
    )
    cc3.metric(
        "Cruzadas (dinámica)", n_cross_dyn,
        delta=f"{n_noncross_dyn} no cruzadas",
        delta_color="off",
        help="Particiones que el modelo continuo no puede alcanzar por "
             "preservación del orden, pero el discreto sí.",
    )

    with st.expander("📖 ¿Qué cuenta cada número y por qué importa?"):
        st.markdown(
            r"""
**Números de Bell — el universo completo.**
$B_N$ cuenta *todas* las particiones de los $N$ osciladores, sin
restricción. Es el tamaño total del retículo que ves dibujado. **No es
exclusivo del caso discreto**: tanto el retículo continuo como el discreto
tienen $B_N$ nodos. Lo que cambia entre modelos no es este número, sino
*qué fracción* del retículo es dinámicamente accesible.

**Números de Catalan — las particiones no cruzadas.**
$C_N$ cuenta las particiones no cruzadas en el sentido clásico. Son
**relevantes para el caso continuo**: como la preservación del orden
circular prohíbe que dos bloques se entrelacen, el continuo se restringe a
esta familia. Los caminos resultantes se ponen en biyección con **caminos
de Dyck**.

**Sucesión de Golomb — los caminos del continuo.**
Aquí está la distinción más fina: Catalan cuenta *nodos* (cuántas
particiones no cruzadas hay), mientras que la **sucesión de Golomb** cuenta
*caminos* (cuántas trayectorias distintas van del ínfimo al supremo en el
diagrama del continuo). No son lo mismo: una cosa es cuántas estaciones hay
en una red de metro, y otra cuántas rutas conectan dos estaciones.

**El caso discreto — sin fórmula cerrada.**
En el modelo discreto, las particiones cruzadas **sí son alcanzables**, así
que el conjunto accesible se acerca al Bell completo. Pero que un nodo sea
alcanzable no significa que el conteo de caminos siga una fórmula bonita:
como muestra la nota, el número de caminos **depende de los parámetros $M$
y $\kappa$** y no tiene fórmula cerrada general. Por eso aparecen valores
como 4, 18 o 270, que no corresponden a ninguna de estas sucesiones
clásicas.

---

| Objeto | Qué cuenta | Dónde aparece |
|---|---|---|
| **Bell** $B_N$ | Todas las particiones (nodos del retículo) | Común a ambos modelos |
| **Catalan** $C_N$ | Particiones no cruzadas (nodos accesibles en el continuo) | El conjunto que el continuo puede visitar |
| **Golomb** | Caminos del ínfimo al supremo en el continuo | Conteo de trayectorias del continuo |
| *(sin fórmula cerrada)* | Caminos en el discreto | Depende de $(M,\kappa)$ |

> **Nota sobre el conteo "Cruzadas (dinámica)".** En esta app marcamos como
cruzada toda partición en la que un bloque *encierra* a un elemento de otro
bloque (criterio de convexidad en el orden). Este criterio es ligeramente
más estricto que el de Kreweras y captura mejor qué es alcanzable bajo la
dinámica con preservación de orden; por eso el número de no cruzadas que
contamos aquí ($%d$) puede diferir de $C_N$ ($%d$).
            """ % (n_noncross_dyn, Cat)
        )

    # Mini-gráfico de barras Bell vs Catalan vs cruzadas
    fig_cnt, ax_cnt = plt.subplots(figsize=(7, 2.6))
    cats = ["Bell\n(todas)", "Catalan\n(no cruz. clás.)",
            "No cruzadas\n(dinámica)", "Cruzadas\n(dinámica)"]
    vals = [B, Cat, n_noncross_dyn, n_cross_dyn]
    cols = ["#888888", "#1f4a8c", "#3a7d3a", "#ba7517"]
    bars = ax_cnt.bar(cats, vals, color=cols, edgecolor="white", width=0.62)
    for b, v in zip(bars, vals):
        ax_cnt.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                    str(v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax_cnt.set_ylim(0, max(vals) * 1.18)
    ax_cnt.spines["top"].set_visible(False)
    ax_cnt.spines["right"].set_visible(False)
    ax_cnt.tick_params(labelsize=8)
    ax_cnt.set_ylabel("número de particiones", fontsize=9)
    st.pyplot(fig_cnt)
    plt.close(fig_cnt)

    st.markdown("---")

    # ---- COMPARACIÓN LADO A LADO ----
    st.markdown("### Comparación: ¿qué puede alcanzar cada modelo?")
    st.markdown(
        "**Arriba**, el modelo **continuo**: las particiones cruzadas están "
        "**tachadas en rojo** porque la preservación del orden circular de las "
        "fases las hace inalcanzables. **Abajo**, el modelo **discreto**: "
        "esas mismas particiones (en naranja) **sí son alcanzables**, porque la "
        "dinámica avanza por saltos enteros sin obstáculo geométrico."
    )

    if N <= 5:
        # Ancho de figura proporcional al nivel más poblado, para que los
        # nodos nunca se empalmen. N=3 es estrecho; N=5 necesita ser ancho.
        _parts = all_partitions(N)
        _lvl = {}
        for _p in _parts:
            _lvl[len(_p)] = _lvl.get(len(_p), 0) + 1
        max_lvl = max(_lvl.values())
        fig_w = max(7, min(22, 1.6 * max_lvl + 3))

        # --- CONTINUO (arriba, a todo el ancho) ---
        st.markdown("#### 🚫 Continuo — particiones cruzadas VEDADAS")
        figL, axL = plt.subplots(figsize=(fig_w, 5.2))
        draw_lattice(N, axL, visited_labels=set(), current_label=None,
                     mode="continuous")
        st.pyplot(figL)
        plt.close(figL)

        st.markdown("---")

        # --- DISCRETO (abajo, a todo el ancho) ---
        st.markdown("#### ✅ Discreto — particiones cruzadas ALCANZABLES")
        figR, axR = plt.subplots(figsize=(fig_w, 5.2))
        draw_lattice(N, axR, visited_labels=visited,
                     current_label=current_lbl, mode="discrete")
        st.pyplot(figR)
        plt.close(figR)
    else:
        st.warning(
            f"Para N={N} hay 203 particiones: el retículo completo sería "
            "ilegible. Se muestra en su lugar el resumen cuantitativo abajo "
            "y la lista de particiones cruzadas. Usa N≤5 para ver el retículo "
            "gráfico completo."
        )

    # ---- Resumen cuantitativo ----
    all_p = all_partitions(N)
    n_total = len(all_p)
    n_cross = sum(1 for p in all_p if partition_is_crossing(p))
    n_visited_cross = sum(
        1 for lbl in visited
        if any(partition_label(p) == lbl and partition_is_crossing(p)
               for p in all_p)
    )

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Particiones totales", n_total,
              help="Número de Bell B_N")
    m2.metric("Cruzadas (vedadas al continuo)", n_cross)
    m3.metric("No cruzadas", n_total - n_cross)
    m4.metric("Cruzadas en tu trayectoria", n_visited_cross,
              help="Cuántas particiones cruzadas visita la simulación actual")

    # ---- Mensaje contextual ----
    if n_visited_cross > 0:
        st.success(
            f"🎯 **Tu trayectoria atraviesa {n_visited_cross} partición(es) "
            f"cruzada(s)** — un camino que el modelo continuo no puede recorrer. "
            "Esas son justamente las particiones tachadas en rojo a la izquierda."
        )
    else:
        st.info(
            "Tu trayectoria actual no pasa por ninguna partición cruzada. "
            "Prueba la plantilla **«Ejemplo de la nota (K₃)»** o coloca dos "
            "osciladores no consecutivos en la misma fase para forzar un cruce."
        )

    # ---- Lista explícita de particiones cruzadas ----
    with st.expander(f"Ver las {n_cross} particiones cruzadas de K_{N}"):
        cross_list = sorted(
            (partition_label(p) for p in all_p if partition_is_crossing(p))
        )
        reached = [c for c in cross_list if c in visited]
        not_reached = [c for c in cross_list if c not in visited]
        st.markdown("**Visitadas por tu trayectoria:** "
                    + (", ".join(f"`{c}`" for c in reached) if reached else "ninguna"))
        st.markdown("**No visitadas (pero alcanzables en el discreto):** "
                    + (", ".join(f"`{c}`" for c in not_reached) if not_reached else "ninguna"))


# ----------------------------------------------------------------------
#  PIE
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Acompaña la nota pedagógica *Particiones cruzadas en el modelo de "
    "Kuramoto discreto*. La dinámica continua preserva el orden circular de "
    "las fases y prohíbe las particiones cruzadas; la discreta avanza por "
    "saltos enteros y las hace realizables."
)
