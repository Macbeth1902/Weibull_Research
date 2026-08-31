## EJECUTAR DESPUÉS DE prueba27.py (que genera output/all_results.pkl)

## FIGURAS PARA EL ARTÍCULO:
##   Fig. 1 — KDE con μ* (una por ciudad, tres periodos superpuestos)
##   Fig. 2 — Gráfico de papel de probabilidad Weibull (WPP)
##   Fig. 3 — FDA empírica vs FDA de mezcla ajustada
##   Fig. 4 — Comparación de k y μ* entre ciudades y periodos
##   Fig. 5 — Curvas de probabilidad de excedencia P(X > θ)
##   Fig. 6 — Evolución temporal de μ* y k por ciudad (tendencias)


import os, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

warnings.filterwarnings("ignore")

## CONFIGURACIÓN
PESO            = 'A'
OUTPUT_ART      = "output/figuras_articulo"
OUTPUT_ANX      = "output/figuras_anexo"
THETA_CURVAS    = np.linspace(20, 45, 300)
DPI             = 180

## Periodos
PERIODOS_ORD = ["1940-1968", "1969-1996", "1997-2026"]
COLORES = {
    "1940-1968": "#1f77b4",
    "1969-1996": "#ff7f0e",
    "1997-2026": "#2ca02c",
}
MARCADORES = {
    "1940-1968": "o",
    "1969-1996": "s",
    "1997-2026": "^",
}

os.makedirs(OUTPUT_ART, exist_ok=True)
os.makedirs(OUTPUT_ANX, exist_ok=True)

## CARGAR RESULTADOS
with open("output/all_results.pkl", "rb") as fh:
    ALL = pickle.load(fh)

ciudades = list(ALL.keys())
print(f"Ciudades: {ciudades}")
print(f"Periodos de ejemplo: {list(ALL[ciudades[0]].keys())}")


## HELPER: recuperar resultado de forma segura
def get_r(all_dict, ciudad, periodo, peso=PESO):
    try:
        return all_dict[ciudad][periodo][peso]
    except KeyError:
        return None


## FIG. 1
def fig1_kde_comparado(ciudad, all_dict):
    'Genera un gráfico de densidad KDE con μ* para cada periodo superpuesto.'

    periodos = [p for p in PERIODOS_ORD if get_r(all_dict, ciudad, p) is not None]
    if not periodos:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for periodo in periodos:
        r     = get_r(all_dict, ciudad, periodo)
        color = COLORES.get(periodo, '#666')
        ax.plot(r['x_grid'], r['densidad'], color=color, lw=2.0,
                label=f'{periodo}  μ*={r["mu_star"]:.1f}°C')
        ax.axvline(r['mu_star'], color=color, lw=1.2, ls='--', alpha=0.7)
    ax.set_xlabel('Temperatura máxima diaria (°C)', fontsize=12)
    ax.set_ylabel('Densidad estimada (KDE)', fontsize=12)
    ax.set_title(f'{ciudad} — Densidad KDE y umbral modal μ* por periodo',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{OUTPUT_ART}/fig1_kde_{ciudad.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Fig.1: {path}")


## FIG. 2 — Papel de probabilidad Weibull
def fig2_wpp(ciudad, all_dict):
    'Genera un gráfico de papel de probabilidad Weibull (WPP) para cada periodo.'

    periodos = [p for p in PERIODOS_ORD if get_r(all_dict, ciudad, p) is not None]
    n_p = len(periodos)
    if n_p == 0:
        return
    fig, axes = plt.subplots(1, n_p, figsize=(5*n_p, 4.5), sharey=False)
    if n_p == 1:
        axes = [axes]
    for ax, periodo in zip(axes, periodos):
        r   = get_r(all_dict, ciudad, periodo)
        rw  = r.get('r_wlr') or {}
        if not rw or 'X' not in rw:
            ax.set_title(f'{periodo}\n(sin datos WLR)'); continue
        X, Y, W = rw['X'], rw['Y'], rw['W']
        Xm, Ym  = rw['Xm'], rw['Ym']
        k_wlr   = r['k_wlr']
        intercept = Ym - k_wlr * Xm
        ## Scatter ponderado
        ax.scatter(X, Y, s=25 + 100*W/W.max(), c='#2874a6',
                   alpha=0.7, zorder=3, label='Clases positivas')
        ## Recta de regresión
        x_l = np.array([X.min()-0.1, X.max()+0.1])
        ax.plot(x_l, k_wlr*x_l + intercept, 'r--', lw=2,
                label=f'k={k_wlr:.3f}, λ={r["lam_wlr"]:.3f}')
        ax.set_xlabel('X = ln(t)', fontsize=10)
        ax.set_ylabel('Y = ln(−ln(1−F))' if ax == axes[0] else '', fontsize=10)
        ax.set_title(f'{periodo}  (R²_WLR={rw["R2_wlr"]:.4f})', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{ciudad} — Papel de probabilidad Weibull',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ART}/fig2_wpp_{ciudad.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Fig.2: {path}")


## FIG. 3 — FDA empírica vs mezcla Weibull ajustada
def fig3_cdf(ciudad, all_dict):

    periodos = [p for p in PERIODOS_ORD if get_r(all_dict, ciudad, p) is not None]
    n_p = len(periodos)
    if n_p == 0:
        return
    fig, axes = plt.subplots(1, n_p, figsize=(5*n_p, 4.5), sharey=True)
    if n_p == 1:
        axes = [axes]
    for ax, periodo in zip(axes, periodos):
        r = get_r(all_dict, ciudad, periodo)
        t = r['t_v']; C = r['C_v']; U = r['U_opt']
        ax.plot(t, C, 'o', color='#154360', ms=5, label='F̂ empírica', zorder=3)
        ax.plot(t, U, '-', color='#922b21', lw=2,
                label=f'Mezcla Weibull\nk={r["k_opt"]:.3f}, λ={r["lam_opt"]:.3f}')
        ax.axhline(r['p0'], color='gray', lw=1, ls=':',
                   label=f'p₀={r["p0"]:.2f}')
        ax.set_xlabel('t = excedencia sobre μ* (°C)', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('F(t)', fontsize=10)
        ax.set_title(f'{periodo}  (R²={r["R2_opt"]:.4f})', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{ciudad} — FDA empírica vs modelo de mezcla Weibull',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ART}/fig3_cdf_{ciudad.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Fig.3: {path}")


## FIG. 4 — Comparación de k y μ* entre ciudades y periodos
def fig4_comparacion(all_dict):
    datos = []
    for ciudad in ciudades:
        for periodo in PERIODOS_ORD:
            r = get_r(all_dict, ciudad, periodo)
            if r:
                datos.append(dict(ciudad=ciudad, periodo=periodo,
                                  k=r['k_opt'], lam=r['lam_opt'],
                                  mu=r['mu_star'], beta=r['k_paper']))
    if not datos:
        return
    periodos_u = PERIODOS_ORD
    ciudades_u = ciudades
    n_per = len(periodos_u)
    x_pos = np.arange(len(ciudades_u))
    ancho = 0.8 / max(n_per, 1)
    offsets = np.linspace(-0.4+ancho/2, 0.4-ancho/2, n_per)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ## Panel izq: k_opt por ciudad y periodo
    ax = axes[0]
    for off, periodo in zip(offsets, periodos_u):
        vals = [next((d['k'] for d in datos
                      if d['ciudad']==c and d['periodo']==periodo), np.nan)
                for c in ciudades_u]
        ax.bar(x_pos+off, vals, width=ancho*0.9,
               label=periodo, color=COLORES.get(periodo,'#999'), alpha=0.85)
    ax.axhline(1.0, color='black', lw=1.2, ls='--', alpha=0.5, label='k=1')
    ax.set_xticks(x_pos); ax.set_xticklabels(ciudades_u, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('k (parámetro de forma)', fontsize=11)
    ax.set_title('Parámetro de forma k por ciudad y periodo', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    ## Panel der: μ* por ciudad y periodo
    ax2 = axes[1]
    for off, periodo in zip(offsets, periodos_u):
        vals = [next((d['mu'] for d in datos
                      if d['ciudad']==c and d['periodo']==periodo), np.nan)
                for c in ciudades_u]
        ax2.bar(x_pos+off, vals, width=ancho*0.9,
                label=periodo, color=COLORES.get(periodo,'#999'), alpha=0.85)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(ciudades_u, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('μ* (°C)', fontsize=11)
    ax2.set_title('Umbral modal μ* por ciudad y periodo', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Parámetros Weibull comparados entre ciudades y periodos',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ART}/fig4_comparacion.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Fig.4: {path}")


## FIG. 5 — Curvas de excedencia P(X > θ)
def fig5_excedencia(ciudad, all_dict):
    periodos = [p for p in PERIODOS_ORD if get_r(all_dict, ciudad, p) is not None]
    if not periodos:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for periodo in periodos:
        r     = get_r(all_dict, ciudad, periodo)
        mu    = r['mu_star']; p0 = r['p0']
        k     = r['k_opt'];   lam = r['lam_opt']
        color = COLORES.get(periodo, '#666')
        mrk   = MARCADORES.get(periodo, 'o')
        mask  = THETA_CURVAS > mu
        thetas_sup = THETA_CURVAS[mask]
        if len(thetas_sup):
            P = (1-p0) * np.exp(-((thetas_sup - mu) / lam)**k)
            ax.plot(thetas_sup, P*100, color=color, lw=2, label=periodo)
        ax.scatter([mu], [(1-p0)*100], color=color, marker=mrk, s=90, zorder=5)
        ax.axvline(mu, color=color, lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel('Temperatura umbral θ (°C)', fontsize=12)
    ax.set_ylabel('P(X > θ)  (%)', fontsize=12)
    ax.set_title(f'{ciudad} — Probabilidad de excedencia P(X > θ)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = f"{OUTPUT_ART}/fig5_excedencia_{ciudad.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Fig.5: {path}")


## FIG. 6 — Evolución temporal de μ* y k (tendencias por ciudad)
def fig6_tendencias(all_dict):
    """Gráfico de líneas: μ* y k en función del periodo para cada ciudad."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_vals = [0, 1, 2]
    labels = ["1940-1968", "1969-1996", "1997-2026"]

    for ciudad in ciudades:
        mus, ks = [], []
        for p in PERIODOS_ORD:
            r = get_r(all_dict, ciudad, p)
            mus.append(r['mu_star'] if r else np.nan)
            ks.append(r['k_paper']   if r else np.nan)

        axes[0].plot(x_vals, mus, marker='o', lw=2, label=ciudad)
        axes[1].plot(x_vals, ks,  marker='s', lw=2, label=ciudad)

    for ax, ylabel, title in [
        (axes[0], 'μ* (°C)',           'Evolución del umbral modal μ*'),
        (axes[1], 'k (forma Weibull)', 'Evolución del parámetro de forma k'),
    ]:
        ax.set_xticks(x_vals); ax.set_xticklabels(labels, rotation=15, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    axes[1].axhline(1.0, color='black', lw=1, ls='--', alpha=0.5, label='k=1')
    fig.suptitle('Tendencias temporales de los parámetros Weibull por ciudad',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ART}/fig6_tendencias.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Fig.6: {path}")


## ANX. 1 — Histograma T>0 con PDF Weibull superpuesta
def anx1_hist_T(ciudad, all_dict):
    periodos = [p for p in PERIODOS_ORD if get_r(all_dict, ciudad, p) is not None]
    n_p = len(periodos)
    if n_p == 0:
        return
    fig, axes = plt.subplots(1, n_p, figsize=(5*n_p, 4), sharey=False)
    if n_p == 1:
        axes = [axes]
    for ax, periodo in zip(axes, periodos):
        r     = get_r(all_dict, ciudad, periodo)
        T_pos = r['T_pos']
        k, lam = r['k_opt'], r['lam_opt']
        ax.hist(T_pos, bins=30, density=True, color='#aed6f1',
                edgecolor='white', alpha=0.8, label='T > 0')
        t_r = np.linspace(T_pos.min()+0.001, T_pos.max(), 300)
        ax.plot(t_r, weibull_min.pdf(t_r, c=k, scale=lam),
                'r-', lw=2, label=f'Weibull(k={k:.3f}, λ={lam:.3f})')
        ax.set_xlabel('T = X − μ* (°C)', fontsize=10)
        ax.set_ylabel('Densidad' if ax == axes[0] else '', fontsize=10)
        ax.set_title(periodo, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{ciudad} — Excedencias T > 0 y PDF Weibull ajustada',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ANX}/anx1_histT_{ciudad.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Anx.1: {path}")


## ANX. 2 — Histograma de temperatura bruta X con KDE
def anx2_hist_X(ciudad, all_dict):
    periodos = [p for p in PERIODOS_ORD if get_r(all_dict, ciudad, p) is not None]
    n_p = len(periodos)
    if n_p == 0:
        return
    fig, axes = plt.subplots(1, n_p, figsize=(5*n_p, 4), sharey=False)
    if n_p == 1:
        axes = [axes]
    for ax, periodo in zip(axes, periodos):
        r = get_r(all_dict, ciudad, periodo)
        ax.hist(r['tmax_raw'], bins=40, color='#abebc6',
                edgecolor='white', alpha=0.85, density=True)
        ax.plot(r['x_grid'], r['densidad'], '#1a5276', lw=2, label='KDE')
        ax.axvline(r['mu_star'], color='#c0392b', lw=2, ls='--',
                   label=f'μ*={r["mu_star"]:.1f}°C')
        ax.axvline(r['mediana'], color='#7d6608', lw=1.5, ls=':',
                   label=f'Med.={r["mediana"]:.1f}°C')
        ax.set_xlabel('Temperatura máxima (°C)', fontsize=10)
        ax.set_ylabel('Densidad' if ax == axes[0] else '', fontsize=10)
        ax.set_title(periodo, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{ciudad} — Histograma de temperatura máxima diaria',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ANX}/anx2_histX_{ciudad.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Anx.2: {path}")


## ANX. 3 — Tabla visual de todos los parámetros
def anx3_tabla_parametros(all_dict):
    datos = []
    for ciudad in ciudades:
        for periodo in PERIODOS_ORD:
            r = get_r(all_dict, ciudad, periodo)
            if r:
                datos.append([ciudad, periodo,
                               f"{r['mu_star']:.2f}", f"{r['k_opt']:.4f}",
                               f"{r['lam_opt']:.4f}", f"{r['k_paper']:.4f}",
                               f"{r['R2_opt']:.4f}", f"{r['KS_stat']:.4f}",
                               f"{r['KS_pval']:.4f}"])

    cols = ['Ciudad', 'Periodo', 'μ* (°C)', 'k_opt', 'λ_opt',
            'β_paper', 'R²', 'KS', 'KS p-val']
    n_row = len(datos)
    fig, ax = plt.subplots(figsize=(14, 0.5*n_row + 1.5))
    ax.axis('off')
    tbl = ax.table(cellText=datos, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(cols))))
    ## Colorear filas por ciudad
    ciudad_actual = None; toggle = False
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
        elif col == 0 and row > 0:
            ciudad_actual = datos[row-1][0]
        if row > 0:
            c = '#eaf0fb' if (row % 2 == 0) else '#ffffff'
            cell.set_facecolor(c)
    fig.suptitle('Parámetros estimados por ciudad y periodo', fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = f"{OUTPUT_ANX}/anx3_tabla_parametros.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ Anx.3: {path}")


## EJECUTAR
print("\n" + "═"*60)
print("FIGURAS PARA EL ARTÍCULO")
print("═"*60)
for ciudad in ciudades:
    print(f"\n── {ciudad}")
    fig1_kde_comparado(ciudad, ALL)
    fig5_excedencia(ciudad, ALL)

fig6_tendencias(ALL)

print("\n" + "═"*60)
print("FIGURAS PARA ANEXO")
print("═"*60)
for ciudad in ciudades:
    print(f"\n── {ciudad}")
    anx1_hist_T(ciudad, ALL)
    anx2_hist_X(ciudad, ALL)
anx3_tabla_parametros(ALL)

print(f"\nFiguras → {OUTPUT_ART}/  y  {OUTPUT_ANX}/")
