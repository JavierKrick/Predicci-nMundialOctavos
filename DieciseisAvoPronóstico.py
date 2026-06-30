import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import defaultdict

# Asumimos que estos datos provienen de tu módulo local
from datos import (
    datos_potencia_torneos_2026, 
    datos_torneos_completos, 
    partidos_filtrados, 
    teams_base
)

# ══════════════════════════════════════════════════════════════════════════════
# §1 CONSTANTES GLOBALES Y MAPEOS
# ══════════════════════════════════════════════════════════════════════════════

BASE_ELO   = 1500   
BASE_POT   = 1850   
PRIOR_MEAN = 1.3    
PRIOR_PJ   = 3.0    
PESO_ELO_MAX   = 0.5
PESO_ELO_MIN   = 0.45
PESO_STATS = 1-PESO_ELO_MAX

ANFITRIONES_2026 = {"México", "Mexico", "Canadá", "Canada", "Estados Unidos", "USA", "EEUU"}

MAPA_EQUIPOS = {
    "South Africa":                     "Sudáfrica",
    "Sweden":                           "Suecia",
    "República Democrática del Congo":  "RD Congo",
    "Algeria":                          "Argelia",
}
MAPA_TORNEOS = {"World Cup": "World Cups"}

datos_normalizados = [
    {**r,
     "equipo": MAPA_EQUIPOS.get(r["equipo"], r["equipo"]),
     "torneo": MAPA_TORNEOS.get(r["torneo"], r["torneo"])}
    for r in datos_torneos_completos
]

dict_potencias = {d["torneo"]: d["potencia_torneo"] for d in datos_potencia_torneos_2026}

# ══════════════════════════════════════════════════════════════════════════════
# §2 MODELO 1: REGRESIÓN ELO (Histórico)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_modelo_elo(partidos):
    X, Y = [], []
    for p in partidos:
        d = p["elo_equipo1"] - p["elo_equipo2"]
        X.extend([d, -d])
        Y.extend([p["goles_equipo1"], p["goles_equipo2"]])
    
    df = pd.DataFrame({"goles": Y, "delta_elo": X, "intercepto": 1.0})
    modelo = sm.GLM(df["goles"], df[["intercepto", "delta_elo"]], family=sm.families.Poisson())
    res = modelo.fit()
    return res.params["intercepto"], res.params["delta_elo"]

B0_PROPIO, B1_PROPIO = entrenar_modelo_elo(partidos_filtrados)
B0_ELO_Paper = 0.25
B1_ELO_Paper = 0.0023    #tomado dee un paper

B0_ELO = B0_PROPIO*(0.5)+B0_ELO_Paper*(0.5) 
B1_ELO = B1_PROPIO*(0.5)+B1_ELO_Paper*(0.5) 

# ══════════════════════════════════════════════════════════════════════════════
# §3 MOTOR EM: CALIBRACIÓN DE LA DIFICULTAD DEL TORNEO (Orientado a Datos)
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def torneo_confianza(potencia):
    return sigmoid((potencia - BASE_POT) / 150.0)

def peso_observacion(n_partidos, potencia):
    return n_partidos * torneo_confianza(potencia)

def _e_step(datos, dict_potencias, params, lambda_elo=5.0):
    num = {"atk": defaultdict(float), "def": defaultdict(float), "gf": defaultdict(float), "ga": defaultdict(float)}
    den = {"atk": defaultdict(float), "def": defaultdict(float)}

    for r in datos:
        eq  = r["equipo"]
        pot = dict_potencias.get(r["torneo"], BASE_POT)
        x_j = (pot - BASE_POT) / 200.0
        
        g_atk = params["atk"][0] + params["atk"][1] * x_j
        g_def = params["def"][0] + params["def"][1] * x_j

        if g_atk < 1e-9 or g_def < 1e-9: continue

        w = peso_observacion(r["partidos_jugados"], pot)

        num["atk"][eq] += w * r["xg_90"] / g_atk
        num["gf"][eq]  += w * r.get("goles_favor_90", r["xg_90"]) / g_atk
        den["atk"][eq] += w

        num["def"][eq] += w * r["xga_90"] / g_def
        num["ga"][eq]  += w * r.get("goles_contra_90", r["xga_90"]) / g_def
        den["def"][eq] += w

    # Anclaje neutral en el entrenamiento
    latente = {}
    for eq in den["atk"]:
        if den["atk"][eq] > 0.0:
            latente[eq] = {
                "atk": (num["atk"][eq] + lambda_elo * PRIOR_MEAN) / (den["atk"][eq] + lambda_elo),
                "gf":  (num["gf"][eq]  + lambda_elo * PRIOR_MEAN) / (den["atk"][eq] + lambda_elo),
                "def": (num["def"][eq] + lambda_elo * PRIOR_MEAN) / (den["def"][eq] + lambda_elo),
                "ga":  (num["ga"][eq]  + lambda_elo * PRIOR_MEAN) / (den["def"][eq] + lambda_elo),
            }
    return latente

def _m_step(datos, latente, dict_potencias):
    xs_atk, ys_atk, xs_def, ys_def = [], [], [], []
    PESO_XG, PESO_GF = 1.0, 0.3

    for r in datos:
        eq = r["equipo"]
        if eq not in latente: continue
        pot = dict_potencias.get(r["torneo"], BASE_POT)
        x = (pot - BASE_POT) / 200.0

        if latente[eq]["atk"] > 1e-9:
            xs_atk.extend([x] * int(PESO_XG * 10))
            ys_atk.extend([r["xg_90"] / latente[eq]["atk"]] * int(PESO_XG * 10))
            xs_atk.extend([x] * int(PESO_GF * 10))
            ys_atk.extend([r.get("goles_favor_90", r["xg_90"]) / latente[eq]["gf"]] * int(PESO_GF * 10))

        if latente[eq]["def"] > 1e-9:
            xs_def.extend([x] * int(PESO_XG * 10))
            ys_def.extend([r["xga_90"] / latente[eq]["def"]] * int(PESO_XG * 10))
            xs_def.extend([x] * int(PESO_GF * 10))
            ys_def.extend([r.get("goles_contra_90", r["xga_90"]) / latente[eq]["ga"]] * int(PESO_GF * 10))

    def fit_ols(xs, ys):
        if len(xs) < 2: return 1.0, 0.0
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        var = sum((x - mx)**2 for x in xs)
        b = sum((x - mx)*(y - my) for x, y in zip(xs, ys)) / var if var > 1e-12 else 0.0
        return my - b * mx, b

    a_atk, b_atk = fit_ols(xs_atk, ys_atk)
    a_def, b_def = fit_ols(xs_def, ys_def)
    return {"atk": (a_atk, b_atk), "def": (a_def, b_def)}

def estimar_modelo_latente(datos, dict_potencias, max_iter=20, tol=1e-5):
    params = {"atk": (1.0, 0.0), "def": (1.0, 0.0)}
    for _ in range(max_iter):
        latente = _e_step(datos, dict_potencias, params)
        params_new = _m_step(datos, latente, dict_potencias)
        if abs(params_new["atk"][1] - params["atk"][1]) < tol:
            break
        params = params_new
    return latente, params_new

_, PARAMS_TORNEO = estimar_modelo_latente(datos_normalizados, dict_potencias)

# ══════════════════════════════════════════════════════════════════════════════
# §4 ACUMULACIÓN Y REGULARIZACIÓN ESTRUCTURAL (PRIOR ELO)
# ══════════════════════════════════════════════════════════════════════════════

def proyectar_a_mundial(q, pot, a, b):
    x = (pot - BASE_POT) / 200.0
    g = a + b * x
    return max(0.0, q * a / g) if g > 1e-9 else q

a_atk, b_atk = PARAMS_TORNEO["atk"]
a_def, b_def = PARAMS_TORNEO["def"]

stats_acumuladas = defaultdict(lambda: {"xg": 0.0, "xga": 0.0, "gf": 0.0, "ga": 0.0, "w": 0.0})

# Acumulación pura proyectada a nivel Mundial
for r in datos_normalizados:
    eq, pot = r["equipo"], dict_potencias.get(r["torneo"], BASE_POT)
    w = peso_observacion(r["partidos_jugados"], pot)
    
    s = stats_acumuladas[eq]
    s["xg"]  += proyectar_a_mundial(r["xg_90"], pot, a_atk, b_atk) * w
    s["gf"]  += proyectar_a_mundial(r.get("goles_favor_90", r["xg_90"]), pot, a_atk, b_atk) * w
    s["xga"] += proyectar_a_mundial(r["xga_90"], pot, a_def, b_def) * w
    s["ga"]  += proyectar_a_mundial(r.get("goles_contra_90", r["xga_90"]), pot, a_def, b_def) * w
    s["w"]   += w

def get_elo_prior(eq, tipo="atk"):
    elo = teams_base.get(eq, [BASE_ELO])[0]
    signo = 1.0 if tipo == "atk" else -1.0
    return PRIOR_MEAN * math.exp(signo * (elo - BASE_ELO) / 600.0)

equipos_stats = {}
peso_prior = PRIOR_PJ * torneo_confianza(BASE_POT)

for eq, s in stats_acumuladas.items():
    denom = s["w"] + peso_prior
    prior_atk = get_elo_prior(eq, "atk")
    prior_def = get_elo_prior(eq, "def")
    
    # Shrinkage hacia la fuerza esperada por su Elo
    xg_base  = (s["xg"]  + peso_prior * prior_atk) / denom
    gf_base  = (s["gf"]  + peso_prior * prior_atk) / denom
    xga_base = (s["xga"] + peso_prior * prior_def) / denom
    ga_base  = (s["ga"]  + peso_prior * prior_def) / denom
    
    # 1. Definir un finishing ratio más estable (centrado en 1.0)
    finishing = (gf_base / xg_base) if xg_base > 1e-9 else 1.0

    # 2. Aplicar un factor de contracción hacia la media (Shrinkage)
    # Si finishing es 1.2 (un 20% mejor que el xG), lo reducimos a 1.04 
    # para no sobreestimar una racha de buena suerte.
    finishing_reg = 1.0 + (finishing - 1.0) * 0.2 

    # 3. Asegurar que no sea menor a un límite lógico (ej. 0.8)
    finishing_reg = max(0.8, min(1.2, finishing_reg))

    equipos_stats[eq] = {
        "xg":  xg_base * finishing_reg,
        # El blend defensivo: si quieres más coherencia, 
        # mantén el peso mayor en xGA (proceso) y solo un toque de GA (resultado).
        "xga": xga_base * 0.9 + ga_base * 0.1 
    }

global_atk = sum(s["xg"] for s in equipos_stats.values()) / len(equipos_stats) if equipos_stats else PRIOR_MEAN
global_def = sum(s["xga"] for s in equipos_stats.values()) / len(equipos_stats) if equipos_stats else PRIOR_MEAN

# ══════════════════════════════════════════════════════════════════════════════
# §5 SIMULACIÓN (Dixon & Coles + NB2 + Cópula de Frank)
# ══════════════════════════════════════════════════════════════════════════════

def prob_nb2(mu, alpha, k):
    if alpha < 1e-9:
        if mu < 1e-12: return 1.0 if k == 0 else 0.0
        return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))
    r = 1.0 / alpha
    p = r / (r + mu)
    return math.exp(
        math.lgamma(k + r) - math.lgamma(k + 1) - math.lgamma(r)
        + r * math.log(p) + k * math.log1p(-p)
    )

def cdf_nb2(mu, alpha, k):
    if k < 0: return 0.0
    return sum(prob_nb2(mu, alpha, x) for x in range(k + 1))

def frank_copula(u, v, theta):
    if abs(theta) < 1e-5: return u * v 
    num = (math.exp(-theta * u) - 1.0) * (math.exp(-theta * v) - 1.0)
    den = math.exp(-theta) - 1.0
    adentro_log = 1.0 + num / den
    if adentro_log <= 0: return 0.0
    return - (1.0 / theta) * math.log(adentro_log)
def calcular_probabilidades(t1, t2, eliminatoria=True):
    elo1 = teams_base.get(t1, [BASE_ELO])[0]
    elo2 = teams_base.get(t2, [BASE_ELO])[0]
    
    if t1 in ANFITRIONES_2026: elo1 += 50
    if t2 in ANFITRIONES_2026: elo2 += 50
    
    loc_atk1, loc_def1 = (1.05, 0.95) if t1 in ANFITRIONES_2026 else (1.0, 1.0)
    loc_atk2, loc_def2 = (1.05, 0.95) if t2 in ANFITRIONES_2026 else (1.0, 1.0)

    s1 = equipos_stats.get(t1, {"xg": PRIOR_MEAN, "xga": PRIOR_MEAN})
    s2 = equipos_stats.get(t2, {"xg": PRIOR_MEAN, "xga": PRIOR_MEAN})
    
    FACTOR_REDUCCION = 1

    lam1_stats = global_atk * (s1["xg"] * loc_atk1 / global_atk) * (s2["xga"] * loc_def2 / global_def) * FACTOR_REDUCCION
    lam2_stats = global_atk * (s2["xg"] * loc_atk2 / global_atk) * (s1["xga"] * loc_def1 / global_def) * FACTOR_REDUCCION
    
    delta_elo = elo1 - elo2
    # 1. Definir función para calcular distribución completa de un par de lambdas
    def generar_distribucion(l1, l2):
        dist = {}
        alpha_duelo = 0.002 + 0.0002 * abs(delta_elo)
        theta_copula = 0.38  # Ajustado al 3.8% de Tau de Kendall real del fútbol

        for i in range(10):
            for j in range(10):
                u1, v1 = cdf_nb2(l1, alpha_duelo, i), cdf_nb2(l2, alpha_duelo, j)
                u0, v0 = cdf_nb2(l1, alpha_duelo, i - 1), cdf_nb2(l2, alpha_duelo, j - 1)
                
                C11 = frank_copula(u1, v1, theta_copula)
                C01 = frank_copula(u0, v1, theta_copula)
                C10 = frank_copula(u1, v0, theta_copula)
                C00 = frank_copula(u0, v0, theta_copula)
                
                dist[(i, j)] = max(C11 - C01 - C10 + C00, 0.0)
        return dist

    # 2. Calcular distribuciones por separado
    lam1_elo = math.exp(B0_ELO + B1_ELO * delta_elo)
    lam2_elo = math.exp(B0_ELO - B1_ELO * delta_elo)
    
    dist_stats = generar_distribucion(lam1_stats, lam2_stats)
    dist_elo   = generar_distribucion(lam1_elo, lam2_elo)

    # 3. Ensamble de probabilidades (Ponderación final - LÓGICA INVERTIDA)
    d = abs(delta_elo)
    factor_distancia = float(np.minimum(1.0, d / 400.0))
    w_elo = PESO_ELO_MAX - (PESO_ELO_MAX - PESO_ELO_MIN) * factor_distancia

    # Capeo de seguridad estructural
    w_elo = max(PESO_ELO_MIN, min(PESO_ELO_MAX, w_elo))

    prob_90 = {k: ((1 - w_elo) * dist_stats[k] + w_elo * dist_elo[k]) 
               for k in dist_stats.keys()}

    top_90 = sorted(prob_90.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # ← NUEVO: calcular goles esperados y prob de empate en 90'
    exp_goals_90 = sum((i + j) * p for (i, j), p in prob_90.items())
    prob_draw_90 = sum(p for (i, j), p in prob_90.items() if i == j)

    if not eliminatoria:
        return {
            "top_90": top_90,
            "exp_goals_90": exp_goals_90,
            "prob_draw_90": prob_draw_90
        }

    # Como ya no tenemos lam1/lam2 únicas, usamos la media de las intensidades 
    # para estimar los goles de la prórroga (un cuarto del tiempo original (no suelen agrega tiiempo))
    # ── PRÓRROGA: Calibración empírica de fatiga y enos tiempo de juego, etc ──
    FACTOR_ET = 0.2  #(tiempo etra+ fatiga) 

    lam1_et = ((1 - w_elo) * lam1_stats + w_elo * lam1_elo) * FACTOR_ET
    lam2_et = ((1 - w_elo) * lam2_stats + w_elo * lam2_elo) * FACTOR_ET
    
    alpha_et = 0.001 + 0.0002 * abs(delta_elo)
    
    prob_et = {
        (ea, eb): prob_nb2(lam1_et, alpha_et, ea) * prob_nb2(lam2_et, alpha_et, eb)
        for ea in range(5) for eb in range(5)
    }
    
    prob_120 = {}
    for (i, j), p90 in prob_90.items():
        if i != j:
            prob_120[(i, j)] = prob_120.get((i, j), 0.0) + p90
        else:
            for (ea, eb), pet in prob_et.items():
                prob_120[(i + ea, j + eb)] = prob_120.get((i + ea, j + eb), 0.0) + (p90 * pet)
                
    suma_120 = sum(prob_120.values())
    prob_120 = {k: v / suma_120 for k, v in prob_120.items()}
    
    p_win1 = sum(p for (i, j), p in prob_120.items() if i > j)
    p_draw = sum(p for (i, j), p in prob_120.items() if i == j) 
    p_win2 = sum(p for (i, j), p in prob_120.items() if i < j)
    
        # Probabilidad de que el equipo 1 gane en penales
    # Usamos tanh para mantenerla entre 0 y 1
    # Probabilidad de que el equipo 1 gane en penales
    p_pen1 = 0.5 + 0.045 * math.tanh(delta_elo / 300.0)

    # El techo histórico absoluto es 54% - 46%
    p_pen1 = max(0.46, min(0.54, p_pen1))

    # Probabilidad final (120' + penales)
    p_final1 = p_win1 + p_draw * p_pen1
    p_final2 = p_win2 + p_draw * (1.0 - p_pen1)
        
    return {
        "top_90": top_90,
        "exp_goals_90": exp_goals_90,
        "prob_draw_90": prob_draw_90,
        "top_120": sorted(prob_120.items(), key=lambda x: x[1], reverse=True)[:5],
        "prob_local": p_final1,   # Probabilidad de ganar en 120'
        "prob_visita": p_final2,  # Probabilidad de ganar en 120'
        "prob_local_120": p_win1, 
        "prob_draw_120": p_draw,
        "prob_visita_120": p_win2
    }

# ══════════════════════════════════════════════════════════════════════════════
# §6 EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

# --- Mapeo de siglas ---
codes = {
    "Sudáfrica": "AFS", "Canadá": "CAN", "Brasil": "BRA", "Japón": "JPN",
    "Alemania": "ALE", "Paraguay": "PAR", "Países Bajos": "NED", "Marruecos": "MAR",
    "Costa de Marfil": "CIV", "Noruega": "NOR", "Francia": "FRA", "Suecia": "SUE",
    "México": "MEX", "Ecuador": "ECU", "Inglaterra": "ENG", "RD Congo": "RDC",
    "Bélgica": "BEL", "Senegal": "SEN", "Estados Unidos": "USA", "Bosnia y Herzegovina": "BIH",
    "España": "ESP", "Austria": "AUT", "Portugal": "POR", "Croacia": "CRO",
    "Suiza": "SUI", "Argelia": "ALG", "Australia": "AUS", "Egipto": "EGY",
    "Argentina": "ARG", "Cabo Verde": "CPV", "Colombia": "COL", "Ghana": "GHA"
}

matches = [
    ("Sudáfrica", "Canadá"), ("Brasil", "Japón"), ("Alemania", "Paraguay"),
    ("Países Bajos", "Marruecos"), ("Costa de Marfil", "Noruega"), ("Francia", "Suecia"),
    ("México", "Ecuador"), ("Inglaterra", "RD Congo"), ("Bélgica", "Senegal"),
    ("Estados Unidos", "Bosnia y Herzegovina"), ("España", "Austria"), ("Portugal", "Croacia"),
    ("Suiza", "Argelia"), ("Australia", "Egipto"), ("Argentina", "Cabo Verde"), ("Colombia", "Ghana")
]

# --- Configuración de anchos ---
w_partido = 35 # Reducido ligeramente porque las siglas ocupan menos
w_top = 35
w_probs = 25

def _fmt(lista): 
    return ", ".join(f"{i}-{j} ({p:.1%})" for (i, j), p in lista)

# --- Encabezado ---
print(f"{'PARTIDO (Pasa de ronda)':<{w_partido}} | {'TOP 90 MINS':<{w_top}} | {'TOP 120 MINS':<{w_top}} | {'PROBS 120'}")
print("─" * (w_partido + w_top + w_top + w_probs + 10))

sum_goals_90 = 0.0
sum_draw_90  = 0.0
n = len(matches)

# --- Bucle de ejecución ---
for t1, t2 in matches:
    try:
        res = calcular_probabilidades(t1, t2, eliminatoria=True)
        
        # Convertimos los nombres a siglas
        t1_c = codes.get(t1, t1[:3].upper())
        t2_c = codes.get(t2, t2[:3].upper())
        
        # Determinamos el favorito usando siglas
        fav = t1 if res['prob_local'] > res['prob_visita'] else t2
        fav_c = codes.get(fav, fav[:3].upper())
        
        prob_fav = max(res['prob_local'], res['prob_visita'])

        partido_str = f"{t1_c} vs {t2_c} ({fav_c} {prob_fav:.0%})"
        
        top_90_3 = _fmt(res['top_90'][:3])
        top_120_3 = _fmt(res['top_120'][:3])
        
        p_l = res['prob_local_120']
        p_e = res['prob_draw_120']
        p_v = res['prob_visita_120']
        probs_120_str = f"L:{p_l:.0%} E:{p_e:.0%} V:{p_v:.0%}"

        print(f"{partido_str:<{w_partido}} | {top_90_3:<{w_top}} | {top_120_3:<{w_top}} | {probs_120_str}")

        sum_goals_90 += res["exp_goals_90"]
        sum_draw_90  += res["prob_draw_90"]

    except Exception as e:
        print(f"Error en {t1} vs {t2}: {e}")

# --- Pie de tabla ---
print("-" * (w_partido + w_top + w_top + w_probs + 10))
print(f"Goles esperados promedio: {sum_goals_90 / n:.2f}")
print(f"Empates esperados promedio: {sum_draw_90 / n:.2%}")



# ══════════════════════════════════════════════════════════════════════════════
# §7 AUDITORÍA DE PARÁMETROS Y PENDIENTES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * (w_partido + w_top + w_top + w_probs + 10))
print(" AUDITORÍA DE PENDIENTES (SLOPES) Y PARÁMETROS DEL MODELO")
print("═" * (w_partido + w_top + w_top + w_probs + 10))

# Modelo 1: Regresión Histórica Elo
print("► MODELO 1 (Regresión GLM Poisson sobre Elo):")
print(f"  Intercepto (B0) : {B0_ELO:.4f}  (Base de goles en duelo igualado)")
# Usamos .6f porque B1_ELO suele ser muy pequeño (ej. 0.0035)
print(f"  Pendiente  (B1) : {B1_ELO:.6f}  (Impacto por cada punto de Elo)") 

# Modelo 2: Motor EM basado en Stats
print("\n► MODELO 2 (Motor EM de Stats Latentes):")
print("  (Ajuste lineal respecto a la Dificultad del Torneo)")
print(f"  Ataque  -> Intercepto: {PARAMS_TORNEO['atk'][0]:.4f} | Pendiente: {PARAMS_TORNEO['atk'][1]:.4f}")
print(f"  Defensa -> Intercepto: {PARAMS_TORNEO['def'][0]:.4f} | Pendiente: {PARAMS_TORNEO['def'][1]:.4f}")
print("═" * (w_partido + w_top + w_top + w_probs + 10) + "\n")