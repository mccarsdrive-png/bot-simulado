"""Backtest de la estrategia de Ra (mismas reglas que bot.py), varios años.
Uso: python backtest.py            (descarga datos y escribe informe_backtest.md)
     python backtest.py --sintetico (prueba sin internet)
"""
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

CAPITAL = 50.0
EMA_F, EMA_S, EMA_T, ATR_N = 20, 50, 200, 14
K_SL, RR = 2.0, 2.0
ATR_MIN, ATR_MAX = 0.003, 0.05
COMISION, SLIP = 0.0026, 0.0005
DD_DIA, DD_SEM = 0.03, 0.06
ANIOS = 4


def descargar(par, anios=ANIOS):
    """Velas de 4h de Bitstamp (API publica, sin claves). par: btcusd, ethusd."""
    fin = int(time.time())
    cur = fin - int(anios * 365.25 * 86400)
    filas = []
    while cur < fin:
        url = f"https://www.bitstamp.net/api/v2/ohlc/{par}/?step=14400&limit=1000&start={cur}"
        req = urllib.request.Request(url, headers={"User-Agent": "ra-backtest"})
        datos = json.load(urllib.request.urlopen(req, timeout=30))["data"]["ohlc"]
        if not datos:
            break
        filas += datos
        ult = int(datos[-1]["timestamp"])
        if ult <= cur:
            break
        cur = ult + 14400
        time.sleep(0.5)
    df = pd.DataFrame(filas).drop_duplicates("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    return df.set_index("ts").sort_index()[["open", "high", "low", "close", "volume"]].iloc[:-1]


def sintetico(semilla, n=8760):
    rng = np.random.default_rng(semilla)
    ret = rng.normal(0.0001, 0.012, n)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.006, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + spread),
                         "low": np.minimum(open_, close) * (1 - spread), "close": close,
                         "volume": 1.0}, index=idx)


def indicadores(df):
    d = df.copy()
    d["ef"] = d["close"].ewm(span=EMA_F, adjust=False).mean()
    d["es"] = d["close"].ewm(span=EMA_S, adjust=False).mean()
    d["et"] = d["close"].ewm(span=EMA_T, adjust=False).mean()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False).mean()
    d["atr_pct"] = d["atr"] / d["close"]
    cruce = (d["ef"] > d["es"]) & (d["ef"].shift(1) <= d["es"].shift(1))
    d["senal"] = cruce & (d["close"] > d["et"]) & d["atr_pct"].between(ATR_MIN, ATR_MAX)
    return d.dropna()


def simular(df, riesgo):
    d = indicadores(df)
    ts, op, hi, lo = d.index, d["open"].values, d["high"].values, d["low"].values
    sen, atr = d["senal"].values, d["atr"].values
    eq, pos, trades = CAPITAL, None, []
    dia_ref = sem_ref = None
    eq_dia = eq_sem = eq
    for i in range(1, len(d)):
        t = ts[i]
        if dia_ref != t.date():
            dia_ref, eq_dia = t.date(), eq
        s = tuple(t.isocalendar()[:2])
        if sem_ref != s:
            sem_ref, eq_sem = s, eq
        if pos:
            sal = None
            if lo[i] <= pos["sl"]:
                sal = pos["sl"] * (1 - SLIP)
            elif hi[i] >= pos["tp"]:
                sal = pos["tp"] * (1 - SLIP)
            if sal:
                pnl = pos["qty"] * (sal - pos["ent"]) - pos["qty"] * sal * COMISION - pos["coste"]
                eq += pnl
                trades.append((t, pnl, pnl / pos["riesgo"], eq))
                pos = None
        pausa = eq <= eq_dia * (1 - DD_DIA) or eq <= eq_sem * (1 - DD_SEM)
        if pos is None and sen[i - 1] and not pausa:
            ent = op[i] * (1 + SLIP)
            dist = K_SL * atr[i - 1]
            r = eq * riesgo
            qty = min(r / dist, eq / (ent * (1 + COMISION)))
            if qty > 0:
                pos = {"ent": ent, "sl": ent - dist, "tp": ent + RR * dist, "qty": qty,
                       "riesgo": r, "coste": qty * ent * COMISION}
    return trades, eq, d


def metricas(trades, eq_final, d):
    n = len(trades)
    if n == 0:
        return {"n": 0}
    pnls = np.array([x[1] for x in trades])
    ganado, perdido = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    curva = np.array([CAPITAL] + [x[3] for x in trades])
    pico = np.maximum.accumulate(curva)
    mdd = ((curva - pico) / pico).min() * 100
    bh = (d["close"].iloc[-1] / d["close"].iloc[0] - 1) * 100
    anios = {}
    for t, p, _, _ in trades:
        anios[t.year] = anios.get(t.year, 0) + p
    return {"n": n, "win": (pnls > 0).mean() * 100, "neto": eq_final - CAPITAL,
            "ret": (eq_final / CAPITAL - 1) * 100, "pf": (ganado / perdido) if perdido > 0 else float("inf"),
            "mdd": mdd, "bh": bh, "anios": anios,
            "r_medio": float(np.mean([x[2] for x in trades]))}


def main():
    sint = "--sintetico" in sys.argv
    activos = {"btcusd": 1, "ethusd": 2}
    filas, detalle = [], []
    for sim, semilla in activos.items():
        try:
            df = sintetico(semilla) if sint else descargar(sim)
        except Exception as e:
            print("Aviso:", e)
            filas.append(f"| {sim} | error de descarga | | | | | | |")
            continue
        periodo = f"{df.index[0].date()} a {df.index[-1].date()} ({len(df)} velas)"
        for riesgo in (0.01, 0.02):
            tr, eqf, d = simular(df, riesgo)
            m = metricas(tr, eqf, d)
            if m["n"] == 0:
                filas.append(f"| {sim} | {riesgo*100:.0f} % | 0 | - | - | - | - | - |")
                continue
            filas.append(f"| {sim} | {riesgo*100:.0f} % | {m['n']} | {m['win']:.0f} % | {m['pf']:.2f} | "
                         f"{m['ret']:+.1f} % | {m['mdd']:.1f} % | {m['bh']:+.0f} % |")
            por_anio = ", ".join(f"{a}: {v:+.2f} €" for a, v in sorted(m["anios"].items()))
            detalle.append(f"- {sim} {riesgo*100:.0f} %: R medio {m['r_medio']:+.2f}. Por año: {por_anio}")
        detalle.append(f"- {sim} periodo: {periodo}")
    texto = ["# Backtest de Ra", "",
             f"Capital inicial {CAPITAL:.0f} €, comisión {COMISION*100:.2f} % y slippage {SLIP*100:.2f} % por lado, "
             "gráfico 4h, sin apalancamiento.", "",
             "| Activo | Riesgo | Operaciones | Ganadoras | Profit factor | Retorno | Drawdown máx | Comprar y mantener |",
             "|---|---|---|---|---|---|---|---|"] + filas + ["", "## Detalle", ""] + detalle
    if sint:
        texto += ["", "(datos simulados: solo prueba técnica)"]
    out = "\n".join(texto)
    print(out)
    with open("informe_backtest.md", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
