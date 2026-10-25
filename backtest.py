
"""Backtest de la estrategia de Ra: BTC/USD, velas de 4 h, ~4 años.
Datos: Bitstamp (API pública, sin claves). Escribe informe_backtest.md."""
import json, time, urllib.request
import numpy as np, pandas as pd

CAP0, RIESGO, K_SL, RR = 50.0, 0.01, 2.0, 2.0
COM, SLIP = 0.0026, 0.0005
ATR_MIN, ATR_MAX = 0.003, 0.05
DD_DIA, DD_SEM = 0.03, 0.06
ANIOS = 4

def descargar():
    fin = int(time.time())
    ini = fin - ANIOS * 365 * 86400
    filas, cur = [], ini
    while cur < fin:
        url = ("https://www.bitstamp.net/api/v2/ohlc/btcusd/"
               f"?step=14400&limit=1000&start={cur}")
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
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["t"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    return df.sort_values("t").reset_index(drop=True)

def indicadores(df):
    c = df["close"]
    df["ef"], df["es"], df["et"] = (c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200))
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df

def simular(df):
    eq, pos, trades = CAP0, None, []
    curva = []
    dia = sem = None
    eq_dia = eq_sem = eq
    for i in range(201, len(df) - 1):
        r, t = df.iloc[i], df.iloc[i]["t"]
        d, s = t.date(), t.isocalendar()[:2]
        if d != dia:
            dia, eq_dia = d, eq
        if s != sem:
            sem, eq_sem = s, eq
        if pos:
            sl, tp = pos["sl"], pos["tp"]
            sal = None
            if r.low <= sl:
                sal = sl * (1 - SLIP)
            elif r.high >= tp:
                sal = tp * (1 - SLIP)
            if sal:
                bruto = (sal - pos["e"]) * pos["q"]
                coste = COM * (pos["e"] + sal) * pos["q"]
                pnl = bruto - coste
                eq += pnl
                trades.append({"entrada": pos["t"], "salida": t, "pnl": pnl,
                               "r": pnl / pos["riesgo"], "eq": eq})
                pos = None
        curva.append(eq)
        if pos:
            continue
        p = df.iloc[i - 1]
        cruce = p.ef <= p.es and r.ef > r.es
        atrp = r.atr / r.close
        parado = eq < eq_dia * (1 - DD_DIA) or eq < eq_sem * (1 - DD_SEM)
        if cruce and r.close > r.et and ATR_MIN <= atrp <= ATR_MAX and not parado:
            e = df.iloc[i + 1].open * (1 + SLIP)
            riesgo = eq * RIESGO
            q = min(riesgo / (K_SL * r.atr), eq / e / (1 + COM))
            pos = {"e": e, "q": q, "sl": e - K_SL * r.atr,
                   "tp": e + K_SL * r.atr * RR, "t": df.iloc[i + 1]["t"],
                   "riesgo": q * K_SL * r.atr}
    return pd.DataFrame(trades), np.array(curva)

def informe(df, tr, curva):
    bh = (df.close.iloc[-1] / df.close.iloc[201] - 1) * 100
    L = ["# Informe backtest de Ra",
         f"Periodo: {df.t.iloc[201].date()} a {df.t.iloc[-1].date()} (BTC/USD, 4 h, Bitstamp)",
         f"Comisión {COM*100:.2f} % por lado + deslizamiento {SLIP*100:.2f} % por lado.", ""]
    if tr.empty:
        L.append("No hubo operaciones.")
    else:
        g, p = tr[tr.pnl > 0].pnl.sum(), -tr[tr.pnl <= 0].pnl.sum()
        pico = np.maximum.accumulate(curva)
        dd = ((pico - curva) / pico).max() * 100
                L += [f"- Capital final: {tr['eq'].iloc[-1]:.2f} € (inicio {CAPO:.0f} €)",
              f"- Resultado: {(tr['eq'].iloc[-1]/CAPO-1)*100:+.1f} %",
              f"- Comprar y mantener BTC: {bh:+.1f} %",
              f"- Operaciones: {len(tr)}",
              f"- Ganadas: {(tr.pnl>0).mean()*100:.0f} %",
              f"- Factor de beneficio: {g/p:.2f}" if p > 0 else "- Factor de beneficio: sin pérdidas",
              f"- R medio por operación: {tr.r.mean():.2f}",
              f"- Caída máxima: {dd:.1f} %", "",
              "Criterio para seguir: al menos 30 operaciones y factor de beneficio > 1,3.",
              "Un resultado bueno en el pasado no garantiza nada en el futuro."]
        tr.to_csv("operaciones_backtest.csv", index=False)
    open("informe_backtest.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    df = indicadores(descargar())
    print("Velas descargadas:", len(df))
    tr, curva = simular(df)
    informe(df, tr, curva)
