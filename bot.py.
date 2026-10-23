"""Bot SIMULADO (paper trading): estrategia de tendencia BTC/USDT en 4h.
Un solo archivo. Dinero ficticio: NO envia ordenes reales ni usa claves.
Uso:  python bot.py        (revisa cada 5 minutos y guarda operaciones_paper.csv)
"""
import csv
import json
import os
import sys
import time
import urllib.request

import pandas as pd

# ---------------- PARAMETROS ----------------
CAPITAL_INICIAL = 50.0
RIESGO_POR_OP = 0.01      # 1 % del capital por operación
EMA_RAPIDA, EMA_LENTA, EMA_TENDENCIA = 20, 50, 200
ATR_PERIODO = 14
K_SL = 2.0                # Stop Loss = 2 ATR
RR = 2.0                  # Ratio Riesgo/Beneficio 1:2
ATR_PCT_MIN, ATR_PCT_MAX = 0.003, 0.05   # filtro de volatilidad (0,3 % - 5 % del precio)
COMISION = 0.0026         # 0,26 % por lado (taker típico en Kraken; ajusta a tu exchange)
SLIPPAGE = 0.0005         # 0,05 % por lado
DD_DIARIO_MAX = 0.03      # parar el día si pierde 3 %
DD_SEMANAL_MAX = 0.06     # parar la semana si pierde 6 %
EXCHANGE, SIMBOLO, TIMEFRAME = "kraken", "BTC/USDT", "4h"


# ---------------- INDICADORES ----------------
def indicadores(df):
    d = df.copy()
    d["ema_f"] = d["close"].ewm(span=EMA_RAPIDA, adjust=False).mean()
    d["ema_s"] = d["close"].ewm(span=EMA_LENTA, adjust=False).mean()
    d["ema_t"] = d["close"].ewm(span=EMA_TENDENCIA, adjust=False).mean()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / ATR_PERIODO, adjust=False).mean()
    d["atr_pct"] = d["atr"] / d["close"]
    cruce = (d["ema_f"] > d["ema_s"]) & (d["ema_f"].shift(1) <= d["ema_s"].shift(1))
    d["senal"] = (cruce & (d["close"] > d["ema_t"])
                  & d["atr_pct"].between(ATR_PCT_MIN, ATR_PCT_MAX))
    return d.dropna()


# ---------------- BOT ----------------
ESTADO = "estado_bot.json"
LOG = "operaciones_paper.csv"
COLUMNAS = ["entrada_ts", "salida_ts", "precio_entrada", "precio_salida",
            "cantidad", "pnl", "r", "capital"]


def estado_inicial():
    return {"equity": CAPITAL_INICIAL, "pos": None, "ultima_vela": None,
            "dia_ref": None, "eq_dia": CAPITAL_INICIAL,
            "sem_ref": None, "eq_sem": CAPITAL_INICIAL, "senal_avisada": None}


def cargar_estado():
    if os.path.exists(ESTADO):
        with open(ESTADO) as f:
            return json.load(f)
    return estado_inicial()


def guardar_estado(st):
    with open(ESTADO, "w") as f:
        json.dump(st, f, indent=2)


def registrar(fila):
    nuevo = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLUMNAS)
        w.writerow(fila)


def procesar(st, prev, bar):
    """Misma lógica que el backtest: gestiona la posición con la vela cerrada `bar`
    y, si la vela anterior tuvo señal, entra en la apertura de `bar`."""
    ts = bar.Index
    eq = st["equity"]

    dia = str(ts.date())
    if st["dia_ref"] != dia:
        st["dia_ref"], st["eq_dia"] = dia, eq
    sem = "%d-%d" % tuple(ts.isocalendar()[:2])
    if st["sem_ref"] != sem:
        st["sem_ref"], st["eq_sem"] = sem, eq

    pos = st["pos"]
    if pos:
        salida = None
        if bar.low <= pos["sl"]:
            salida = pos["sl"] * (1 - SLIPPAGE)       # si se tocan ambos, stop primero
        elif bar.high >= pos["tp"]:
            salida = pos["tp"] * (1 - SLIPPAGE)
        if salida:
            bruto = pos["qty"] * (salida - pos["entrada"])
            coste = pos["qty"] * salida * COMISION
            pnl = bruto - coste - pos["coste_ent"]
            eq += pnl
            registrar([pos["ts"], ts.isoformat(), round(pos["entrada"], 2), round(salida, 2),
                       round(pos["qty"], 6), round(pnl, 4), round(pnl / pos["riesgo"], 3), round(eq, 2)])
            print(f"[{ts}] CIERRE simulado: pnl {pnl:+.2f} EUR | capital {eq:.2f} EUR")
            st["pos"] = pos = None

    pausa = (eq <= st["eq_dia"] * (1 - DD_DIARIO_MAX)
             or eq <= st["eq_sem"] * (1 - DD_SEMANAL_MAX))

    if (not pos) and bool(prev.senal) and not pausa:
        entrada = bar.open * (1 + SLIPPAGE)
        dist = K_SL * prev.atr
        riesgo = eq * RIESGO_POR_OP
        qty = min(riesgo / dist, eq / (entrada * (1 + COMISION)))
        if qty > 0:
            st["pos"] = {"ts": ts.isoformat(), "entrada": float(entrada),
                         "sl": float(entrada - dist), "tp": float(entrada + RR * dist),
                         "qty": float(qty), "riesgo": float(riesgo),
                         "coste_ent": float(qty * entrada * COMISION)}
            print(f"[{ts}] ENTRADA simulada: {qty:.5f} BTC a {entrada:.2f} | "
                  f"SL {entrada - dist:.2f} | TP {entrada + RR * dist:.2f}")
    st["equity"] = float(eq)


def obtener_velas(ex):
    ohlcv = ex.fetch_ohlcv(SIMBOLO, TIMEFRAME, limit=700)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df.iloc[:-1]          # descarta la vela que aún está abierta


def ciclo(ex, st):
    d = indicadores(obtener_velas(ex))
    filas = list(d.itertuples())
    if st["ultima_vela"] is None:
        st["ultima_vela"] = filas[-1].Index.isoformat()
        print(f"Bot iniciado. Capital simulado: {st['equity']:.2f} EUR. "
              f"Esperando la próxima vela de {TIMEFRAME}...")
        return
    ultima = pd.Timestamp(st["ultima_vela"])
    for i in range(1, len(filas)):
        if filas[i].Index > ultima:
            procesar(st, filas[i - 1], filas[i])
            st["ultima_vela"] = filas[i].Index.isoformat()

    ult = filas[-1]
    if bool(ult.senal) and not st["pos"] and st["senal_avisada"] != ult.Index.isoformat():
        st["senal_avisada"] = ult.Index.isoformat()
        print(f"[{ult.Index}] SEÑAL DE ENTRADA detectada. Se entraría en la apertura de la próxima vela.")


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def velas_binance():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=4h&limit=700"
    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in _get_json(url)]


def velas_kraken():
    url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSDT&interval=240"
    data = _get_json(url)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    clave = [k for k in data["result"] if k != "last"][0]
    return [[int(k[0]) * 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[6])]
            for k in data["result"][clave]]


class Mercado:
    """Descarga velas de 4h de BTC/USDT de API públicas (Binance y, si falla, Kraken)."""

    def fetch_ohlcv(self, simbolo, timeframe, limit=700):
        errores = []
        for fuente in (velas_binance, velas_kraken):
            try:
                return fuente()
            except Exception as e:
                errores.append(f"{fuente.__name__}: {e}")
        raise RuntimeError(" | ".join(errores))


def main():
    ex = Mercado()
    st = cargar_estado()
    una_vez = "--una-vez" in sys.argv
    while True:
        try:
            ciclo(ex, st)
            guardar_estado(st)
        except Exception as e:                 # fallo de red, etc.: reintenta más tarde
            print("Aviso:", e)
        if una_vez:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
