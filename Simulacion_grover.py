"""Grover con decoherencia: despolarización por qubit sobre la matriz de densidad.
Canal (4 operadores de Kraus) aplicado a cada qubit UNA vez por iteración, después de G.
Marcado: |1...1>. Preparación y medición ideales. Determinista (sin azar).
Uso: python simulacion_grover.py   (requiere numpy y matplotlib)"""
import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = np.array([[0,1],[1,0]], complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1,-1]).astype(complex); I2 = np.eye(2, dtype=complex)

def op_en_qubit(op, q, n):
    m = np.array([[1]], complex)
    for i in range(n): m = np.kron(m, op if i == q else I2)
    return m

def kraus_por_qubit(n, p):
    """Lista (uno por qubit) de los 4 operadores de Kraus del canal despolarizante."""
    return [[np.sqrt(1-p)*np.eye(2**n)] + [np.sqrt(p/3)*op_en_qubit(o, q, n) for o in (X, Y, Z)] for q in range(n)]

def evolucion(n, p, kmax):
    dim = 2**n; w = dim-1
    s = np.ones(dim)/np.sqrt(dim)
    Us = 2*np.outer(s, s) - np.eye(dim)
    Uw = np.eye(dim); Uw[w, w] = -1
    G = Us @ Uw
    canales = kraus_por_qubit(n, p)
    rho = np.outer(s, s).astype(complex)
    out = [rho.copy()]
    for _ in range(kmax):
        rho = G @ rho @ G.conj().T                      # unitario ideal
        for ops in canales:                              # canal en cada qubit
            rho = sum(K @ rho @ K.conj().T for K in ops)
        out.append(rho.copy())
    return out, w

def metricas(rhos, w):
    P = np.array([r[w, w].real for r in rhos])
    pureza = np.array([np.trace(r @ r).real for r in rhos])
    coh = np.array([np.abs(r).sum() - np.abs(np.diag(r)).sum() for r in rhos])
    traza_ok = max(abs(np.trace(r)-1) for r in rhos) < 1e-9
    herm_ok = max(np.abs(r-r.conj().T).max() for r in rhos) < 1e-9
    return P, pureza, coh, traza_ok and herm_ok

PS = [0, 0.01, 0.05, 0.1, 0.2]; NS = [4, 6]
filas, curvas = [], []
print("Validación caso ideal vs sin^2((2k+1)θ):")
for n in NS:
    N = 2**n; kmax = 2*int(np.floor(np.pi/4*np.sqrt(N))) + 2; kfirst = int(np.floor(np.pi/4*np.sqrt(N))) + 3
    th = np.arcsin(1/np.sqrt(N))
    for p in PS:
        rhos, w = evolucion(n, p, kmax)
        P, pur, coh, ok = metricas(rhos, w)
        assert ok, "falló traza/hermiticidad"
        if p == 0:
            err = np.abs(P - np.sin((2*np.arange(kmax+1)+1)*th)**2).max()
            print(f"  N={N}: error máximo = {err:.2e}")
        k = np.arange(kmax+1); E = k[1:]/np.maximum(P[1:], 1e-12)
        kopt = int(P[:kfirst].argmax())
        filas.append([N, p, kopt, round(P[kopt], 4), round(E.min(), 3), int(k[1:][E.argmin()]), round(1/N, 4), (N+1)/2])
        for kk in range(kmax+1):
            curvas.append([N, p, kk, P[kk], pur[kk], coh[kk]])

with open("tabla_resultados.csv", "w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["N","p","k_optimo","P_max","E_min_consultas","k_de_E_min","P_azar_1/N","E_clasico_(N+1)/2"]); w_.writerows(filas)
with open("curvas_k.csv", "w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["N","p","k","P_exito","pureza_Tr_rho2","coherencia_l1"]); w_.writerows(curvas)

print("\n N     p     k_opt  P_max   E_min   (clásico (N+1)/2)")
for r in filas: print(f"{r[0]:>3} {r[1]:>6} {r[2]:>5} {r[3]:>8.3f} {r[4]:>7.2f}   {r[7]}")

# Figura 1: P(k)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, n in zip(ax, NS):
    N = 2**n
    for p in PS:
        d = [c for c in curvas if c[0] == N and c[1] == p]
        a.plot([c[2] for c in d], [c[3] for c in d], marker="o", ms=3, label=f"p={p}")
    a.axhline(1/N, color="k", ls="--", lw=1, label="1/N"); a.set_title(f"N={N}")
    a.set_xlabel("k (iteraciones)"); a.set_ylabel("P_éxito(k)"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig1_P_vs_k.png", dpi=200); plt.close()

# Figura 2: coherencia y pureza vs k (N=64)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for p in PS:
    d = [c for c in curvas if c[0] == 64 and c[1] == p]
    ax[0].plot([c[2] for c in d], [c[5] for c in d], label=f"p={p}"); ax[1].plot([c[2] for c in d], [c[4] for c in d], label=f"p={p}")
ax[0].set_ylabel("coherencia (suma |ρ_ij|, i≠j)"); ax[1].set_ylabel("pureza Tr(ρ²)")
for a in ax: a.set_xlabel("k"); a.legend(fontsize=7)
fig.suptitle("N=64"); plt.tight_layout(); plt.savefig("fig2_coherencia_pureza.png", dpi=200); plt.close()

# Figura 3: min E vs p contra clásico
pg = np.concatenate([np.linspace(0, 0.1, 11), np.linspace(0.12, 0.6, 25)])
fig, ax = plt.subplots(figsize=(5.5, 3.8))
print()
for n in NS:
    N = 2**n; kmax = 2*int(np.floor(np.pi/4*np.sqrt(N))) + 2; ys = []
    for p in pg:
        rhos, w = evolucion(n, p, kmax); P = metricas(rhos, w)[0]
        ys.append((np.arange(1, kmax+1)/np.maximum(P[1:], 1e-12)).min())
    l, = ax.plot(pg, ys, label=f"Grover N={N}"); ax.axhline((N+1)/2, ls="--", color=l.get_color(), alpha=.5)
    cruce = next((p for p, y in zip(pg, ys) if y > (N+1)/2), None)
    print(f"N={N}: Grover deja de ganar en consultas esperadas (vs {(N+1)/2}) desde p ≈ {cruce}")
ax.set_yscale("log"); ax.set_xlabel("p por qubit, por iteración"); ax.set_ylabel("consultas esperadas mín."); ax.legend()
plt.tight_layout(); plt.savefig("fig3_Emin_vs_p.png", dpi=200)