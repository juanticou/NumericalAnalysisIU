import numpy as np
import pandas as pd

# ==========================================
# 1. ECUACIONES NO LINEALES
# ==========================================

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0: return None
    iteraciones = []
    x_ant = a
    for i in range(max_iter):
        xm = (a + b) / 2
        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        
        iteraciones.append({"Iter": i+1, "x_n": xm, "f(x)": f(xm), "E_Abs": e_abs, "E_Rel": e_rel})
        
        # Criterio de parada con Error Relativo
        if i > 0 and e_rel < tol: break
        
        if f(a) * f(xm) < 0: b = xm
        else: a = xm
        x_ant = xm
    return pd.DataFrame(iteraciones)

def regla_falsa(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0: return None
    iteraciones = []
    x_ant = a
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        xm = b - (fb * (b - a)) / (fb - fa)
        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        
        iteraciones.append({"Iter": i+1, "x_n": xm, "f(x)": f(xm), "E_Abs": e_abs, "E_Rel": e_rel})
        
        if i > 0 and e_rel < tol: break
        
        if f(a) * f(xm) < 0: b = xm
        else: a = xm
        x_ant = xm
    return pd.DataFrame(iteraciones)

def punto_fijo(g, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        xn = g(x_ant)
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        
        iteraciones.append({"Iter": i+1, "x_n": xn, "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

def newton(f, df, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        dfx = df(x_ant)
        if dfx == 0: break
        xn = x_ant - f(x_ant) / dfx
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        
        iteraciones.append({"Iter": i+1, "x_n": xn, "f(x)": f(xn), "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

def secante(f, x0, x1, tol, max_iter):
    iteraciones = []
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0: break
        xn = x1 - (f1 * (x1 - x0)) / (f1 - f0)
        e_abs = abs(xn - x1)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        
        iteraciones.append({"Iter": i+1, "x_n": xn, "f(x)": f(xn), "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x0, x1 = x1, xn
    return pd.DataFrame(iteraciones)

def raices_multiples(f, df, ddf, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        fx, dfx, ddfx = f(x_ant), df(x_ant), ddf(x_ant)
        denominador = (dfx**2) - (fx * ddfx)
        if denominador == 0: break
        xn = x_ant - (fx * dfx) / denominador
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        
        iteraciones.append({"Iter": i+1, "x_n": xn, "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

# ==========================================
# 2. SISTEMAS DE ECUACIONES LINEALES
# ==========================================

def jacobi(A, b, x0, tol, max_iter):
    D = np.diag(np.diag(A))
    LU = A - D
    x_ant = x0
    iteraciones = []
    for i in range(max_iter):
        xn = np.linalg.solve(D, b - np.dot(LU, x_ant))
        e_abs = np.linalg.norm(xn - x_ant, np.inf)
        e_rel = e_abs / np.linalg.norm(xn, np.inf)
        
        iteraciones.append({"Iter": i+1, "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones)

def gauss_seidel(A, b, x0, tol, max_iter):
    L = np.tril(A)
    U = A - L
    x_ant = x0
    iteraciones = []
    for i in range(max_iter):
        xn = np.linalg.solve(L, b - np.dot(U, x_ant))
        e_abs = np.linalg.norm(xn - x_ant, np.inf)
        e_rel = e_abs / np.linalg.norm(xn, np.inf)
        
        iteraciones.append({"Iter": i+1, "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones)

def sor(A, b, x0, tol, max_iter, w):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    Tw = np.linalg.inv(D + w*L) @ ((1-w)*D - w*U)
    Cw = w * np.linalg.inv(D + w*L) @ b
    x_ant = x0
    iteraciones = []
    for i in range(max_iter):
        xn = Tw @ x_ant + Cw
        e_abs = np.linalg.norm(xn - x_ant, np.inf)
        e_rel = e_abs / np.linalg.norm(xn, np.inf)
        
        iteraciones.append({"Iter": i+1, "E_Abs": e_abs, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones)

# ==========================================
# 3. INTERPOLACIÓN (Se mantienen igual ya que no son iterativos por tol)
# ==========================================

def vandermonde(x, y):
    A = np.vander(x, increasing=True)
    return np.linalg.solve(A, y)

def lagrange(x_puntos, y_puntos):
    import sympy as sp
    t = sp.Symbol('x')
    polinomio = 0
    for i in range(len(x_puntos)):
        L = 1
        for j in range(len(x_puntos)):
            if i != j:
                L *= (t - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        polinomio += y_puntos[i] * L
    return sp.simplify(polinomio)

def newton_interpolante(x, y):
    n = len(y)
    tabla = np.zeros([n, n])
    tabla[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x[i+j] - x[i])
    return tabla[0, :]

def spline_cubico(x, y):
    from scipy.interpolate import CubicSpline
    return CubicSpline(x, y)