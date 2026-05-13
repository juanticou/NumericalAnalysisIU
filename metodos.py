import numpy as np
import pandas as pd
import sympy as sp

# ==========================================
# 1. ECUACIONES NO LINEALES (Lógica Flexible)
# ==========================================

def biseccion(f, a, b, tol, max_iter):
    iteraciones = []
    x_ant = a
    for i in range(max_iter):
        xm = (a + b) / 2
        fxm = f(xm)
        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        
        iteraciones.append({
            "Iter": i + 1, 
            "x_n": xm, 
            "f(x)": fxm, 
            "E_Abs": e_abs, 
            "E_Rel": e_rel
        })
        
        if i > 0 and e_rel < tol: 
            break
        
        # Lógica de decisión de intervalo (Flexible: corre aunque no haya cambio de signo)
        if f(a) * fxm < 0:
            b = xm
        else:
            a = xm
        x_ant = xm
    return pd.DataFrame(iteraciones)

def regla_falsa(f, a, b, tol, max_iter):
    iteraciones = []
    x_ant = a
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        # Evitar división por cero si f(a) == f(b)
        if fb - fa == 0: break
        
        xm = b - (fb * (b - a)) / (fb - fa)
        fxm = f(xm)
        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        
        iteraciones.append({
            "Iter": i + 1, 
            "x_n": xm, 
            "f(x)": fxm, 
            "E_Abs": e_abs, 
            "E_Rel": e_rel
        })
        
        if i > 0 and e_rel < tol: break
        
        if fa * fxm < 0:
            b = xm
        else:
            a = xm
        x_ant = xm
    return pd.DataFrame(iteraciones)

def punto_fijo(g, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        xn = g(x_ant)
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        
        iteraciones.append({"Iter": i + 1, "x_n": xn, "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

def newton(f, df, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        derivada = df(x_ant)
        if derivada == 0: break
        
        xn = x_ant - f(x_ant) / derivada
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        
        iteraciones.append({"Iter": i + 1, "x_n": xn, "f(x)": f(xn), "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

def secante(f, x0, x1, tol, max_iter):
    iteraciones = []
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0: break
        
        xn = x1 - (f1 * (x1 - x0)) / (f1 - f0)
        e_rel = abs(xn - x1) / abs(xn) if xn != 0 else abs(xn - x1)
        
        iteraciones.append({"Iter": i + 1, "x_n": xn, "f(x)": f(xn), "E_Rel": e_rel})
        
        if e_rel < tol: break
        x0, x1 = x1, xn
    return pd.DataFrame(iteraciones)

def raices_multiples(f, df, ddf, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        fv, dfv, ddfv = f(x_ant), df(x_ant), ddf(x_ant)
        denominador = (dfv**2) - (fv * ddfv)
        if denominador == 0: break
        
        xn = x_ant - (fv * dfv) / denominador
        e_rel = abs(xn - x_ant) / abs(xn) if xn != 0 else abs(xn - x_ant)
        
        iteraciones.append({"Iter": i + 1, "x_n": xn, "f(x)": f(xn), "E_Rel": e_rel})
        
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

# ==========================================
# 2. SISTEMAS LINEALES
# ==========================================

def jacobi(A, b, x0, tol, max_iter):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D) @ (L + U)
    C = np.linalg.inv(D) @ b
    
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        xn = T @ x_ant + C
        e_rel = np.linalg.norm(xn - x_ant, np.inf) / np.linalg.norm(xn, np.inf)
        iteraciones.append({"Iter": i+1, "E_Rel": e_rel})
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones)

def gauss_seidel(A, b, x0, tol, max_iter):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D - L) @ U
    C = np.linalg.inv(D - L) @ b
    
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        xn = T @ x_ant + C
        e_rel = np.linalg.norm(xn - x_ant, np.inf) / np.linalg.norm(xn, np.inf)
        iteraciones.append({"Iter": i+1, "E_Rel": e_rel})
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones)

def sor(A, b, x0, tol, max_iter, w):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D - w*L) @ ((1-w)*D + w*U)
    C = w * np.linalg.inv(D - w*L) @ b
    
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        xn = T @ x_ant + C
        e_rel = np.linalg.norm(xn - x_ant, np.inf) / np.linalg.norm(xn, np.inf)
        iteraciones.append({"Iter": i+1, "E_Rel": e_rel})
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones)

# ==========================================
# 3. INTERPOLACIÓN (Retornando SymPy)
# ==========================================

def vandermonde(x_puntos, y_puntos):
    n = len(x_puntos)
    A = np.vander(x_puntos, increasing=True)
    coefs = np.linalg.solve(A, y_puntos)
    t = sp.Symbol('x')
    polinomio = sum(coefs[i] * t**i for i in range(n))
    return sp.expand(polinomio)

def lagrange(x_puntos, y_puntos):
    t = sp.Symbol('x')
    polinomio = 0
    n = len(x_puntos)
    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L *= (t - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        polinomio += y_puntos[i] * L
    return sp.simplify(polinomio)

def newton_interpolante(x_puntos, y_puntos):
    n = len(y_puntos)
    tabla = np.zeros([n, n])
    tabla[:,0] = y_puntos
    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x_puntos[i+j] - x_puntos[i])
    
    coefs = tabla[0, :]
    t = sp.Symbol('x')
    polinomio = coefs[0]
    acumulado = 1
    for i in range(1, n):
        acumulado *= (t - x_puntos[i-1])
        polinomio += coefs[i] * acumulado
    return sp.simplify(polinomio)

def spline_cubico(x_puntos, y_puntos):
    from scipy.interpolate import CubicSpline
    import sympy as sp
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    cs = CubicSpline(x, y, bc_type='not-a-knot')
    
    # Generar representación en texto por tramos
    t = sp.Symbol('x')
    tramos = []
    for i in range(len(x) - 1):
        a, b, c, d = cs.c[:, i]
        p = sp.simplify(a*(t - x[i])**3 + b*(t - x[i])**2 + c*(t - x[i]) + d)
        tramos.append(f"x \in [{x[i]}, {x[i+1]}]: {sp.latex(p)}")
    return cs, tramos