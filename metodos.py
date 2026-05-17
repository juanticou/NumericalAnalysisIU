import numpy as np
import pandas as pd
import sympy as sp

# ==========================================
# 1. ECUACIONES NO LINEALES (Tres Errores)
# ==========================================

def biseccion(f, a, b, tol, max_iter):
    iteraciones = []
    x_ant = a
    for i in range(max_iter):
        xm = (a + b) / 2
        fxm = f(xm)
        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        e_cond = abs(fxm)  # Error de condición / residuo
        
        iteraciones.append({
            "Iter": i + 1, "x_n": xm, "f(x)": fxm, 
            "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond
        })
        if i > 0 and e_rel < tol: break
        
        if f(a) * fxm < 0: b = xm
        else: a = xm
        x_ant = xm
    return pd.DataFrame(iteraciones)

def regla_falsa(f, a, b, tol, max_iter):
    iteraciones = []
    x_ant = a
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        if fb - fa == 0: break
        xm = b - (fb * (b - a)) / (fb - fa)
        fxm = f(xm)
        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        e_cond = abs(fxm)
        
        iteraciones.append({
            "Iter": i + 1, "x_n": xm, "f(x)": fxm, 
            "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond
        })
        if i > 0 and e_rel < tol: break
        
        if fa * fxm < 0: b = xm
        else: a = xm
        x_ant = xm
    return pd.DataFrame(iteraciones)

def punto_fijo(g, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        try:
            xn = g(x_ant)
        except:
            break
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        e_cond = e_abs # En punto fijo no hay f(x) directa, el residuo es |x_n - g(x_n)|
        
        iteraciones.append({
            "Iter": i + 1, "x_n": xn, "f(x)": xn - x_ant, 
            "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond
        })
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

def newton(f, df, x0, tol, max_iter):
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        derivada = df(x_ant)
        
        # ⚠️ ALERTA: Validación de la derivada cercana a cero
        if abs(derivada) < 1e-12:
            raise ValueError(
                f"La derivada f'(x) se aproximó críticamente a cero ({derivada:.2e}) en x = {x_ant:.6f} (Iteración {i+1}). "
                f"El método no puede continuar porque divergerá o generará una división por cero."
            )
            
        xn = x_ant - f(x_ant) / derivada
        fxn = f(xn)
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        e_cond = abs(fxn)
        
        iteraciones.append({
            "Iter": i + 1, "x_n": xn, "f(x)": fxn, 
            "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond
        })
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)

def secante(f, x0, x1, tol, max_iter):
    iteraciones = []
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0: break
        xn = x1 - (f1 * (x1 - x0)) / (f1 - f0)
        fxn = f(xn)
        e_abs = abs(xn - x1)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        e_cond = abs(fxn)
        
        iteraciones.append({
            "Iter": i + 1, "x_n": xn, "f(x)": fxn, 
            "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond
        })
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
        fxn = f(xn)
        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        e_cond = abs(fxn)
        
        iteraciones.append({
            "Iter": i + 1, "x_n": xn, "f(x)": fxn, 
            "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond
        })
        if e_rel < tol: break
        x_ant = xn
    return pd.DataFrame(iteraciones)
# ==========================================
# 2. SISTEMAS LINEALES
# ==========================================

def calcular_radio_espectral(T):
    """Calcula el radio espectral de una matriz (máximo valor propio en módulo)"""
    valores_propios = np.linalg.eigvals(T)
    radio = max(abs(valores_propios))
    return radio

def jacobi(A, b, x0, tol, max_iter):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D) @ (L + U)
    C = np.linalg.inv(D) @ b
    
    radio_espectral = calcular_radio_espectral(T)   
    
    iteraciones = []
    x_ant = x0
    for i in range(max_iter):
        xn = T @ x_ant + C
        e_rel = np.linalg.norm(xn - x_ant, np.inf) / np.linalg.norm(xn, np.inf)
        e_abs = np.linalg.norm(xn - x_ant, np.inf)
        residuo = b - np.dot(A, xn)
        e_cond = np.linalg.norm(residuo, np.inf)
        iteraciones.append({"Iter": i+1, "E_Rel": e_rel, "E_Abs": e_abs, "E_Cond": e_cond})
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones), T, radio_espectral

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
        e_abs = np.linalg.norm(xn - x_ant, np.inf)
        residuo = b - np.dot(A, xn)
        e_cond = np.linalg.norm(residuo, np.inf)
        iteraciones.append({"Iter": i+1, "E_Rel": e_rel, "E_Abs": e_abs, "E_Cond": e_cond})
        
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones), T, calcular_radio_espectral(T)

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
        # (Esto va dentro del ciclo iterativo de tus funciones en metodos.py)
    
        e_abs = np.linalg.norm(xn - x_ant, np.inf) 
        norm_new = np.linalg.norm(xn, np.inf)
        e_rel = e_abs / norm_new if norm_new != 0 else e_abs
        
        # 2. NUEVO: Error de Condición (Norma del Residuo)
        # Ax_n se calcula multiplicando la matriz original A por tu x actual
        residuo = b - np.dot(A, xn)
        e_cond = np.linalg.norm(residuo, np.inf)
        
        # 3. Guardar los TRES en la fila
        iteraciones.append({"Iter": i + 1, "E_Abs": e_abs, "E_Rel": e_rel, "E_Cond": e_cond})
        if e_rel < tol: break
        x_ant = xn
    return xn, pd.DataFrame(iteraciones), T, calcular_radio_espectral(T)

# ==========================================
# 3. INTERPOLACIÓN Y ANÁLISIS DE ERROR
# ==========================================
from scipy.interpolate import CubicSpline, interp1d

def vandermonde(x_puntos, y_puntos):
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    n = len(x)
    V = np.vander(x, increasing=True)
    coefs = np.linalg.solve(V, y)
    
    t = sp.Symbol('x')
    polinomio = sum(coefs[i] * (t**i) for i in range(n))
    return sp.simplify(polinomio)

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

def spline_lineal(x_puntos, y_puntos):
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    
    # Modelo numérico ejecutable
    f_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
    
    # Construcción de ecuaciones por tramos para mostrar en LaTeX
    tramos_txt = []
    for i in range(len(x) - 1):
        m = (y[i+1] - y[i]) / (x[i+1] - x[i])
        b = y[i] - m * x[i]
        t = sp.Symbol('x')
        eq = sp.simplify(m*t + b)
        tramos_txt.append(f"x \\in [{x[i]}, {x[i+1]}]: {sp.latex(eq)}")
        
    return f_interp, tramos_txt

def spline_cubico(x_puntos, y_puntos):
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    
    cs = CubicSpline(x, y, bc_type='not-a-knot')
    
    t = sp.Symbol('x')
    tramos_txt = []
    for i in range(len(x) - 1):
        a, b, c, d = cs.c[:, i]
        p = sp.simplify(a*(t - x[i])**3 + b*(t - x[i])**2 + c*(t - x[i]) + d)
        tramos_txt.append(f"x \\in [{x[i]}, {x[i+1]}]: {sp.latex(p)}")
        
    return cs, tramos_txt