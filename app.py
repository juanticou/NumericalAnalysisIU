import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import metodos as m 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Numerical Analysis Toolbox", layout="wide")

# Estilo personalizado
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border-radius: 8px; }
    .stDataFrame { border: 1px solid #e6e9ef; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- BARRA LATERAL ---
st.sidebar.title("🛠️ Configuración")
categoria = st.sidebar.selectbox("Categoría", ["Ecuaciones No Lineales", "Sistemas Lineales", "Interpolación"])

menus = {
    "Ecuaciones No Lineales": ["Bisección", "Regla Falsa", "Punto Fijo", "Newton", "Secante", "Raíces Múltiples"],
    "Sistemas Lineales": ["Jacobi", "Gauss-Seidel", "SOR"],
    "Interpolación": ["Vandermonde", "Lagrange", "Newton Interpolante", "Spline Cubico"]
}

metodo = st.sidebar.selectbox("Método", menus[categoria])
st.sidebar.divider()

st.title(f"🚀 {metodo}")

# --- PARÁMETROS DE ENTRADA ---
with st.expander("⚙️ Parámetros de Entrada", expanded=True):
    col_left, col_right = st.columns(2)

    if categoria == "Ecuaciones No Lineales":
        with col_left:
            f_input = st.text_input("Función f(x)", "x**2 - 5", key="f_input_nl")
            try:
                x_sym = sp.Symbol('x')
                locs = {'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp, 
                        'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E}
                f_expr = sp.parse_expr(f_input.replace('^', '**'), local_dict=locs)
                df_expr = sp.diff(f_expr, x_sym)
                ddf_expr = sp.diff(df_expr, x_sym)
                f = sp.lambdify(x_sym, f_expr, 'numpy')
                df = sp.lambdify(x_sym, df_expr, 'numpy')
                ddf = sp.lambdify(x_sym, ddf_expr, 'numpy')
                st.info("**Interpretación Matemática:**")
                st.latex(rf"f(x) = {sp.latex(f_expr)}")
            except Exception as e:
                st.error(f"Error en f(x): {e}")

            if metodo in ["Bisección", "Regla Falsa"]:
                a = st.number_input("Extremo a", value=1.0)
                b = st.number_input("Extremo b", value=2.0)
            elif metodo == "Secante":
                x0 = st.number_input("x0", value=1.0)
                x1 = st.number_input("x1", value=2.0)
            else:
                x0 = st.number_input("Punto inicial x0", value=1.0)

        with col_right:
            tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-7, format="%.e")
            max_iter = st.number_input("Máximo Iteraciones", min_value=1, value=100)

    elif categoria == "Sistemas Lineales":
        dim = st.number_input("Dimensión", min_value=2, value=3)
        st.write("Matriz A:")
        A = st.data_editor(pd.DataFrame(np.eye(dim)), key="matriz_A_editor").to_numpy()
        st.write("Vector b:")
        b_vals = st.data_editor(pd.DataFrame(np.zeros(dim), columns=["b"]), key="vector_b_editor").to_numpy().flatten()
        with col_left:
            tol_sist = st.number_input("Tolerancia", min_value=1e-15, value=1e-7, format="%.e")
            w = st.number_input("w", min_value=0.01, max_value=1.99, value=1.0) if metodo == "SOR" else 1.0
        with col_right:
            max_i_sist = st.number_input("Max Iter", min_value=1, value=100)

    elif categoria == "Interpolación":
        st.write("### 1. Puntos de la función")
        df_puntos = st.data_editor(
            pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 5.0, 3.0]}), 
            num_rows="dynamic", 
            key="tabla_puntos_interp"
        )
        xp, yp = df_puntos['x'].values, df_puntos['y'].values

# --- LÓGICA DE EJECUCIÓN ---

# --- CASO ESPECIAL: INTERPOLACIÓN (Tiempo Real) ---
if categoria == "Interpolación":
    try:
        if len(xp) > 1 and len(set(xp)) == len(xp):
            # Cálculo automático
            if metodo == "Vandermonde": modelo = m.vandermonde(xp, yp)
            elif metodo == "Lagrange": modelo = m.lagrange(xp, yp)
            elif metodo == "Newton Interpolante": modelo = m.newton_interpolante(xp, yp)
            elif metodo == "Spline Cubico": modelo, tramos_txt = m.spline_cubico(xp, yp)

            # Mostrar Polinomio
            st.write("### 2. Expresión Matemática")
            if metodo == "Spline Cubico":
                for t in tramos_txt: st.latex(t)
            else:
                st.latex(rf"P(x) = {sp.latex(modelo)}")

            st.divider()
            col_ev, col_gr = st.columns([1, 2])

            with col_ev:
                st.write("### 3. Evaluar f(x)")
                x_eval = st.number_input("Punto x:", value=float(xp[0]), format="%.4f", key="input_x_eval_interp")
                if metodo == "Spline Cubico":
                    y_eval = float(modelo(x_eval))
                else:
                    f_num = sp.lambdify(sp.Symbol('x'), modelo, 'numpy')
                    y_eval = float(f_num(x_eval))
                st.metric(label=f"Resultado f({x_eval})", value=f"{y_eval:.6f}")

            with col_gr:
                fig, ax = plt.subplots(figsize=(8, 5))
                x_min, x_max = min(min(xp), x_eval) - 0.5, max(max(xp), x_eval) + 0.5
                x_range = np.linspace(x_min, x_max, 500)
                if metodo == "Spline Cubico":
                    ax.plot(x_range, modelo(x_range), color='#4b4bff', label="Spline", linewidth=2)
                else:
                    f_plt = sp.lambdify(sp.Symbol('x'), modelo, 'numpy')
                    ax.plot(x_range, f_plt(x_range), color='#ff4b4b', label="Polinomio", linewidth=2)
                ax.scatter(xp, yp, color='black', s=60, label="Puntos Base", zorder=5)
                ax.scatter([x_eval], [y_eval], color='green', s=120, edgecolors='white', label="Evaluación", zorder=6)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.warning("⚠️ Asegúrate de que las X sean únicas y haya al menos 2 puntos.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- OTROS MÉTODOS (Requieren Botón) ---
elif st.button("🚀 EJECUTAR CÁLCULO"):
    try:
        res = None
        if categoria == "Ecuaciones No Lineales":
            if metodo == "Bisección": res = m.biseccion(f, a, b, tol, max_iter)
            elif metodo == "Regla Falsa": res = m.regla_falsa(f, a, b, tol, max_iter)
            elif metodo == "Punto Fijo": res = m.punto_fijo(g, x0, tol, max_iter)
            elif metodo == "Newton": res = m.newton(f, df, x0, tol, max_iter)
            elif metodo == "Secante": res = m.secante(f, x0, x1, tol, max_iter)
            elif metodo == "Raíces Múltiples": res = m.raices_multiples(f, df, ddf, x0, tol, max_iter)
            
            if res is not None and not res.empty:
                st.success("✅ Cálculo finalizado")
                st.dataframe(res.style.format("{:.2e}"))
        
        elif categoria == "Sistemas Lineales":
            x0_s = np.zeros(dim)
            if metodo == "Jacobi": sol, res = m.jacobi(A, b_vals, x0_s, tol_sist, max_i_sist)
            elif metodo == "Gauss-Seidel": sol, res = m.gauss_seidel(A, b_vals, x0_s, tol_sist, max_i_sist)
            elif metodo == "SOR": sol, res = m.sor(A, b_vals, x0_s, tol_sist, max_i_sist, w)
            
            st.subheader("✅ Solución:")
            st.write(sol)
            st.dataframe(res)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")