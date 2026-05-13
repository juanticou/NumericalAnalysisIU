import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
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

# --- BARRA LATERAL: SELECCIÓN ---
st.sidebar.title("🛠️ Configuración")
categoria = st.sidebar.selectbox("Categoría de Métodos", ["Ecuaciones No Lineales", "Sistemas Lineales", "Interpolación"])

menus = {
    "Ecuaciones No Lineales": ["Bisección", "Regla Falsa", "Punto Fijo", "Newton", "Secante", "Raíces Múltiples"],
    "Sistemas Lineales": ["Jacobi", "Gauss-Seidel", "SOR"],
    "Interpolación": ["Vandermonde", "Lagrange", "Newton Interpolante", "Spline Cubico"]
}

metodo = st.sidebar.selectbox("Método Específico", menus[categoria])
st.sidebar.divider()

# --- PANEL PRINCIPAL ---
st.title(f"Análisis Numérico: {metodo}")

# Contenedor para parámetros
with st.expander("⚙️ Parámetros de Entrada", expanded=True):
    col_left, col_right = st.columns(2)

    # --- LÓGICA PARA ECUACIONES NO LINEALES ---
    if categoria == "Ecuaciones No Lineales":
        with col_left:
            f_input = st.text_input("Función f(x)", "x**2 - np.log(x) - 5", help="Puedes usar 'x**2', 'np.sin(x)', 'exp(x)', etc.")
            
            # DERIVACIÓN AUTOMÁTICA CON SYMPY
            try:
                x_sym = sp.Symbol('x')
                # Limpieza de string para que SymPy lo entienda
                f_clean = f_input.replace("np.", "")
                expr = sp.sympify(f_clean)
                
                # Derivadas simbólicas
                df_expr = sp.diff(expr, x_sym)
                ddf_expr = sp.diff(df_expr, x_sym)
                
                # Conversión a funciones de NumPy para el motor matemático
                f = sp.lambdify(x_sym, expr, 'numpy')
                df = sp.lambdify(x_sym, df_expr, 'numpy')
                ddf = sp.lambdify(x_sym, ddf_expr, 'numpy')
                
                st.info("**Información Simbólica:**")
                st.latex(f"f'(x) = {sp.latex(df_expr)}")
                if metodo == "Raíces Múltiples":
                    st.latex(f"f''(x) = {sp.latex(ddf_expr)}")
            except Exception as e:
                st.error(f"Error procesando la función: {e}")

            # Inputs de intervalos/puntos iniciales
            if metodo in ["Bisección", "Regla Falsa"]:
                a = st.number_input("Extremo a", value=2.0, format="%.4f")
                b = st.number_input("Extremo b", value=3.0, format="%.4f")
            elif metodo == "Secante":
                x0 = st.number_input("x0", value=1.0)
                x1 = st.number_input("x1", value=2.0)
            else:
                x0 = st.number_input("Punto inicial x0", value=1.0)
                
            if metodo == "Punto Fijo":
                g_input = st.text_input("Función g(x)", "np.sqrt(np.log(x) + 5)")
                g = lambda x: eval(g_input, {"np": np, "x": x})

        with col_right:
            tol = st.number_input("Tolerancia (Error Relativo)", value=1e-7, format="%.e")
            max_iter = st.number_input("Máximo iteraciones", value=100, step=1)

    # --- LÓGICA PARA SISTEMAS LINEALES ---
    elif categoria == "Sistemas Lineales":
        dim = st.number_input("Dimensión (n)", min_value=2, max_value=10, value=3)
        st.write("Edita la Matriz A y el Vector b:")
        c1, c2 = st.columns([3, 1])
        with c1:
            A = st.data_editor(pd.DataFrame(np.eye(dim)), key="mat_A").to_numpy()
        with c2:
            b = st.data_editor(pd.DataFrame(np.zeros(dim), columns=["b"]), key="vec_b").to_numpy().flatten()
        
        with col_left:
            tol_sist = st.number_input("Tolerancia", value=1e-7, format="%.e")
            w = st.slider("Relajación (w)", 0.1, 1.9, 1.0) if metodo == "SOR" else 1.0
        with col_right:
            max_i_sist = st.number_input("Máximo iteraciones", value=100)

    # --- LÓGICA PARA INTERPOLACIÓN ---
    elif categoria == "Interpolación":
        st.write("Puntos (x, y):")
        df_puntos = st.data_editor(pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 5.0, 3.0]}), num_rows="dynamic")
        xp, yp = df_puntos['x'].values, df_puntos['y'].values

# --- EJECUCIÓN ---
if st.button("🚀 CALCULAR"):
    try:
        res = None
        if categoria == "Ecuaciones No Lineales":
            if metodo == "Bisección": res = m.biseccion(f, a, b, tol, max_iter)
            elif metodo == "Regla Falsa": res = m.regla_falsa(f, a, b, tol, max_iter)
            elif metodo == "Punto Fijo": res = m.punto_fijo(g, x0, tol, max_iter)
            elif metodo == "Newton": res = m.newton(f, df, x0, tol, max_iter)
            elif metodo == "Secante": res = m.secante(f, x0, x1, tol, max_iter)
            elif metodo == "Raíces Múltiples": res = m.raices_multiples(f, df, ddf, x0, tol, max_iter)
            
            if res is not None:
                st.success(f"Convergencia lograda en la iteración {len(res)}")
                tab1, tab2 = st.tabs(["📊 Gráfica de Convergencia", "📋 Tabla de Datos"])
                with tab1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(res["Iter"], res["E_Rel"], label="E. Relativo", marker='o', color='#ff4b4b')
                    ax.plot(res["Iter"], res["E_Abs"], label="E. Absoluto", linestyle='--', color='#1f77b4')
                    ax.set_yscale('log')
                    ax.set_xlabel("Iteración")
                    ax.set_ylabel("Error (Escala Log)")
                    ax.grid(True, which="both", alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)
                with tab2:
                    st.dataframe(res.style.format("{:.2e}"), use_container_width=True)
                    csv = res.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar CSV", data=csv, file_name=f"{metodo}_results.csv")

        elif categoria == "Sistemas Lineales":
            x0_sist = np.zeros(len(b))
            if metodo == "Jacobi": sol, res = m.jacobi(A, b, x0_sist, tol_sist, max_i_sist)
            elif metodo == "Gauss-Seidel": sol, res = m.gauss_seidel(A, b, x0_sist, tol_sist, max_i_sist)
            elif metodo == "SOR": sol, res = m.sor(A, b, x0_sist, tol_sist, max_i_sist, w)
            
            st.subheader("✅ Solución Encontrada:")
            st.write(sol)
            st.line_chart(res[["E_Rel", "E_Abs"]])
            st.dataframe(res)

        elif categoria == "Interpolación":
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(xp, yp, color='black', zorder=5, label="Puntos Originales")
            x_plot = np.linspace(min(xp), max(xp), 200)
            
            polinomio_visual = None
            
            if metodo == "Vandermonde":
                polinomio_visual = m.vandermonde(xp, yp)
                f_num = sp.lambdify(sp.Symbol('x'), polinomio_visual, 'numpy')
                y_plot = f_num(x_plot)
                
            elif metodo == "Lagrange":
                polinomio_visual = m.lagrange(xp, yp)
                f_num = sp.lambdify(sp.Symbol('x'), polinomio_visual, 'numpy')
                y_plot = [float(polinomio_visual.subs(sp.Symbol('x'), v)) for v in x_plot]
                
            elif metodo == "Newton Interpolante":
                polinomio_visual = m.newton_interpolante(xp, yp)
                f_num = sp.lambdify(sp.Symbol('x'), polinomio_visual, 'numpy')
                y_plot = f_num(x_plot)
                
            elif metodo == "Spline Cubico":
                cs = m.spline_cubico(xp, yp)
                y_plot = cs(x_plot)
                st.info("Nota: El Spline Cúbico genera múltiples polinomios por tramos.")

            # --- IMPRESIÓN DEL POLINOMIO ---
            if polinomio_visual is not None:
                st.subheader("📝 Polinomio Interpolante Encontrado:")
                st.latex(f"P(x) = {sp.latex(polinomio_visual)}")
                
                with st.expander("Ver coeficientes expandidos"):
                    st.write(sp.expand(polinomio_visual))

            # --- GRÁFICA ---
            ax.plot(x_plot, y_plot, color='#ff4b4b', label=f"Curva {metodo}")
            ax.set_title(f"Resultado de Interpolación por {metodo}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    except ValueError as ve:
        st.error(f"❌ Error de Validación: {ve}")
    except Exception as e:
        st.error(f"⚠️ Error Crítico: {e}")