import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metodos as m  # Importamos tu archivo de lógica

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Numerical Analysis Toolbox", layout="wide")

# Estilo personalizado para mejorar la apariencia
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
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

    # --- INPUTS PARA ECUACIONES NO LINEALES ---
    if categoria == "Ecuaciones No Lineales":
        with col_left:
            f_input = st.text_input("Función f(x)", "x**2 - np.log(x) - 5", help="Usa sintaxis de numpy, ej: np.exp(x), np.sin(x)")
            f = lambda x: eval(f_input, {"np": np, "x": x})
            
            if metodo in ["Bisección", "Regla Falsa"]:
                a = st.number_input("Extremo inferior (a)", value=1.0, format="%.4f")
                b = st.number_input("Extremo superior (b)", value=2.0, format="%.4f")
            elif metodo == "Secante":
                x0 = st.number_input("x0 (primer punto)", value=1.0)
                x1 = st.number_input("x1 (segundo punto)", value=2.0)
            else:
                x0 = st.number_input("Valor inicial x0", value=1.0)
                
            if metodo == "Punto Fijo":
                g_input = st.text_input("Función g(x)", "np.sqrt(np.log(x) + 5)")
                g = lambda x: eval(g_input, {"np": np, "x": x})
            
            if metodo in ["Newton", "Raíces Múltiples"]:
                df_input = st.text_input("Derivada f'(x)", "2*x - 1/x")
                df = lambda x: eval(df_input, {"np": np, "x": x})
            
            if metodo == "Raíces Múltiples":
                ddf_input = st.text_input("Segunda derivada f''(x)", "2 + 1/x**2")
                ddf = lambda x: eval(ddf_input, {"np": np, "x": x})

        with col_right:
            tol = st.number_input("Tolerancia (Error Relativo)", value=1e-7, format="%.e")
            max_iter = st.number_input("Máximo de iteraciones", value=100, step=1)

    # --- INPUTS PARA SISTEMAS LINEALES ---
    elif categoria == "Sistemas Lineales":
        dim = st.number_input("Dimensión del sistema (n)", min_value=2, max_value=10, value=3)
        st.write("Edita la Matriz A y el Vector b:")
        c1, c2 = st.columns([3, 1])
        with c1:
            A = st.data_editor(pd.DataFrame(np.eye(dim)), key="mat_A").to_numpy()
        with c2:
            b = st.data_editor(pd.DataFrame(np.zeros(dim), columns=["b"]), key="vec_b").to_numpy().flatten()
        
        with col_left:
            tol_sist = st.number_input("Tolerancia", value=1e-7, format="%.e")
            w = st.slider("Factor de relajación (w)", 0.1, 1.9, 1.0) if metodo == "SOR" else 1.0
        with col_right:
            max_i_sist = st.number_input("Máximo iteraciones", value=100)

    # --- INPUTS PARA INTERPOLACIÓN ---
    elif categoria == "Interpolación":
        st.write("Puntos de datos (x, y):")
        df_puntos = st.data_editor(pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 5.0, 3.0]}), num_rows="dynamic")
        xp, yp = df_puntos['x'].values, df_puntos['y'].values

# --- LÓGICA DE EJECUCIÓN ---
if st.button("🚀 Calcular"):
    try:
        res = None
        # Selección de método y ejecución
        if categoria == "Ecuaciones No Lineales":
            if metodo == "Bisección": res = m.biseccion(f, a, b, tol, max_iter)
            elif metodo == "Regla Falsa": res = m.regla_falsa(f, a, b, tol, max_iter)
            elif metodo == "Punto Fijo": res = m.punto_fijo(g, x0, tol, max_iter)
            elif metodo == "Newton": res = m.newton(f, df, x0, tol, max_iter)
            elif metodo == "Secante": res = m.secante(f, x0, x1, tol, max_iter)
            elif metodo == "Raíces Múltiples": res = m.raices_multiples(f, df, ddf, x0, tol, max_iter)
            
            if res is not None:
                st.success(f"Convergencia alcanzada en {len(res)} iteraciones.")
                t1, t2 = st.tabs(["📊 Gráfica de Convergencia", "📋 Tabla de Datos"])
                with t1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(res["Iter"], res["E_Rel"], label="Error Relativo", marker='o', color='#ff4b4b')
                    ax.plot(res["Iter"], res["E_Abs"], label="Error Absoluto", linestyle='--', color='#1f77b4')
                    ax.set_yscale('log')
                    ax.set_xlabel("Iteración")
                    ax.set_ylabel("Error (log scale)")
                    ax.grid(True, which="both", alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)
                with t2:
                    st.dataframe(res.style.format({"x_n": "{:.8f}", "f(x)": "{:.2e}", "E_Abs": "{:.2e}", "E_Rel": "{:.2e}"}), use_container_width=True)

        elif categoria == "Sistemas Lineales":
            x0_sist = np.zeros(len(b))
            if metodo == "Jacobi": sol, res = m.jacobi(A, b, x0_sist, tol_sist, max_i_sist)
            elif metodo == "Gauss-Seidel": sol, res = m.gauss_seidel(A, b, x0_sist, tol_sist, max_i_sist)
            elif metodo == "SOR": sol, res = m.sor(A, b, x0_sist, tol_sist, max_i_sist, w)
            
            st.write("### ✅ Solución Aproximada:")
            st.json({f"x{i}": round(v, 8) for i, v in enumerate(sol)})
            
            t1, t2 = st.tabs(["📉 Evolución del Error", "📋 Tabla de Iteraciones"])
            with t1:
                st.line_chart(res[["E_Rel", "E_Abs"]])
            with t2:
                st.dataframe(res.style.format("{:.2e}"))

        elif categoria == "Interpolación":
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(xp, yp, color='black', zorder=5, label="Puntos")
            x_plot = np.linspace(min(xp), max(xp), 200)
            
            if metodo == "Vandermonde":
                coefs = m.vandermonde(xp, yp)
                y_plot = np.polyval(coefs[::-1], x_plot)
                st.info(f"Coeficientes del polinomio (menor a mayor grado): {coefs}")
            elif metodo == "Lagrange":
                pol = m.lagrange(xp, yp)
                st.latex(f"P(x) = {pol}")
                y_plot = [float(pol.subs('x', val)) for val in x_plot]
            elif metodo == "Newton Interpolante":
                coefs = m.newton_interpolante(xp, yp)
                st.write("Diferencias Divididas:", coefs)
                # Aquí iría evaluación Newton... por ahora usamos spline para visualizar
                y_plot = m.spline_cubico(xp, yp)(x_plot)
            elif metodo == "Spline Cubico":
                cs = m.spline_cubico(xp, yp)
                y_plot = cs(x_plot)

            ax.plot(x_plot, y_plot, label=f"Interpolación {metodo}", color="#ff4b4b")
            ax.legend()
            st.pyplot(fig)

    except ValueError as ve:
        st.error(f"❌ Error de Validación: {ve}")
    except Exception as e:
        st.error(f"⚠️ Error inesperado: {e}")
        st.info("Sugerencia: Verifica que la función esté definida en el intervalo y que no haya divisiones por cero.")