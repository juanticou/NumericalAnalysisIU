import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import metodos as m  # Asegúrate de tener metodos.py actualizado en la misma carpeta

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

    # --- INPUTS: ECUACIONES NO LINEALES ---
    if categoria == "Ecuaciones No Lineales":
        with col_left:
            f_input = st.text_input("Función f(x)", "x**2 - 5", key="f_input_nl")
            try:
                x_sym = sp.Symbol('x')
                locs = {'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp, 
                        'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E}
                f_expr = sp.parse_expr(f_input.replace('^', '**'), local_dict=locs)
                
                # Generamos derivadas automáticas con SymPy
                df_expr = sp.diff(f_expr, x_sym)
                ddf_expr = sp.diff(df_expr, x_sym)
                
                st.info("**Interpretación Matemática:**")
                st.latex(rf"f(x) = {sp.latex(f_expr)}")
            except Exception as e:
                st.error(f"Error al interpretar f(x): {e}")

            if metodo in ["Bisección", "Regla Falsa"]:
                a = st.number_input("Extremo a", value=1.0, key="nl_a")
                b = st.number_input("Extremo b", value=2.0, key="nl_b")
            elif metodo == "Secante":
                x0 = st.number_input("Punto x0", value=1.0, key="nl_x0_sec")
                x1 = st.number_input("Punto x1", value=2.0, key="nl_x1_sec")
            else:
                x0 = st.number_input("Punto inicial x0", value=1.0, key="nl_x0")

            if metodo == "Punto Fijo":
                g_input = st.text_input("Función g(x)", "sqrt(5)", key="g_input_pf")
                try:
                    g_expr = sp.parse_expr(g_input.replace('^', '**'), local_dict=locs)
                    st.latex(rf"g(x) = {sp.latex(g_expr)}")
                except Exception as e:
                    st.error(f"Error en g(x): {e}")

        with col_right:
            tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-7, format="%.e", key="nl_tol")
            max_iter = st.number_input("Máximo Iteraciones", min_value=1, value=100, key="nl_max_iter")

    # --- INPUTS: SISTEMAS LINEALES ---
    elif categoria == "Sistemas Lineales":
        dim = st.number_input("Dimensión de la matriz (n)", min_value=2, max_value=20, value=3, key="sist_dim")
        
        st.write("Matriz de Coeficientes (A):")
        df_A = st.data_editor(pd.DataFrame(np.eye(dim)), key="matriz_A_editor")
        
        st.write("Vector de Términos Independientes (b):")
        df_b = st.data_editor(pd.DataFrame(np.zeros(dim), columns=["b"]), key="vector_b_editor")
        
        with col_left:
            tol_sist = st.number_input("Tolerancia", min_value=1e-15, value=1e-7, format="%.e", key="sist_tol")
            w = st.number_input("Factor de relajación (w)", min_value=0.01, max_value=1.99, value=1.0, key="sist_w") if metodo == "SOR" else 1.0
        with col_right:
            max_i_sist = st.number_input("Máximo de Iteraciones", min_value=1, value=100, key="sist_max_i")

    # --- INPUTS: INTERPOLACIÓN ---
    # --- INPUTS: INTERPOLACIÓN ---
    elif categoria == "Interpolación":
        st.write("### 1. Puntos de la Función Base (Datos Experimentales)")
        st.info("💡 Agrega al menos 5 o más puntos ordenados en X para analizar correctamente los porcentajes de error (10% al 40%).")
        
        # Tabla extendida inicial por defecto
        df_puntos = st.data_editor(
            pd.DataFrame({
                'x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                'y': [1.0, 2.7, 5.2, 3.1, 1.5, 4.3, 7.2]
            }), 
            num_rows="dynamic", 
            key="tabla_puntos_interp"
        )
        
        # Limpieza y ordenamiento de puntos
        df_puntos = df_puntos.dropna().sort_values(by='x')
        xp, yp = df_puntos['x'].values, df_puntos['y'].values
        
        # Modificación de menú lateral dinámico para incluir Spline Lineal
        menus["Interpolación"] = ["Vandermonde", "Lagrange", "Newton Interpolante", "Spline Lineal", "Spline Cubico"]

# --- LÓGICA DE EJECUCIÓN AISLADA ---

if categoria == "Interpolación":
    try:
        if len(xp) > 2 and len(set(xp)) == len(xp):
            # --- Ajustar Modelo con el 100% de los datos para visualización principal ---
            if metodo == "Vandermonde": modelo = m.vandermonde(xp, yp)
            elif metodo == "Lagrange": modelo = m.lagrange(xp, yp)
            elif metodo == "Newton Interpolante": modelo = m.newton_interpolante(xp, yp)
            elif metodo == "Spline Lineal": modelo, tramos_txt = m.spline_lineal(xp, yp)
            elif metodo == "Spline Cubico": modelo, tramos_txt = m.spline_cubico(xp, yp)

            st.write("### 2. Expresión Matemática del Polinomio / Curva Solución")
            if metodo in ["Spline Lineal", "Spline Cubico"]:
                for t in tramos_txt: st.latex(t)
            else:
                st.latex(rf"P(x) = {sp.latex(modelo)}")

            st.divider()
            col_gr, col_err = st.columns([1, 1])

            with col_gr:
                st.write("### 3. Gráfica del Ajuste (100% de Datos)")
                fig, ax = plt.subplots(figsize=(8, 5))
                x_range = np.linspace(min(xp) - 0.2, max(xp) + 0.2, 500)
                
                if metodo in ["Spline Lineal", "Spline Cubico"]:
                    ax.plot(x_range, modelo(x_range), color='#4b4bff', label=metodo, linewidth=2)
                else:
                    f_plt = sp.lambdify(sp.Symbol('x'), modelo, 'numpy')
                    ax.plot(x_range, f_plt(x_range), color='#ff4b4b', label=f"Polinomio {metodo}", linewidth=2)
                
                ax.scatter(xp, yp, color='black', s=60, label="Puntos Reales", zorder=5)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col_err:
                st.write("### 4. Análisis del Error por Almacenamiento/Prueba")
                
                porcentajes = [10, 20, 30, 40]
                errores_mse = []
                
                # Simulación de partición para cada porcentaje de testeo solicitado
                for pct in porcentajes:
                    n_test = max(1, int(np.ceil(len(xp) * (pct / 100.0))))
                    if len(xp) - n_test < 2: 
                        # Evitar quedarnos con menos de 2 puntos para interpolar
                        n_test = max(1, len(xp) - 3)
                    
                    # Seleccionamos índices distribuidos uniformemente para el test
                    idx_test = np.linspace(1, len(xp) - 2, n_test, dtype=int) if len(xp) > 3 else [1]
                    idx_train = [i for i in range(len(xp)) if i not in idx_test]
                    
                    x_train, y_train = xp[idx_train], yp[idx_train]
                    x_test, y_test = xp[idx_test], yp[idx_test]
                    
                    try:
                        # Entrenamos el modelo usando solo el remanente de datos (100 - pct)%
                        if metodo == "Vandermonde": mod_e = m.vandermonde(x_train, y_train)
                        elif metodo == "Lagrange": mod_e = m.lagrange(x_train, y_train)
                        elif metodo == "Newton Interpolante": mod_e = m.newton_interpolante(x_train, y_train)
                        elif metodo == "Spline Lineal": mod_e, _ = m.spline_lineal(x_train, y_train)
                        elif metodo == "Spline Cubico": mod_e, _ = m.spline_cubico(x_train, y_train)
                        
                        # Evaluación numérica en los datos almacenados/ocultos
                        if metodo in ["Spline Lineal", "Spline Cubico"]:
                            y_pred = mod_e(x_test)
                        else:
                            f_eval = sp.lambdify(sp.Symbol('x'), mod_e, 'numpy')
                            y_pred = f_eval(x_test)
                            
                        # Cálculo del Error Absoluto Medio (MAE)
                        mae = np.mean(abs((y_test - y_pred)))
                        errores_mse.append(mae)
                    except:
                        errores_mse.append(np.nan)
                
                # Gráfica Comparativa de Error vs Porcentaje de Almacenamiento
                fig_err, ax_err = plt.subplots(figsize=(8, 5))
                ax_err.plot(porcentajes, errores_mse, marker='o', linestyle='--', color='#e11d48', linewidth=2, label="MAE")
                ax_err.set_xlabel("Porcentaje de Datos Almacenados/Ocultos (%)", fontweight='bold')
                ax_err.set_ylabel("Error Absoluto Medio (MAE)", fontweight='bold')
                ax_err.set_xticks(porcentajes)
                ax_err.set_yscale('log') if any(e > 1e-3 for e in errores_mse if not np.isnan(e)) else None
                ax_err.grid(True, alpha=0.3)
                ax_err.legend()
                st.pyplot(fig_err)
                
                # Mostrar métricas en formato tabla
                df_error_res = pd.DataFrame({
                    "Datos Ocultados (%)": [f"{p}%" for p in porcentajes],
                    "Error Medio (MAE)": [f"{e:.5e}" if not np.isnan(e) else "N/A" for e in errores_mse]
                })
                st.dataframe(df_error_res, use_container_width=True)

        else:
            st.warning("⚠️ Asegúrate de que las coordenadas X sean totalmente únicas y que tengas por lo menos 3 puntos en la tabla.")
    except Exception as e:
        st.error(f"⚠️ Error en el procesamiento del método de interpolación: {e}")
# --- CASO 2: REQUERIR BOTÓN (Ecuaciones No Lineales y Sistemas Lineales) ---
else:
    if st.button("🚀 EJECUTAR CÁLCULO"):
        try:
            # --- CÁLCULO SÓLO SI SE SELECCIONÓ ECUACIONES NO LINEALES ---
            if categoria == "Ecuaciones No Lineales":
                # Compilamos numéricamente f, df y ddf justo aquí para evitar cruce de variables
                f = sp.lambdify(x_sym, f_expr, 'numpy')
                df = sp.lambdify(x_sym, df_expr, 'numpy')
                ddf = sp.lambdify(x_sym, ddf_expr, 'numpy')
                
                res = None
                if metodo == "Bisección": res = m.biseccion(f, a, b, tol, max_iter)
                elif metodo == "Regla Falsa": res = m.regla_falsa(f, a, b, tol, max_iter)
                elif metodo == "Newton": res = m.newton(f, df, x0, tol, max_iter)
                elif metodo == "Secante": res = m.secante(f, x0, x1, tol, max_iter)
                elif metodo == "Raíces Múltiples": res = m.raices_multiples(f, df, ddf, x0, tol, max_iter)
                elif metodo == "Punto Fijo":
                    g = sp.lambdify(x_sym, g_expr, 'numpy')
                    res = m.punto_fijo(g, x0, tol, max_iter)
                
                if res is not None and not res.empty:
                    st.success(f"✅ Cálculo finalizado con éxito en {len(res)} iteraciones.")
                    col_tabla, col_grafica = st.columns([1, 1])
                    
                    with col_tabla:
                        st.subheader("📋 Tabla de Iteraciones")
                        st.dataframe(res.style.format({
                            "x_n": "{:.6f}", "f(x)": "{:.2e}", 
                            "E_Abs": "{:.2e}", "E_Rel": "{:.2e}", "E_Cond": "{:.2e}"
                        }), use_container_width=True)
                    
                    with col_grafica:
                        st.subheader("📊 Comparación de Tipos de Errores")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(res["Iter"], res["E_Abs"], marker='o', label="Error Absoluto ($E_{abs}$)", color='#ff4b4b', lw=2)
                        ax.plot(res["Iter"], res["E_Rel"], marker='s', label="Error Relativo ($E_{rel}$)", color='#4b4bff', lw=2)
                        ax.plot(res["Iter"], res["E_Cond"], marker='^', label="Error de Condición ($|f(x_n)|$)", color='#22c55e', lw=2)
                        
                        ax.set_yscale('log')
                        ax.set_xlabel("Iteración", fontweight='bold')
                        ax.set_ylabel("Magnitud del Error (Escala Log)", fontweight='bold')
                        ax.grid(True, which="both", alpha=0.3, linestyle='--')
                        ax.legend(facecolor='#f8f9fa')
                        st.pyplot(fig)

            # --- CÁLCULO SÓLO SI SE SELECCIONÓ SISTEMAS LINEALES ---
            elif categoria == "Sistemas Lineales":
                matriz_A = df_A.to_numpy()
                vector_b = df_b.to_numpy().flatten()
                x0_s = np.zeros(len(vector_b))
                
                if any(np.diag(matriz_A) == 0):
                    st.warning("⚠️ Diagonal con ceros detectada. El método podría fallar.")
                
                if metodo == "Jacobi": 
                    sol, res, T_mat, rho = m.jacobi(matriz_A, vector_b, x0_s, tol_sist, max_i_sist)
                elif metodo == "Gauss-Seidel": 
                    sol, res, T_mat, rho = m.gauss_seidel(matriz_A, vector_b, x0_s, tol_sist, max_i_sist)
                elif metodo == "SOR": 
                    sol, res, T_mat, rho = m.sor(matriz_A, vector_b, x0_s, tol_sist, max_i_sist, w)
                
                st.success("✅ Cálculo finalizado")
                col_sol, col_conv = st.columns([1, 1])
                
                with col_sol:
                    st.subheader("🎯 Solución Calculada:")
                    st.write(sol)
                    st.subheader("📋 Historial de Convergencia:")
                    st.dataframe(res.style.format("{:.2e}"), use_container_width=True)

                with col_conv:
                    st.subheader("🧬 Análisis de Convergencia")
                    st.metric(label=f"Radio Espectral ρ(T) de {metodo}", value=f"{rho:.5f}")
                    if rho < 1:
                        st.success(f"📈 **ρ(T) = {rho:.5f} < 1**: El método **CONVERGERÁ** de manera garantizada.")
                    else:
                        st.error(f"📉 **ρ(T) = {rho:.5f} ≥ 1**: El método **DIVERGERÁ**.")
                    
                    st.write(f"**Matriz de Transición $T$ para {metodo}:**")
                    st.dataframe(pd.DataFrame(T_mat).style.format("{:.4f}"), use_container_width=True)
                    
        except Exception as e:
            st.error(f"⚠️ Error al procesar el cálculo: {e}")