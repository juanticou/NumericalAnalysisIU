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
    "Interpolación": ["Vandermonde", "Lagrange", "Newton Interpolante", "Spline Lineal", "Spline Cubico"]
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
            # 📝 GUÍA DE SINTAXIS PARA EL USUARIO
            with st.expander("📝 Guía de Sintaxis Matemática (Haz clic para expandir)", expanded=False):
                st.markdown("""
                Para que el sistema interprete correctamente tus funciones, sigue estas reglas:
                * **Multiplicaciones explícitas:** Usa siempre `*`. Escribe `2*x` en lugar de `2x`.
                * **Potencias / Exponentes:** Usa `**` o `^` (ejemplo: `x**2` o `x^2`).
                * **Funciones Trigonométricas:** `sin(x)`, `cos(x)`, `tan(x)`.
                * **Otras funciones comunes:** `exp(x)` (para $e^x$), `ln(x)` o `log(x)` (logaritmo natural), `sqrt(x)` ($\sqrt{x}$).
                * **Constantes:** `pi` ($\pi$), `e` ($e$).
                """)

            f_input = st.text_input("Función f(x)", "x**2 - 5", key="f_input_nl")
            
            # Inicialización segura de variables de control
            f_valida = False
            
            try:
                x_sym = sp.Symbol('x')
                locs = {'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp, 
                        'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E}
                f_expr = sp.parse_expr(f_input.replace('^', '**'), local_dict=locs)
                
                # Derivadas automáticas simbólicas
                df_expr = sp.diff(f_expr, x_sym)
                ddf_expr = sp.diff(df_expr, x_sym)
                
                st.info("**Interpretación Matemática:**")
                st.latex(rf"f(x) = {sp.latex(f_expr)}")
                f_valida = True
            except Exception as e:
                st.error(f"Error al interpretar f(x): {e}")

            # Inputs numéricos condicionales según el método seleccionado
            if metodo in ["Bisección", "Regla Falsa"]:
                a = st.number_input("Extremo a", value=1.0, key="nl_a")
                b = st.number_input("Extremo b", value=3.0, key="nl_b")
            elif metodo == "Secante":
                x0 = st.number_input("Punto inicial x0", value=1.0, key="nl_x0_sec")
                x1 = st.number_input("Punto inicial x1", value=3.0, key="nl_x1_sec")
            else:
                x0 = st.number_input("Punto inicial x0", value=1.0, key="nl_x0")

            if metodo == "Punto Fijo":
                g_input = st.text_input("Función de iteración g(x)", "sqrt(5)", key="g_input_pf")
                try:
                    g_expr = sp.parse_expr(g_input.replace('^', '**'), local_dict=locs)
                    st.latex(rf"g(x) = {sp.latex(g_expr)}")
                except Exception as e:
                    st.error(f"Error en g(x): {e}")

        with col_right:
            tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-7, format="%.e", key="nl_tol")
            max_iter = st.number_input("Máximo Iteraciones", min_value=1, value=100, key="nl_max_iter")
            
            # 📊 GRÁFICA DE LA FUNCIÓN EN TIEMPO REAL
            if f_valida:
                st.write("**Vista Previa de la Curva de f(x):**")
                try:
                    f_preview = sp.lambdify(x_sym, f_expr, 'numpy')
                    
                    # Definimos un rango dinámico óptimo en X de acuerdo a las entradas del usuario
                    if metodo in ["Bisección", "Regla Falsa"]:
                        rango_x = np.linspace(min(a, b) - 2, max(a, b) + 2, 400)
                    elif metodo == "Secante":
                        rango_x = np.linspace(min(x0, x1) - 3, max(x0, x1) + 3, 400)
                    else:
                        rango_x = np.linspace(x0 - 4, x0 + 4, 400)
                    
                    fig_prev, ax_prev = plt.subplots(figsize=(6, 3.2))
                    ax_prev.plot(rango_x, f_preview(rango_x), color='#ef4444', lw=2.5, label="$f(x)$")
                    ax_prev.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.2) # Eje X = 0
                    
                    # Marcamos visualmente los puntos de partida colocados por el usuario
                    if metodo in ["Bisección", "Regla Falsa"]:
                        ax_prev.axvline(a, color='#f59e0b', linestyle=':', lw=2, label=f'a ({a})')
                        ax_prev.axvline(b, color='#8b5cf6', linestyle=':', lw=2, label=f'b ({b})')
                    elif metodo == "Secante":
                        ax_prev.scatter([x0, x1], [f_preview(x0), f_preview(x1)], color='black', s=50, zorder=5, label='Puntos iniciales')
                    else:
                        ax_prev.scatter([x0], [f_preview(x0)], color='black', s=60, zorder=5, label=f'x0 ({x0})')
                    
                    ax_prev.grid(True, alpha=0.3, linestyle='--')
                    ax_prev.legend(facecolor='#f8f9fa', fontsize=9)
                    ax_prev.set_xlabel("x")
                    ax_prev.set_ylabel("f(x)")
                    st.pyplot(fig_prev)
                except Exception as graph_err:
                    st.caption(f"No se pudo generar la vista previa gráfica: {graph_err}")
                    
    # --- INPUTS: SISTEMAS LINEALES ---
    elif categoria == "Sistemas Lineales":
        st.write("### 1. Configuración del Sistema Lineal ($Ax = b$)")
        
        # Control del tamaño de la matriz
        n = st.number_input("Dimensión del sistema (n x n)", min_value=2, max_value=10, value=3, step=1, key="sl_dimension")
        
        st.write("🔧 **Modifique los valores en las tablas para configurar su sistema:**")
        
        # Distribución en columnas para Matriz A, Vector b y Vector Inicial x0
        col_mat_A, col_vec_b, col_vec_x0 = st.columns([3, 1, 1])
        
        with col_mat_A:
            st.caption("Matriz de Coeficientes (A)")
            # Matriz identidad por defecto para evitar celdas vacías
            default_A = np.eye(n)
            df_A = st.data_editor(pd.DataFrame(default_A), key="matrix_A_editor", use_container_width=True)
            
        with col_vec_b:
            st.caption("Vector b")
            default_b = np.ones(n)
            df_b = st.data_editor(pd.DataFrame(default_b, columns=["b"]), key="vector_b_editor", use_container_width=True)
            
        with col_vec_x0:
            st.caption("Vector Inicial (x0)")
            default_x0 = np.zeros(n)
            df_x0 = st.data_editor(pd.DataFrame(default_x0, columns=["x0"]), key="vector_x0_editor", use_container_width=True)
            
        # Parámetros de control del método numérico
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-7, format="%.e", key="sl_tol")
            max_iter = st.number_input("Máximo Iteraciones", min_value=1, value=100, key="sl_max_iter")
        with col_s2:
            if metodo == "SOR":
                omega = st.number_input("Factor de Relajación (w)", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="sl_omega")
            else:
                omega = 1.0

    # --- INPUTS: INTERPOLACIÓN ---
    elif categoria == "Interpolación":
        st.write("### 1. Puntos de la Función Base (Datos Experimentales)")
        st.info("💡 Agrega suficientes puntos (por ejemplo, 6 o más) para poder dividirlos en Entrenamiento y Validación de forma óptima.")
        
        # Tabla extendida por defecto para pruebas cómodas
        df_puntos = st.data_editor(
            pd.DataFrame({
                'x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                'y': [1.0, 2.7, 5.2, 3.1, 1.5, 4.3, 7.2]
            }), 
            num_rows="dynamic", 
            key="tabla_puntos_interp"
        )
        
        # Limpieza y ordenamiento estricto por el eje X
        df_puntos = df_puntos.dropna().sort_values(by='x')
        xp, yp = df_puntos['x'].values, df_puntos['y'].values
        
        # Slider dinámico para controlar la proporción del ajuste
        pct_train = st.slider("Porcentaje de Datos de Entrenamiento (%)", min_value=50, max_value=100, value=100, step=10, key="interp_pct_split")
        
        # Aseguramos el menú dinámico con Spline Lineal incluido
        menus["Interpolación"] = ["Vandermonde", "Lagrange", "Newton Interpolante", "Spline Lineal", "Spline Cubico"]

# --- LÓGICA DE EJECUCIÓN AISLADA ---

if categoria == "Interpolación":
    try:
        if len(xp) > 2 and len(set(xp)) == len(xp):
            
            # --- 1. DIVISIÓN DINÁMICA DE ENTRENAMIENTO / VALIDACIÓN ---
            if pct_train == 100:
                xp_train, yp_train = xp, yp
                xp_val, yp_val = np.array([]), np.array([])
            else:
                n_val = int(np.ceil(len(xp) * (1 - pct_train / 100)))
                
                # Garantizamos que queden al menos 3 puntos para entrenar los modelos (Spline cúbico lo exige)
                if len(xp) - n_val < 3:
                    n_val = len(xp) - 3
                
                if n_val > 0 and len(xp) > 3:
                    # Seleccionamos estratégicamente puntos internos distribuidos de manera uniforme para validación
                    idx_val = np.linspace(1, len(xp) - 2, n_val, dtype=int)
                    idx_train = [i for i in range(len(xp)) if i not in idx_val]
                    
                    xp_train, yp_train = xp[idx_train], yp[idx_train]
                    xp_val, yp_val = xp[idx_val], yp[idx_val]
                else:
                    xp_train, yp_train = xp, yp
                    xp_val, yp_val = np.array([]), np.array([])
                    st.warning("⚠️ Elementos insuficientes para realizar un split. Usando el 100% para entrenamiento.")

            # --- 2. AJUSTE DE MODELOS UTILIZANDO EXCLUSIVAMENTE ENTRENAMIENTO ---
            if metodo == "Vandermonde": modelo = m.vandermonde(xp_train, yp_train)
            elif metodo == "Lagrange": modelo = m.lagrange(xp_train, yp_train)
            elif metodo == "Newton Interpolante": modelo = m.newton_interpolante(xp_train, yp_train)
            elif metodo == "Spline Lineal": modelo, tramos_txt = m.spline_lineal(xp_train, yp_train)
            elif metodo == "Spline Cubico": modelo, tramos_txt = m.spline_cubico(xp_train, yp_train)

            # --- 3. IMPRESIÓN DEL POLINOMIO SOLUCIÓN EN LA INTERFAZ ---
            st.write("### 2. Expresión Matemática de la Curva Ajustada (Datos de Entrenamiento)")
            if metodo in ["Spline Lineal", "Spline Cubico"]:
                for t in tramos_txt: st.latex(t)
            else:
                st.latex(rf"P(x) = {sp.latex(modelo)}")

            st.divider()
            
            # --- 4. RENDERIZADO DE GRÁFICA Y MÉTRICAS EN DOS COLUMNAS ---
            col_gr, col_err = st.columns([2, 1])

            with col_gr:
                st.write("### 3. Gráfica de Ajuste y Residuos Visuales")
                fig, ax = plt.subplots(figsize=(8, 5))
                x_range = np.linspace(min(xp) - 0.2, max(xp) + 0.2, 500)
                
                # Pintar la curva continua generada por el entrenamiento
                if metodo in ["Spline Lineal", "Spline Cubico"]:
                    ax.plot(x_range, modelo(x_range), color='#2563eb', label=f"Curva {metodo}", linewidth=2)
                else:
                    f_plt = sp.lambdify(sp.Symbol('x'), modelo, 'numpy')
                    ax.plot(x_range, f_plt(x_range), color='#2563eb', label=f"Polinomio {metodo}", linewidth=2)
                
                # Dibujar los nodos que sí se usaron para armar el polinomio
                ax.scatter(xp_train, yp_train, color='#2563eb', s=80, marker='o', label="Puntos Entrenamiento", zorder=5)
                
                # Si hay datos de validación, graficar los errores ("Lo que me estoy equivocando")
                if len(xp_val) > 0:
                    ax.scatter(xp_val, yp_val, color='#dc2626', s=90, marker='X', label="Puntos Validación (Ocultos)", zorder=6)
                    
                    # Obtener las predicciones sobre el set de validación
                    if metodo in ["Spline Lineal", "Spline Cubico"]:
                        yp_pred = modelo(xp_val)
                    else:
                        f_eval = sp.lambdify(sp.Symbol('x'), modelo, 'numpy')
                        yp_pred = f_eval(xp_val)
                    
                    # Trazar líneas verticales que reflejen visualmente el error/residuo de cada punto oculto
                    for i, (xv, yv, y_hat) in enumerate(zip(xp_val, yp_val, yp_pred)):
                        ax.vlines(xv, yv, y_hat, colors='#b91c1c', linestyles='dashed', linewidth=1.8,
                                  label="Residuo (Error de predicción)" if i == 0 else "")
                
                ax.legend(facecolor='#f8f9fa')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel("Eje X")
                ax.set_ylabel("Eje Y")
                st.pyplot(fig)

            with col_err:
                st.write("### 4. Cuantificación del Error")
                
                if len(xp_val) > 0:
                    # Cálculo numérico del error sobre los datos ocultos
                    mae = np.mean(np.abs(yp_val - yp_pred))
                    mse = np.mean((yp_val - yp_pred) ** 2)
                    
                    st.metric(label="Error Absoluto Medio (MAE)", value=f"{mae:.5f}")
                    st.metric(label="Error Cuadrático Medio (MSE)", value=f"{mse:.5f}")
                    
                    # Mostrar tabla de desviaciones punto por punto
                    df_desviacion = pd.DataFrame({
                        "X (Validación)": xp_val,
                        "Y Real": yp_val,
                        "Y Calculado": yp_pred,
                        "Error Absoluto": np.abs(yp_val - yp_pred)
                    })
                    st.write("**Desglose numérico por nodo:**")
                    st.dataframe(df_desviacion.style.format("{:.4f}"), use_container_width=True)
                else:
                    st.success("🎯 **Entrenamiento al 100%**")
                    st.info("""
                    Al ajustar con todos los datos experimentales, el polinomio/spline pasa **exactamente** por encima de todos los nodos. 
                    
                    Por ende, el error de aproximación teórica en esos puntos específicos es exactamente **0.00000**. Mueve el control de porcentaje hacia la izquierda para observar cómo cambia el polinomio cuando desconoce parte de la información.
                    """)
                    
        else:
            st.warning("⚠️ Asegúrate de que las coordenadas X sean totalmente únicas y que tengas por lo menos 3 puntos en la tabla.")
    except Exception as e:
        st.error(f"⚠️ Error en el procesamiento del método de interpolación: {e}")
# --- CASO 2: REQUERIR BOTÓN (Ecuaciones No Lineales y Sistemas Lineales) ---
else:
    if st.button("🚀 EJECUTAR CÁLCULO"):
        try:
           # --- CÁLCULO SÓLO SI SE SELECCÓ ECUACIONES NO LINEALES ---
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
                    # 🔍 --- NUEVA LÓGICA DE DIAGNÓSTICO DE CONVERGENCIA ---
                    ultima_iter = res.iloc[-1]
                    raiz_x = ultima_iter["x_n"]
                    f_valor = ultima_iter["f(x)"]
                    
                    # Criterio fundamental: ¿f(x) es cero bajo la tolerancia?
                    convergio_a_cero = abs(f_valor) <= tol
                    
                    # Criterio secundario: ¿Se detuvo porque x_n ya no cambiaba?
                    se_estanco = ultima_iter["E_Rel"] <= tol
                    
                    st.divider()
                    
                    # CASO 1: Éxito rotundo (f(x) tiende a cero)
                    if convergio_a_cero:
                        st.success(f"""
                        🎯 **¡El método CONVERGIÓ exitosamente a un Cero!**
                        * **Raíz aproximada encontrada ($x$):** `{raiz_x:.10f}`
                        * **Valor de $f(x)$:** `{f_valor:.2e}` (Efectivamente 0)
                        * **Iteraciones requeridas:** `{len(res)}` de `{max_iter}`
                        """)
                        
                    # CASO 2: Convergió/Se detuvo en un punto que NO es cero (Mínimo local, asíntota, etc.)
                    elif se_estanco:
                        st.warning(f"""
                        ⚠️ **El método se estabilizó, pero NO convergió a un cero.**
                        La distancia entre iteraciones sucesivas es menor que la tolerancia, por lo que el algoritmo dejó de moverse; sin embargo, la función **no es cero** en este punto. Podría tratarse de un mínimo local o un estancamiento.
                        * **Punto de parada ($x$):** `{raiz_x:.10f}`
                        * **Valor final de $f(x)$:** `{f_valor:.6f}` (No es una raíz)
                        * **Iteraciones ejecutadas:** `{len(res)}`
                        """)
                        
                    # CASO 3: El método agotó los ciclos sin cumplir tolerancias
                    else:
                        st.error(f"""
                        ❌ **El método NO alcanzó la convergencia.**
                        Se alcanzó el límite máximo de **{max_iter}** iteraciones sin lograr que el algoritmo se estabilizara ni encontrara un cero.
                        * **Último valor de $x_n$ calculado:** `{raiz_x:.10f}`
                        * **Último valor de $f(x)$:** `{f_valor:.2e}`
                        """)
                    
                    # --- RENDERIZADO DE TABLAS Y GRÁFICAS ABAJO ---
                    col_tabla, col_grafica = st.columns([1, 1])
                    
                    with col_tabla:
                        st.subheader("📋 Tabla de Iteraciones")
                        st.dataframe(res.style.format({
                            "x_n": "{:.6f}", "f(x)": "{:.2e}", 
                            "E_Abs": "{:.2e}", "E_Rel": "{:.2e}"
                        }), use_container_width=True)
                    
                    with col_grafica:
                        st.subheader("📊 Historial de Errores")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(res["Iter"], res["E_Abs"], marker='o', label="Error Absoluto ($E_{abs}$)", color='#ff4b4b', lw=2)
                        ax.plot(res["Iter"], res["E_Rel"], marker='s', label="Error Relativo ($E_{rel}$)", color='#4b4bff', lw=2)
                        ax.plot(res["Iter"], np.abs(res["f(x)"]), marker='^', label="Error de Condición ($|f(x)|$)", color='#22c55e', lw=2)
                        
                        ax.set_yscale('log')
                        ax.set_xlabel("Iteración", fontweight='bold')
                        ax.set_ylabel("Magnitud (Escala Log)", fontweight='bold')
                        ax.grid(True, which="both", alpha=0.3, linestyle='--')
                        ax.legend(facecolor='#f8f9fa')
                        st.pyplot(fig)

            # --- CÁLCULO SÓLO SI SE SELECCIONÓ SISTEMAS LINEALES ---
            elif categoria == "Sistemas Lineales":
                matriz_A = df_A.to_numpy()
                vector_b = df_b.to_numpy().flatten()
                x0_s = np.zeros(len(vector_b))
                tol_sist = tol
                max_i_sist = max_iter
                w = omega
                
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
                    
                                    # --- GRÁFICA DE ERRORES PARA SISTEMAS LINEALES ---
                if res is not None and not res.empty:
                    st.subheader("📊 Gráfica de Convergencia del Sistema")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Eje X dinámico
                    x_axis = res["Iter"] if "Iter" in res.columns else (res.index + 1)
                    
                    # Eje Y: Trazar los 3 errores (si existen en el DataFrame)
                    if "E_Abs" in res.columns:
                        ax.plot(x_axis, res["E_Abs"], marker='o', color='#ff4b4b', linewidth=2, label="Error Absoluto ($E_{abs}$)")
                    if "E_Rel" in res.columns:
                        ax.plot(x_axis, res["E_Rel"], marker='s', color='#4b4bff', linewidth=2, label="Error Relativo ($E_{rel}$)")
                    if "E_Cond" in res.columns:
                        ax.plot(x_axis, res["E_Cond"], marker='^', color='#22c55e', linewidth=2, label="Error de Condición ($||b - Ax||$)")
                    
                    ax.set_yscale('log')
                    ax.set_xlabel("Iteración", fontweight='bold')
                    ax.set_ylabel("Magnitud del Error (Escala Log)", fontweight='bold')
                    ax.grid(True, which="both", alpha=0.3, linestyle='--')
                    ax.legend(facecolor='#f8f9fa')
                    
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"⚠️ Error al procesar el cálculo: {e}")