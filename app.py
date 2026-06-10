import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import re
import nltk
from nltk.corpus import stopwords

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y RECURSOS
# ==============================================================================
st.set_page_config(
    page_title="Dashboard: IA & Habilidades Blandas",
    page_icon="🧠",
    layout="wide"
)

# Descarga de recursos de NLTK para el análisis de abstracts
@st.cache_resource
def iniciar_recursos_nltk():
    nltk.download('stopwords', quiet=True)

iniciar_recursos_nltk()

# ==============================================================================
# 2. CARGA Y OPTIMIZACIÓN DEL DATASET (scopus.csv)
# ==============================================================================
@st.cache_data
def cargar_y_limpiar_data():
    try:
        # Lee el archivo scopus.csv desde la raíz de tu GitHub
        df = pd.read_csv('scopus_limpio.csv')
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'scopus.csv' en la raíz del repositorio de GitHub.")
        st.stop()
        
    # Limpieza y tipado de columnas críticas
    df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0).astype(int)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
    
    # Clasificación estricta de Scopus para Open Access
    df['Open Access Temp'] = df['Open Access'].fillna('').astype(str).str.strip()
    df['Tipo_Acceso'] = df['Open Access Temp'].apply(
        lambda x: 'Suscripción/Pago 🔒' if x in ['', 'Sin información', 'nan', 'None', 'Restricted'] else 'Acceso Abierto 🔓'
    )
    return df

df = cargar_y_limpiar_data()

# ==============================================================================
# 3. BARRA LATERAL (SIDEBAR) - DISEÑO TECNOLÓGICO Y FILTROS OPTIMIZADOS
# ==============================================================================

# Inyección de estilos CSS avanzados para personalizar los componentes según el tema oscuro
st.sidebar.markdown(
    """
    <style>
    /* Estilizar el contenedor de la barra lateral */
    [data-testid="stSidebar"] {
        background-color: #111625;
    }
    
    /* Títulos con estética de Inteligencia Artificial (Neón sutil) */
    .sidebar-title {
        color: #4fc3f7 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .sidebar-subtitle {
        color: #b0bec5 !important;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Estilo personalizado para las tarjetas de métricas en la barra lateral */
    .metric-card {
        background: linear-gradient(135deg, #1e2640 0%, #151b30 100%);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #263238;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        color: #00e676;
        font-size: 20px;
        font-weight: bold;
    }
    .metric-lbl {
        color: #90a4ae;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Encabezado personalizado de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">⚙️ PANEL DE CONTROL</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-subtitle">Filtros dinámicos para el análisis bibliométrico</p>', unsafe_allow_html=True)

# --- FILTRO 1: RANGO DE AÑOS ---
st.sidebar.markdown("<b style='color:#ffffff;'>📅 Período de Publicación (Scopus):</b>", unsafe_allow_html=True)
min_ano = int(df['Year'].min()) if not df.empty else 2000
max_ano = int(df['Year'].max()) if not df.empty else 2026

rango_anos = st.sidebar.slider(
    "", # Dejamos la etiqueta nativa vacía porque ya usamos HTML arriba para tunear el color
    min_value=min_ano,
    max_value=max_ano,
    value=(min_ano, max_ano),
    help="Desplaza los nodos para ajustar la ventana del tiempo de la investigación académica."
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# --- FILTRO 2: TIPO DE ACCESO (ESTILO CHECKBOX MINIMALISTA) ---
st.sidebar.markdown("<b style='color:#ffffff;'>🔓 Condición de Acceso:</b>", unsafe_allow_html=True)
opciones_acceso = list(df['Tipo_Acceso'].unique())

accesos_seleccionados = []
# Construimos filas ordenadas para los controles
for opcion in opciones_acceso:
    if st.sidebar.checkbox(f" {opcion}", value=True, key=f"chk_{opcion}"):
        accesos_seleccionados.append(opcion)

# Sistema de seguridad por si el usuario desmarca todo por error
if not accesos_seleccionados:
    accesos_seleccionados = opciones_acceso

# --- FILTRADO DE LA DATA ---
df_filtrado = df[
    (df['Year'] >= rango_anos[0]) & 
    (df['Year'] <= rango_anos[1]) & 
    (df['Tipo_Acceso'].isin(accesos_seleccionados))
]

# --- TARJETAS DE MÉTRICAS INTEGRADAS EN EL PANEL ---
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#90a4ae; font-size:12px; font-weight:bold; letter-spacing:1px;'>ESTADO DE LA MUESTRA</p>", unsafe_allow_html=True)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    st.sidebar.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val">{len(df_filtrado)}</div>
            <div class="metric-lbl">Artículos</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
with col_sb2:
    st.sidebar.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val">{df_filtrado['Cited by'].sum():,}</div>
            <div class="metric-lbl">Citas Tot</div>
        </div>
        """, 
        unsafe_allow_html=True
    )# ==============================================================================
# 4. DISEÑO DE LA SECCIÓN SUPERIOR: TÍTULO Y DESCRIPCIÓN
# ==============================================================================
st.title("El rol de la IA en las habilidades blandas")

st.markdown("""
### DESCRIPCIÓN:
Este cuaderno de trabajo utiliza un dataset especializado obtenido mediante una extracción bibliométrica avanzada desde la plataforma científica **Scopus**. El conjunto de datos recopila la producción académica global que intersecta las tecnologías de **Inteligencia Artificial** con el desarrollo, evaluación e impacto de las **Habilidades Blandas (Soft Skills)** en el mercado laboral actual.
""")

st.write("---")

# Pequeño resumen de métricas en la parte superior para diseño moderno
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Documentos Filtrados", f"{len(df_filtrado)}")
with kpi2:
    st.metric("Total Citas del Grupo", f"{df_filtrado['Cited by'].sum():,}")
with kpi3:
    st.metric("Rango Temporal", f"{rango_anos[0]} - {rango_anos[1]}")

st.write("---")

# Configuración del estilo general de los gráficos
sns.set_theme(style="whitegrid")
paleta_principal = "Blues_r"

# ==============================================================================
# 5. RENDERIZADO DE LOS 6 GRÁFICOS REQUERIDOS
# ==============================================================================

# DISPOSICIÓN EN MATRIZ: Fila 1 (Gráficos 1 y 2)
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Gráfico 1: Distribución de Publicaciones por Año")
    if not df_filtrado.empty:
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        df_anos = df_filtrado['Year'].value_counts().sort_index()
        sns.lineplot(x=df_anos.index, y=df_anos.values, marker="o", color="#1565c0", linewidth=2.5, ax=ax1)
        ax1.set_xlabel("Año de Publicación")
        ax1.set_ylabel("Cantidad de Documentos")
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.info("Sin datos")

with col_b:
    st.subheader("Gráfico 2: Top 10 de Artículos Más Citados")
    df_top10_art = df_filtrado.sort_values(by='Cited by', ascending=False).head(10)
    if not df_top10_art.empty:
        df_top10_art['Title_Corto'] = df_top10_art['Title'].apply(lambda x: str(x)[:40] + '...' if len(str(x)) > 40 else str(x))
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=df_top10_art, x='Cited by', y='Title_Corto', hue='Title_Corto', palette=paleta_principal, legend=False, ax=ax2, edgecolor="black")
        ax2.set_xlabel("Total de Citas (Cited by)")
        ax2.set_ylabel("Título del Artículo")
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("Sin datos")

st.write("---")

# DISPOSICIÓN EN MATRIZ: Fila 2 (Gráficos 3 y 4)
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Gráfico 3: Distribución Porcentual de las Top 7 Keywords")
    # Procesamiento de la columna 'Author Keywords'
    keywords_lista = []
    for kw_celda in df_filtrado['Author Keywords'].dropna():
        if kw_celda != 'Sin información':
            # Separar por punto y coma o coma según guarde Scopus
            for kw in re.split(r'[;,]', str(kw_celda)):
                kw_limpio = kw.strip().title()
                if kw_limpio:
                    keywords_lista.append(kw_limpio)
                    
    if keywords_lista:
        top7_kw = pd.DataFrame(collections.Counter(keywords_lista).most_common(7), columns=['Keyword', 'Conteo'])
        fig3, ax3 = plt.subplots(figsize=(7, 7))
        colores_pie = sns.color_palette("pastel", 7)
        ax3.pie(top7_kw['Conteo'], labels=top7_kw['Keyword'], autopct='%1.1f%%', startangle=140, colors=colores_pie, 
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}, textprops={'fontweight': 'bold'})
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("No se encontraron Keywords indexadas para este filtro.")

with col_d:
    st.subheader("Gráfico 4: Impacto del Tipo de Acceso vs Citaciones")
    if len(df_filtrado) > 0 and len(df_filtrado['Tipo_Acceso'].unique()) > 1:
        fig4, ax4 = plt.subplots(figsize=(8, 5.5))
        sns.violinplot(
            data=df_filtrado, x='Tipo_Acceso', y='Cited by',
            palette=['#78909c', '#ff7043'], hue='Tipo_Acceso', legend=False,
            inner='box', cut=0, ax=ax4
        )
        ax4.set_ylabel('Cantidad de Citaciones (Cited by)')
        ax4.set_xlabel('Condición de Publicación en Scopus')
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.info("Se necesitan ambos tipos de acceso en el filtro para pintar la comparativa del violín.")

st.write("---")

# DISPOSICIÓN EN MATRIZ: Fila 3 (Gráficos 5 y 6)
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Gráfico 5: Top 10 Autores con Mayor Impacto")
    autor_citas_totales = {}
    for _, fila in df_filtrado.iterrows():
        celda_autores = fila['Authors']
        citas_num = int(fila['Cited by'])
        if celda_autores != 'Sin información' and pd.notnull(celda_autores):
            for autor in [a.strip() for a in str(celda_autores).split(',') if a.strip()]:
                autor_citas_totales[autor] = autor_citas_totales.get(autor, 0) + citas_num

    if autor_citas_totales:
        df_top_autores = pd.DataFrame(list(autor_citas_totales.items()), columns=['Autor', 'Total_Citas'])
        df_top_autores = df_top_autores.sort_values(by='Total_Citas', ascending=False).head(10)
        df_top_autores['Autor_Corto'] = df_top_autores['Autor'].apply(lambda x: x.split(',')[0])
        
        fig5, ax5 = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=df_top_autores, x='Total_Citas', y='Autor_Corto', hue='Autor_Corto', palette="mako", legend=False, ax=ax5, edgecolor="black")
        ax5.set_xlabel("Total de Citaciones Acumuladas")
        ax5.set_ylabel("Autores (Primer Apellido)")
        plt.tight_layout()
        st.pyplot(fig5)
    else:
        st.info("Sin datos de autores.")

with col_f:
    st.subheader("Gráfico 6: Análisis de Palabras en Abstracts")
    lista_bigramas = []
    palabras_vacias = set(stopwords.words('english') + stopwords.words('spanish'))
    palabras_vacias.update(['abstract', 'paper', 'research', 'study', 'results', 'author', 'methodology', 'using', 'analysis', 'ieee'])
    
    for resumen in df_filtrado['Abstract'].dropna():
        if resumen != 'Sin información' and isinstance(resumen, str):
            texto_limpio = re.sub(r'[^a-zA-Z\s]', '', resumen.lower())
            palabras_filtradas = [p for p in texto_limpio.split() if p not in palabras_vacias and len(p) > 2]
            for i in range(len(palabras_filtradas) - 1):
                lista_bigramas.append(f"{palabras_filtradas[i]} {palabras_filtradas[i+1]}")
                
    if lista_bigramas:
        df_bigramas = pd.DataFrame(collections.Counter(lista_bigramas).most_common(10), columns=['Bigrama', 'Frecuencia'])
        fig6, ax6 = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=df_bigramas, x='Frecuencia', y='Bigrama', hue='Bigrama', palette="flare_r", legend=False, ax=ax6, edgecolor="black")
        ax6.set_xlabel("Frecuencia de Aparición (Conteo)")
        ax6.set_ylabel("Bigramas (Conceptos en Pareja)")
        plt.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("Sin datos de texto suficientes para extraer bigramas.")
