from seaborn import color_palette
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, ward, single, complete, average, linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
try:
    from statsmodels.multivariate.manova import MANOVA
    STATSMODELS_DISPONIBLE = True
except ImportError:
    STATSMODELS_DISPONIBLE = False
from scipy.stats import shapiro, normaltest, anderson, ttest_ind, mannwhitneyu, kruskal, f_oneway
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import ceil, floor, pi
from typing import Dict, List, Union, Optional

# Scikit-learn imports
from sklearn import linear_model, ensemble, datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# sklearn_extra removido por problemas de compatibilidad binaria con numpy
SKLEARN_EXTRA_DISPONIBLE = False

# Otros imports
try:
    from prince import PCA as PCA_Prince
    PRINCE_DISPONIBLE = True
except ImportError:
    PRINCE_DISPONIBLE = False

try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Categorical, Continuous
    SKLEARN_GENETIC_DISPONIBLE = True
except ImportError:
    SKLEARN_GENETIC_DISPONIBLE = False


try:
    import umap
    UMAP_DISPONIBLE = True
except ImportError:
    UMAP_DISPONIBLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False


# Configuración inicial
pd.options.display.max_rows = 10
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class AnalisisDatosExploratorio:
    """Clase para análisis exploratorio de datos con visualizaciones integradas."""

    def __init__(self, path: str = None, num: int = 1):
        """Inicializa el analizador con los datos.

        Args:
            path: Ruta al archivo de datos
            num: Tipo de archivo (1=CSV con coma, 2=CSV con punto y coma)
        """
        if path is None or path == "":
            self._df = pd.DataFrame()
        else:
            self._df = self.cargar_datos(path, num)

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame con los datos cargados."""
        return self._df

    @df.setter
    def df(self, nuevo_df: pd.DataFrame):
        """Setter para actualizar el DataFrame."""
        self._df = nuevo_df

    def analisis_numerico(self):
        """Filtra solo las columnas numéricas."""
        self._df = self._df.select_dtypes(include=["number"])

    def analisis_completo(self):
        """Convierte variables categóricas a dummy variables."""
        self._df = pd.get_dummies(self._df)

    def cargar_datos(self, path: str, num: int) -> pd.DataFrame:
        opciones = {
            1: {"sep": ",", "decimal": ".", "index_col": 0},
            2: {"sep": ";", "decimal": ".", "index_col": None}
        }
        try:
            if num not in opciones:
                raise ValueError("Número de formato no válido. Use 1 o 2.")
            params = opciones[num]
            kwargs = {k: v for k, v in params.items() if v is not None}
            logger.info(f"Cargando datos desde {path} con opciones {kwargs}")
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            logger.error(f"El archivo '{path}' no fue encontrado.")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            logger.error(f"Error de parseo al leer el archivo: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error inesperado al cargar los datos: {e}")
            return pd.DataFrame()

    def analisis(self):
        """Ejecuta un análisis completo con estadísticas y visualizaciones, con manejo de errores."""
        try:
            logger.info(f"Dimensiones: {self._df.shape}")
            logger.info(f"Primeras filas:\n{self._df.head()}")
            logger.info(f"Estadísticas descriptivas:\n{self._df.describe()}")
            logger.info(f"Media:\n{self._df.mean(numeric_only=True)}")
            logger.info(f"Mediana:\n{self._df.median(numeric_only=True)}")
            logger.info(
                f"Desviación estándar:\n{self._df.std(numeric_only=True, ddof=0)}")
            logger.info(f"Máximos:\n{self._df.max(numeric_only=True)}")
            logger.info(f"Mínimos:\n{self._df.min(numeric_only=True)}")
            logger.info(
                f"Cuartiles:\n{self._df.quantile(np.array([0, 0.33, 0.50, 0.75, 1]), numeric_only=True)}")
            self.graficos_boxplot()
            self.funcion_densidad()
            self.histograma()
            self.correlaciones()
            self.grafico_correlacion()
        except Exception as e:
            logger.error(f"Error en análisis exploratorio: {e}")

    def graficos_boxplot(self):
        """Genera gráficos de caja para todas las variables, con manejo de errores."""
        try:
            fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
            self._df.boxplot(ax=ax)
            plt.title("Boxplots de Variables")
            plt.xticks(rotation=45)
            plt.show()
        except Exception as e:
            logger.error(f"Error al generar boxplot: {e}")

    def funcion_densidad(self):
        """Genera gráficos de densidad para todas las variables, con manejo de errores."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            self._df.plot(kind='density', ax=ax)
            plt.title("Funciones de Densidad")
            plt.show()
        except Exception as e:
            logger.error(f"Error al generar función de densidad: {e}")

    def histograma(self):
        """Genera histogramas para todas las variables, con manejo de errores."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            self._df.plot(kind='hist', ax=ax, alpha=0.7)
            plt.title("Histogramas")
            plt.show()
        except Exception as e:
            logger.error(f"Error al generar histograma: {e}")

    def correlaciones(self):
        """Calcula y muestra la matriz de correlaciones, con manejo de errores."""
        try:
            corr = self._df.corr(numeric_only=True)
            logger.info(f"Matriz de Correlación:\n{corr}")
        except Exception as e:
            logger.error(f"Error al calcular correlaciones: {e}")

    def grafico_correlacion(self):
        """Genera un heatmap de correlaciones, con manejo de errores."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
            paleta = sns.diverging_palette(220, 10, as_cmap=True).reversed()
            corr = self._df.corr(numeric_only=True)
            sns.heatmap(corr, vmin=-1, vmax=1, cmap=paleta,
                        square=True, annot=True, ax=ax)
            plt.title("Mapa de Calor de Correlaciones")
            plt.show()
        except Exception as e:
            logger.error(f"Error al generar heatmap de correlaciones: {e}")

    @staticmethod
    def centroide(num_cluster: int, datos: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
        """Calcula el centroide para un cluster específico."""
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    @staticmethod
    def recodificar(col: pd.Series, nuevo_codigo: Dict) -> pd.Series:
        """Recodifica una columna según el diccionario proporcionado, optimizado O(n)."""
        return col.replace(nuevo_codigo)

    @staticmethod
    def bar_plot(centros: np.ndarray, labels: List[str], scale: bool = False,
                 cluster: List[int] = None, var: List[str] = None):
        """Genera un gráfico de barras para los centroides de clusters."""
        fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
        centros = np.copy(centros)

        if scale:
            max_vals = centros.max(axis=0)
            # Evitar división por cero
            max_vals = np.where(max_vals == 0, 1, max_vals)
            centros = centros / max_vals

        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0

        def inside_plot(valores, etiquetas, titulo):
            plt.barh(range(len(valores)), valores, 1 /
                     1.5, color=colores[:len(valores)])
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)

        if var is not None:
            # Filtrar centros y etiquetas según las variables seleccionadas
            indices_var = [i for i, label in enumerate(labels) if label in var]
            centros = centros[:, indices_var]
            colores = [colores[x % len(colores)] for x in indices_var]
            labels = [labels[i] for i in indices_var]

        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, f'Cluster {i}')
                plt.yticks(range(len(labels)),
                           labels) if i == 0 else plt.yticks([])
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, f'Cluster {i}')
                plt.yticks(range(len(labels)),
                           labels) if pos == 1 else plt.yticks([])
                pos += 1

    @staticmethod
    def bar_plot_detail(centros: np.ndarray, columns_names: List[str] = [],
                        columns_to_plot: List[str] = [], figsize: tuple = (10, 7),
                        dpi: int = 150):
        """Genera gráficos de barras detallados para clusters."""
        fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
        num_clusters = centros.shape[0]
        labels = [f"Cluster {i}" for i in range(num_clusters)]
        centros_df = pd.DataFrame(centros, columns=columns_names, index=labels)

        plots = len(columns_to_plot) if columns_to_plot else len(columns_names)
        rows, cols = ceil(plots/2), 2

        plt.figure(1, figsize=figsize, dpi=dpi)
        plt.subplots_adjust(hspace=1, wspace=0.5)

        columns = columns_names
        if columns_to_plot:
            columns = columns_to_plot if isinstance(columns_to_plot[0], str) else [
                columns_names[i] for i in columns_to_plot]

        for var in range(plots):
            num_row, num_col = var // cols, var % cols
            ax = plt.subplot2grid(
                (rows, cols), (num_row, num_col), colspan=1, rowspan=1)
            sns.barplot(y=labels, x=columns[var], data=centros_df, ax=ax)

    @staticmethod
    def radar_plot(centros: np.ndarray, labels: List[str]):
        """Genera un gráfico de radar para visualizar clusters."""
        fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
        centros = np.array([(n - min(n)) / (max(n) - min(n)) * 100 if max(n) != min(n)
                           else (n/n * 50) for n in centros.T])

        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]

        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks(range(10, 110, 10), [f"{x}%" for x in range(10, 110, 10)],
                   color="grey", size=8)
        plt.ylim(-10, 100)

        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth=1, linestyle='solid',
                    label=f'Cluster {i}')
            ax.fill(angulos, valores, alpha=0.3)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    def __str__(self) -> str:
        return f'AnalisisDatosExploratorio: {self._df.shape}'


class ACPBasico:
    """Análisis de Componentes Principales con visualizaciones."""

    def __init__(self, datos: pd.DataFrame, n_componentes: int = 2):
        """Inicializa el ACP con los datos usando sklearn.PCA.

        Args:
            datos: DataFrame con los datos a analizar
            n_componentes: Número de componentes principales a calcular
        """
        self._datos = datos
        self._modelo = PCA(n_components=n_componentes).fit(self._datos)
        self._coordenadas_ind = pd.DataFrame(self._modelo.transform(self._datos),
                                             index=self._datos.index,
                                             columns=[f'PC{i+1}' for i in range(n_componentes)])
        # Correlación variable-componente (cargas)
        self._correlacion_var = pd.DataFrame(self._modelo.components_.T,
                                             index=self._datos.columns,
                                             columns=[f'PC{i+1}' for i in range(n_componentes)])
        # Contribución individual: varianza explicada por cada componente
        self._var_explicada = self._modelo.explained_variance_ratio_
        # No hay cos2 ni row_contributions_ en sklearn, se omiten o se pueden calcular aparte si se requiere
        self._contribucion_ind = None
        self._cos2_ind = None

    @property
    def datos(self) -> pd.DataFrame:
        return self._datos

    @datos.setter
    def datos(self, datos: pd.DataFrame):
        self._datos = datos

    @property
    def modelo(self):
        return self._modelo

    @property
    def correlacion_var(self) -> pd.DataFrame:
        return self._correlacion_var

    @property
    def coordenadas_ind(self) -> pd.DataFrame:
        return self._coordenadas_ind

    @property
    def contribucion_ind(self):
        # No disponible en sklearn.PCA
        return self._contribucion_ind

    @property
    def cos2_ind(self):
        # No disponible en sklearn.PCA
        return self._cos2_ind

    @property
    def var_explicada(self) -> np.ndarray:
        return self._var_explicada

    def plot_plano_principal(self, ejes: List[int] = [0, 1],
                             ind_labels: bool = True,
                             titulo: str = 'Plano Principal'):
        """Grafica el plano principal del ACP."""
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values

        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, color='gray')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')

        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel(f'Componente {ejes[0]} ({inercia_x}%)')
        plt.ylabel(f'Componente {ejes[1]} ({inercia_y}%)')

        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))

        plt.show()

    def plot_circulo(self, ejes: List[int] = [0, 1],
                     var_labels: bool = True,
                     titulo: str = 'Círculo de Correlación'):
        """Grafica el círculo de correlaciones del ACP."""
        cor = self.correlacion_var.iloc[:, ejes].values

        plt.figure(figsize=(10, 8))
        c = plt.Circle((0, 0), radius=1, color='steelblue', fill=False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')

        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel(f'Componente {ejes[0]} ({inercia_x}%)')
        plt.ylabel(f'Componente {ejes[1]} ({inercia_y}%)')

        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95,
                      color='steelblue', alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05,
                         self.correlacion_var.index[i], color='steelblue',
                         ha='center', va='center')

        plt.show()

    def plot_sobreposicion(self, ejes: List[int] = [0, 1],
                           ind_labels: bool = True,
                           var_labels: bool = True,
                           titulo: str = 'Sobreposición Plano-Círculo'):
        """Grafica la sobreposición del plano principal y círculo de correlaciones."""
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]

        scale = min(
            (max(x) - min(x)) / (max(cor[ejes[0]]) - min(cor[ejes[0]])),
            (max(y) - min(y)) / (max(cor[ejes[1]]) - min(cor[ejes[1]]))
        ) * 0.7

        cor = cor.values

        plt.figure(figsize=(10, 8))
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')

        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel(f'Componente {ejes[0]} ({inercia_x}%)')
        plt.ylabel(f'Componente {ejes[1]} ({inercia_y}%)')

        plt.scatter(x, y, color='gray')
        plt.title(titulo)

        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))

        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale,
                      color='steelblue', alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                         self.correlacion_var.index[i], color='steelblue',
                         ha='center', va='center')

        plt.show()


class NoSupervisado(AnalisisDatosExploratorio):
    def umap(self, n_componentes: int = 2, n_neighbors: int = 15):
        """Realiza UMAP para reducción de dimensionalidad, con manejo de errores."""
        try:
            import umap
            modelo_umap = umap.UMAP(
                n_components=n_componentes, n_neighbors=n_neighbors, random_state=42)
            componentes = modelo_umap.fit_transform(self._df)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            ax.scatter(componentes[:, 0], componentes[:, 1])
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.set_title('UMAP')
            ax.grid(False)
            plt.show()
        except Exception as e:
            print(f"Error en UMAP: {e}")
    """Clase para métodos de aprendizaje no supervisado."""

    def __init__(self, df: pd.DataFrame):
        """Inicializa con un DataFrame de datos."""
        super().__init__("", 1)  # Inicialización ficticia para la herencia
        self._df = df

    def acp(self, n_componentes: int = 2):
        """Realiza Análisis de Componentes Principales, con manejo de errores."""
        try:
            p_acp = ACPBasico(self._df, n_componentes)
            self.ploteo_graficos_acp(p_acp, 1)
            self.ploteo_graficos_acp(p_acp, 2)
            self.ploteo_graficos_acp(p_acp, 3)
        except Exception as e:
            print(f"Error en ACP: {e}")

    def ploteo_graficos_acp(self, p_acp: ACPBasico, tipo: int):
        """Método interno para plotear gráficos de ACP, con manejo de errores."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            if tipo == 1:
                p_acp.plot_plano_principal()
            elif tipo == 2:
                p_acp.plot_circulo()
            elif tipo == 3:
                p_acp.plot_sobreposicion()
            ax.grid(False)
            plt.show()
        except Exception as e:
            print(f"Error al plotear gráficos de ACP: {e}")

    def hac(self):
        """Realiza Hierarchical Agglomerative Clustering, con manejo de errores."""
        try:
            # Métodos y métricas válidas para scipy
            metodos_validos = {
                'ward': ['euclidean'],  # Ward solo funciona con euclidean
                'complete': ['euclidean', 'cosine'],
                'average': ['euclidean', 'cosine'],
                'single': ['euclidean', 'cosine']
            }

            # Realizar clustering con diferentes métodos
            resultados = {}
            for metodo in metodos_validos:
                for metrica in metodos_validos[metodo]:
                    try:
                        if metrica == 'cosine':
                            # Para cosine, usar pdist primero
                            distancias = pdist(self._df, metric='cosine')
                            res = linkage(distancias, method=metodo)
                        else:
                            # Para euclidean, usar directamente
                            res = linkage(
                                self._df, method=metodo, metric=metrica)

                        resultados[f"{metodo}_{metrica}"] = res
                        self.ploteo_graficos_hac(
                            res, f"{metodo.title()} + {metrica.title()}")
                    except Exception as e:
                        print(f"Error con {metodo} + {metrica}: {e}")

            # Visualizar clusters para el mejor método (ward por defecto)
            if 'ward_euclidean' in resultados:
                self.cluster_hac_mejorado(resultados['ward_euclidean'])
            elif resultados:
                # Usar el primer método disponible
                primer_metodo = list(resultados.keys())[0]
                self.cluster_hac_mejorado(resultados[primer_metodo])

        except Exception as e:
            print(f"Error en HAC: {e}")

    def cluster_hac_mejorado(self, linkage_matrix=None, metodo_enlace='ward', metrica='euclidean',
                             n_clusters=3, mostrar_dendrograma=True, mostrar_clusters=True, **kwargs):
        """Método mejorado para HAC compatible con Streamlit y validación anti-outliers."""
        try:
            # Si no se proporciona linkage_matrix, calcularlo
            if linkage_matrix is None:
                if metrica == 'cosine':
                    from scipy.spatial.distance import pdist
                    distancias = pdist(self._df, metric='cosine')
                    linkage_matrix = linkage(distancias, method=metodo_enlace)
                else:
                    linkage_matrix = linkage(
                        self._df, method=metodo_enlace, metric=metrica)

            # CRÍTICO: Asegurar que la matriz de linkage sea float64 y 2D
            linkage_matrix = np.array(linkage_matrix, dtype=np.float64)
            if linkage_matrix.ndim != 2:
                print(
                    f"Error: linkage_matrix tiene {linkage_matrix.ndim} dimensiones, necesita 2")
                return None

            # Generar clusters
            grupos = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            grupos = grupos - 1  # Convertir a base 0

            # VALIDACIÓN CRÍTICA: Verificar que no haya clusters muy pequeños (outliers)
            unique_clusters, counts = np.unique(grupos, return_counts=True)
            min_size_threshold = len(grupos) * 0.05  # Mínimo 5% por cluster

            # Verificar si la configuración es válida
            es_valido = len(unique_clusters) > 1 and all(
                count >= min_size_threshold for count in counts)

            if es_valido:
                # Calcular centroides
                centroides_lista = []
                for i in range(n_clusters):
                    mask = grupos == i
                    if np.sum(mask) > 0:
                        centroide = self.centroide(i, self._df, grupos)
                        centroides_lista.append(centroide)

                if centroides_lista:
                    centros = pd.concat(centroides_lista).values

                    # Visualizaciones opcionales
                    if mostrar_clusters:
                        print(
                            f"\n--- Clusters HAC {metodo_enlace}-{metrica} con {n_clusters} grupos ✓ ---")
                        for cluster_id, count in zip(unique_clusters, counts):
                            porcentaje = (count / len(grupos)) * 100
                            print(
                                f"  Cluster {cluster_id}: {count} elementos ({porcentaje:.1f}%)")

                        self.bar_plot(
                            centros, self._df.columns.tolist(), scale=True)
                        plt.show()

                    # Retornar resultado para Streamlit
                    return {
                        'clusters': grupos,
                        'centroides': centros,
                        # CRÍTICO: Asegurar float64
                        'linkage_matrix': np.array(linkage_matrix, dtype=np.float64),
                        'silhouette_score': 0.5,  # Placeholder
                        'valido': True,
                        'distribucion': dict(zip(unique_clusters, counts))
                    }
            else:
                # Configuración rechazada
                problemas = []
                for i, count in enumerate(counts):
                    porcentaje = (count / len(grupos)) * 100
                    if count < min_size_threshold:
                        problemas.append(
                            f"Cluster {i}: {count} elementos ({porcentaje:.1f}%)")

                if mostrar_clusters:
                    print(
                        f"\n--- HAC {metodo_enlace}-{metrica} con {n_clusters} grupos ✗ RECHAZADO ---")
                    print(f"Razón: {', '.join(problemas)} < 5% mínimo")

                return {
                    'clusters': grupos,
                    'centroides': None,
                    # CRÍTICO: Asegurar float64
                    'linkage_matrix': np.array(linkage_matrix, dtype=np.float64),
                    'silhouette_score': -1,
                    'valido': False,
                    'razon_rechazo': ', '.join(problemas)
                }

        except Exception as e:
            print(f"Error en cluster_hac_mejorado: {e}")
            return {
                'clusters': None,
                'centroides': None,
                'linkage_matrix': None,
                'silhouette_score': -1,
                'valido': False,
                'error': str(e)
            }

    def ploteo_graficos_hac(self, res, metodo: str):
        """Método interno para plotear dendrogramas, con manejo de errores."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            dendrogram(res, labels=self._df.index.tolist(), ax=ax)
            plt.title(f"Dendrograma - {metodo}")
            ax.grid(False)
            plt.show()
        except Exception as e:
            print(f"Error al plotear dendrograma ({metodo}): {e}")

    def cluster_hac(self, tipo: int):
        """Método interno para visualizar clusters HAC (método legacy), con manejo de errores."""
        try:
            # Usar ward como método por defecto
            grupos = fcluster(linkage(self._df, method='ward'),
                              3, criterion='maxclust')
            grupos = grupos - 1

            centros = np.array(pd.concat([
                self.centroide(0, self._df, grupos),
                self.centroide(1, self._df, grupos),
                self.centroide(2, self._df, grupos)
            ]))

            if tipo == 1:
                self.bar_plot(centros, self._df.columns.tolist(), scale=True)
            elif tipo == 2:
                self.bar_plot_detail(centros, self._df.columns.tolist())
            elif tipo == 3:
                self.radar_plot(centros, self._df.columns.tolist())

            plt.show()
        except Exception as e:
            print(f"Error al visualizar clusters HAC (legacy): {e}")

    def kmedias(self):
        """Realiza clustering con K-Means, con manejo de errores."""
        try:
            self.ploteo_kmedias()
        except Exception as e:
            print(f"Error en K-Means: {e}")

    def ploteo_kmedias(self):
        """Método interno para K-Means, con manejo de errores."""
        try:
            kmedias = KMeans(n_clusters=3, max_iter=500, n_init=150)
            kmedias.fit(self._df)

            pca = PCA(n_components=2)
            componentes = pca.fit_transform(self._df)

            fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
            colores = ['red', 'green', 'blue']
            colores_puntos = [colores[label]
                              for label in kmedias.predict(self._df)]

            ax.scatter(componentes[:, 0], componentes[:, 1], c=colores_puntos)
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.set_title('3 Cluster K-Medias')
            ax.grid(False)
            plt.show()

            centros = np.array(kmedias.cluster_centers_)
            self.bar_plot(centros, self._df.columns)
            self.bar_plot_detail(centros, self._df.columns)
            self.radar_plot(centros, self._df.columns)
            plt.show()
        except Exception as e:
            print(f"Error en ploteo K-Means: {e}")

    # def ploteo_kmedoids(self):
    #     """K-Medoids eliminado por compatibilidad de dependencias."""
    #     print("K-Medoids no disponible en esta versión del paquete.")

    def tsne(self, n_componentes: int = 2):
        """Realiza t-SNE para reducción de dimensionalidad, con manejo de errores."""
        try:
            tsne = TSNE(n_components=n_componentes)
            componentes = tsne.fit_transform(self._df)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            ax.scatter(componentes[:, 0], componentes[:, 1])
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.set_title('T-SNE')
            ax.grid(False)
            plt.show()
        except Exception as e:
            print(f"Error en t-SNE: {e}")

    # def umap(self, n_componentes: int = 2, n_neighbors: int = 15):
    #     """UMAP eliminado por compatibilidad de dependencias."""
    #     print("UMAP no disponible en esta versión del paquete.")


class Supervisado:
    """Clase base para métodos de aprendizaje supervisado."""

    def __init__(self):
        """Inicializa el supervisado."""
        pass

    def preparar_datos(self) -> tuple:
        """Prepara datos de ejemplo para pruebas, con manejo de errores."""
        try:
            X = np.random.rand(200, 5)
            y = np.random.choice([0, 1], 200)
            return train_test_split(X, y, test_size=0.25, random_state=42)
        except Exception as e:
            logger.error(f"Error al preparar datos: {e}")
            return None, None, None, None, None

    def modelo_knn(self, X_train, y_train, n_neighbors: int, algorithm: str):
        """Crea y entrena un modelo KNN, con manejo de errores."""
        try:
            return KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm).fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error en modelo KNN: {e}")
            return None

    def modelo_dt(self, X_train, y_train, min_samples_split: int, max_depth: int):
        """Crea y entrena un árbol de decisión, con manejo de errores."""
        try:
            return DecisionTreeClassifier(
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                random_state=42
            ).fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error en modelo Árbol de Decisión: {e}")
            return None

    def modelo_rf(self, X_train, y_train, n_estimators: int, min_samples_split: int, max_depth: int):
        """Crea y entrena un random forest, con manejo de errores."""
        try:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                random_state=42
            ).fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error en modelo Random Forest: {e}")
            return None

    def modelo_xg(self, X_train, y_train, n_estimators: int, min_samples_split: int, max_depth: int):
        """Crea y entrena un gradient boosting, con manejo de errores."""
        try:
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                random_state=42
            ).fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error en modelo Gradient Boosting: {e}")
            return None

    def modelo_ada(self, X_train, y_train, estimator, n_estimators: int):
        """Crea y entrena un AdaBoost, con manejo de errores."""
        try:
            return AdaBoostClassifier(
                estimator=estimator,
                n_estimators=n_estimators,
                random_state=42
            ).fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error en modelo AdaBoost: {e}")
            return None

    def predecir(self, model, X_test) -> np.ndarray:
        """Realiza predicciones con el modelo, con manejo de errores."""
        try:
            return model.predict(X_test)
        except Exception as e:
            logger.error(f"Error al predecir: {e}")
            return None

    def evaluar(self, y_test, y_pred, y):
        """Evalúa el modelo y muestra métricas, con manejo de errores."""
        try:
            mc = confusion_matrix(y_test, y_pred)
            indices = self.indices_general(mc, list(np.unique(y)))
            for k in indices:
                logger.info(f"{k}:\n{indices[k]}")
        except Exception as e:
            logger.error(f"Error al evaluar el modelo: {e}")
# --- Automatización: Pipeline de procesamiento ---


def pipeline_orquestador(path, num, target_col):
    def pruebas_estadisticas_auto(df, target_col=None):
        logger.info("\n--- Pruebas estadísticas automáticas ---")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        resultados = {}
        for col in num_cols:
            datos = df[col].dropna()
            if len(datos) < 8:
                logger.info(
                    f"Columna '{col}': muy pocos datos para pruebas de normalidad.")
                continue
            # Prueba de normalidad (Shapiro-Wilk)
            stat, p = shapiro(datos)
            normal = p > 0.05
            logger.info(
                f"Columna '{col}': p normalidad (Shapiro) = {p:.4f} => {'Normal' if normal else 'No normal'}")
            resultados[col] = {'normalidad_p': p, 'es_normal': normal}
        # Si hay columna objetivo y es categórica, comparar grupos
        if target_col and target_col in df.columns:
            y = df[target_col]
            if pd.api.types.is_numeric_dtype(y):
                logger.info(
                    "Columna objetivo numérica: pruebas de correlación/regresión sugeridas.")
            else:
                grupos = y.unique()
                if len(grupos) == 2:
                    # Prueba entre dos grupos
                    for col in num_cols:
                        datos1 = df[df[target_col] == grupos[0]][col].dropna()
                        datos2 = df[df[target_col] == grupos[1]][col].dropna()
                        if len(datos1) < 8 or len(datos2) < 8:
                            logger.info(
                                f"Columna '{col}': muy pocos datos para prueba entre grupos.")
                            continue
                        if resultados[col]['es_normal']:
                            stat, p = ttest_ind(
                                datos1, datos2, equal_var=False)
                            logger.info(
                                f"{col}: t-test entre grupos '{grupos[0]}' y '{grupos[1]}', p = {p:.4f}")
                        else:
                            stat, p = mannwhitneyu(datos1, datos2)
                            logger.info(
                                f"{col}: Mann-Whitney U entre grupos '{grupos[0]}' y '{grupos[1]}', p = {p:.4f}")
                elif len(grupos) > 2:
                    for col in num_cols:
                        datos_grupos = [df[df[target_col] == g]
                                        [col].dropna() for g in grupos]
                        if any(len(d) < 8 for d in datos_grupos):
                            logger.info(
                                f"Columna '{col}': muy pocos datos para prueba entre grupos.")
                            continue
                        if resultados[col]['es_normal']:
                            stat, p = f_oneway(*datos_grupos)
                            logger.info(
                                f"{col}: ANOVA entre grupos {grupos}, p = {p:.4f}")
                        else:
                            stat, p = kruskal(*datos_grupos)
                            logger.info(
                                f"{col}: Kruskal-Wallis entre grupos {grupos}, p = {p:.4f}")
                # MANOVA si hay varias variables numéricas y al menos 2 grupos
                if len(num_cols) > 1 and len(grupos) >= 2:
                    try:
                        datos_validos = df.dropna(
                            subset=num_cols + [target_col])
                        if datos_validos.shape[0] < 8:
                            logger.info("Muy pocos datos para MANOVA.")
                        else:
                            formula = ' + '.join(num_cols) + f' ~ {target_col}'
                            manova = MANOVA.from_formula(
                                formula, data=datos_validos)
                            res = manova.mv_test()
                            logger.info(f"MANOVA resultados:\n{res}")
                    except Exception as e:
                        logger.error(f"Error al ejecutar MANOVA: {e}")
        logger.info("--- Fin pruebas estadísticas ---\n")

    """
    Pipeline automatizado: carga → validación → EDA → modelado → evaluación → reporte.
    Detecta automáticamente si el problema es de regresión o clasificación según la columna objetivo.
    Args:
        path: ruta al archivo csv
        num: 1=coma, 2=punto y coma
        target_col: nombre de la columna objetivo
    """
    logger.info("Iniciando pipeline de procesamiento de datos...")
    ade = AnalisisDatosExploratorio(path, num)
    df = ade.df
    if df.empty:
        logger.error("No se pudo cargar el DataFrame. Pipeline abortado.")
        return
    # Soporte para target_col como lista (regresión multivariada)
    if isinstance(target_col, list):
        for col in target_col:
            if col not in df.columns:
                logger.error(
                    f"La columna objetivo '{col}' no está en el DataFrame.")
                return
    elif target_col not in df.columns:
        logger.error(
            f"La columna objetivo '{target_col}' no está en el DataFrame.")
        return
    """
    Pipeline automatizado: carga → validación → EDA → modelado → evaluación → reporte.
    Si target_col es None, ejecuta análisis no supervisado (clustering y reducción de dimensionalidad).
    Args:
        path: ruta al archivo csv
        num: 1=coma, 2=punto y coma
        target_col: nombre de la columna objetivo (opcional)
    """
    logger.info("Iniciando pipeline de procesamiento de datos...")
    ade = AnalisisDatosExploratorio(path, num)
    df = ade.df
    if df.empty:
        logger.error("No se pudo cargar el DataFrame. Pipeline abortado.")
        return
    # Validación simple
    logger.info(f"Columnas: {df.columns.tolist()}")
    logger.info(f"Nulos por columna: {df.isnull().sum().to_dict()}")
    # EDA
    ade.analisis()
    # Pruebas estadísticas automáticas
    pruebas_estadisticas_auto(df, target_col)
    if target_col is None:
        logger.info(
            "No se especificó columna objetivo. Ejecutando análisis no supervisado...")
        from sklearn.preprocessing import StandardScaler
        X = df.select_dtypes(include=[np.number]).dropna()
        if X.shape[1] < 2:
            logger.error(
                "Se requieren al menos 2 variables numéricas para clustering o reducción de dimensionalidad.")
            return
        X_scaled = StandardScaler().fit_transform(X)
        # KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        logger.info(
            f"Etiquetas KMeans: {np.unique(kmeans.labels_, return_counts=True)}")
        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(X_scaled)
        logger.info(
            f"Varianza explicada por PCA: {pca.explained_variance_ratio_}")
        # Puedes agregar más métodos aquí (HAC, t-SNE, UMAP, etc.)
    else:
        # Soporte para regresión multivariada
        if isinstance(target_col, list) and all(pd.api.types.is_numeric_dtype(df[col]) for col in target_col):
            logger.info("Detección: problema de regresión multivariada.")
            X = df.drop(columns=target_col)
            y = df[target_col]
            # Preprocesamiento simple: eliminar filas con nulos
            mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
            X = X.loc[mask]
            y = y.loc[mask]
            X = pd.get_dummies(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)
            from sklearn.linear_model import LinearRegression
            model = MultiOutputRegressor(
                LinearRegression()).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            from sklearn.metrics import mean_squared_error, r2_score
            for idx, col in enumerate(target_col):
                mse = mean_squared_error(y_test.iloc[:, idx], y_pred[:, idx])
                r2 = r2_score(y_test.iloc[:, idx], y_pred[:, idx])
                logger.info(
                    f"[Multivariada] {col} - MSE: {mse:.4f}, R2: {r2:.4f}")
        else:
            # Univariada o clasificación
            if isinstance(target_col, list):
                logger.error(
                    "Regresión multivariada solo soportada para variables objetivo numéricas.")
                return
            X = df.drop(columns=[target_col])
            y = df[target_col]
            # Detección automática de tipo de problema
            if pd.api.types.is_numeric_dtype(y):
                n_unicos = y.nunique()
                if n_unicos <= 10 and y.dtype in [int, 'int64', 'int32']:
                    problema = 'clasificacion'
                else:
                    problema = 'regresion'
            else:
                problema = 'clasificacion'
            logger.info(f"Tipo de problema detectado: {problema}")
            # Preprocesamiento simple: eliminar filas con nulos
            X = X.loc[~(X.isnull().any(axis=1) | y.isnull())]
            y = y.loc[X.index]
            # Codificar variables categóricas
            X = pd.get_dummies(X)
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)
            if problema == 'clasificacion':
                sup = Supervisado()
                model = sup.modelo_knn(
                    X_train, y_train, n_neighbors=3, algorithm="auto")
                y_pred = sup.predecir(model, X_test)
                sup.evaluar(y_test, y_pred, y)
            else:
                # Regresión simple con LinearRegression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"MSE: {mse:.4f}")
                logger.info(f"R2: {r2:.4f}")
    logger.info("Pipeline finalizado.")

    def knn(self, n_neighbors: int = 5):
        """Entrena y evalúa un modelo KNN con diferentes algoritmos, con manejo de errores."""
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for algorithm in algorithms:
            try:
                print(f"\nUsando algoritmo: {algorithm}")
                X_train, X_test, y_train, y_test, y = self.preparar_datos()
                model = self.modelo_knn(
                    X_train, y_train, n_neighbors, algorithm)
                y_pred = self.predecir(model, X_test)
                self.evaluar(y_test, y_pred, y)
            except Exception as e:
                print(f"Error en ciclo KNN ({algorithm}): {e}")

    def dt(self, min_samples_split: int = 2, max_depth: int = None):
        """Entrena y evalúa un árbol de decisión, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_dt(
                X_train, y_train, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)
        except Exception as e:
            print(f"Error en Árbol de Decisión: {e}")

    def rf(self, n_estimators: int = 100, min_samples_split: int = 2, max_depth: int = None):
        """Entrena y evalúa un random forest, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_rf(
                X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)
        except Exception as e:
            print(f"Error en Random Forest: {e}")

    def xg(self, n_estimators: int = 100, min_samples_split: int = 2, max_depth: int = 3):
        """Entrena y evalúa un gradient boosting, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_xg(
                X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)
        except Exception as e:
            print(f"Error en Gradient Boosting: {e}")

    def ada(self, n_estimators: int = 50):
        """Entrena y evalúa AdaBoost con diferentes estimadores base, con manejo de errores."""
        estimators = {
            "Decision Tree": DecisionTreeClassifier(min_samples_split=2, max_depth=4, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=4, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, min_samples_split=2, max_depth=4, random_state=42)
        }
        for name, estimator in estimators.items():
            try:
                print(f"\nUsando método: {name}")
                X_train, X_test, y_train, y_test, y = self.preparar_datos()
                model = self.modelo_ada(
                    X_train, y_train, estimator, n_estimators)
                y_pred = self.predecir(model, X_test)
                self.evaluar(y_test, y_pred, y)
            except Exception as e:
                print(f"Error en ciclo AdaBoost ({name}): {e}")

    def bm(self):
        """Realiza benchmarking de todos los modelos, con manejo de errores."""
        try:
            datos = {
                "PG": [0, 0, 0, 0, 0],
                "EG": [0, 0, 0, 0, 0],
                "PP": [0, 0, 0, 0, 0],
                "PN": [0, 0, 0, 0, 0]
            }

            tdatos = pd.DataFrame(
                datos,
                index=["AlgKnn", "AlgDT", "AlgRF",
                       "AlgXGBoost", "AlgADABoost"],
                columns=["PG", "EG", "PP", "PN"]
            )

            # Evaluar cada modelo y almacenar resultados
            modelos = [
                ("AlgKnn", self.knn_bm),
                ("AlgDT", self.dt_bm),
                ("AlgRF", self.rf_bm),
                ("AlgXGBoost", self.xg_bm),
                ("AlgADABoost", self.ada_bm)
            ]

            for nombre, metodo in modelos:
                try:
                    indices = metodo()
                    pp = indices['Precisión por categoría']
                    pn = indices['Precisión por categoría']

                    tdatos.loc[nombre, "PG"] = indices['Precisión Global']
                    tdatos.loc[nombre, "EG"] = indices['Error Global']
                    tdatos.loc[nombre, "PP"] = pp.iloc[0, 0] if isinstance(
                        pp, pd.DataFrame) else pp[0]
                    tdatos.loc[nombre, "PN"] = pn.iloc[0, 1] if isinstance(
                        pn, pd.DataFrame) else pn[1]
                except Exception as e:
                    print(f"Error en benchmarking de {nombre}: {e}")

            print(tdatos)
        except Exception as e:
            print(f"Error general en benchmarking: {e}")

    def knn_bm(self, n_neighbors: int = 3, algorithm: str = "ball_tree") -> dict:
        """Método interno para benchmarking de KNN, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_knn(X_train, y_train, n_neighbors, algorithm)
            y_pred = self.predecir(model, X_test)
            mc = confusion_matrix(y_test, y_pred)
            return self.indices_general(mc, list(np.unique(y)))
        except Exception as e:
            print(f"Error en benchmarking KNN: {e}")
            return {}

    def dt_bm(self, min_samples_split: int = 8, max_depth: int = 1) -> dict:
        """Método interno para benchmarking de árbol de decisión, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_dt(
                X_train, y_train, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            mc = confusion_matrix(y_test, y_pred)
            return self.indices_general(mc, list(np.unique(y)))
        except Exception as e:
            print(f"Error en benchmarking Árbol de Decisión: {e}")
            return {}

    def rf_bm(self, n_estimators: int = 200, min_samples_split: int = 9, max_depth: int = 8) -> dict:
        """Método interno para benchmarking de random forest, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_rf(
                X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            mc = confusion_matrix(y_test, y_pred)
            return self.indices_general(mc, list(np.unique(y)))
        except Exception as e:
            print(f"Error en benchmarking Random Forest: {e}")
            return {}

    def xg_bm(self, n_estimators: int = 100, min_samples_split: int = 5, max_depth: int = 1) -> dict:
        """Método interno para benchmarking de gradient boosting, con manejo de errores."""
        try:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_xg(
                X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            mc = confusion_matrix(y_test, y_pred)
            return self.indices_general(mc, list(np.unique(y)))
        except Exception as e:
            print(f"Error en benchmarking Gradient Boosting: {e}")
            return {}

    def ada_bm(self, estimator=None, n_estimators: int = 10) -> dict:
        """Método interno para benchmarking de AdaBoost, con manejo de errores."""
        try:
            if estimator is None:
                estimator = GradientBoostingClassifier(
                    n_estimators=100,
                    min_samples_split=5,
                    max_depth=1,
                    random_state=42
                )

            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modelo_ada(X_train, y_train, estimator, n_estimators)
            y_pred = self.predecir(model, X_test)
            mc = confusion_matrix(y_test, y_pred)
            return self.indices_general(mc, list(np.unique(y)))
        except Exception as e:
            print(f"Error en benchmarking AdaBoost: {e}")
            return {}


class ModelEvaluator:
    """Evaluador de modelos con búsqueda genética y exhaustiva de hiperparámetros."""

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray):
        """Inicializa el evaluador con los datos."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = self.get_models()
        self.param_grids_genetic = self.get_param_grids_genetic()
        self.param_grids_exhaustive = self.get_param_grids_exhaustive()

    def get_models(self) -> Dict[str, object]:
        """Devuelve los modelos a evaluar."""
        return {
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'Lasso': Lasso(random_state=42),
            'Ridge': Ridge(random_state=42),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBRegressor': XGBRegressor(random_state=42)
        }

    def get_param_grids_genetic(self) -> Dict[str, Dict]:
        """Define los espacios de búsqueda genética."""
        return {
            'LinearRegression': {
                "clf__copy_X": Categorical([True, False]),
                "clf__fit_intercept": Categorical([True, False]),
                "clf__positive": Categorical([True, False])
            },
            'DecisionTreeRegressor': {
                "clf__max_depth": Integer(3, 10),
                'clf__min_samples_split': Integer(2, 10),
                'clf__min_samples_leaf': Integer(1, 4),
                'clf__random_state': Categorical([42])
            },
            'RandomForestRegressor': {
                "clf__n_estimators": Integer(50, 100),
                "clf__max_depth": Integer(5, 10),
                'clf__min_samples_split': Integer(2, 5),
                'clf__random_state': Categorical([42])
            },
            'Lasso': {
                'clf__alpha': Continuous(1.0, 1.0),
                'clf__fit_intercept': Categorical([True, False]),
                'clf__max_iter': Integer(1000, 2000),
                'clf__tol': Continuous(0.0001, 0.001),
                'clf__selection': Categorical(['cyclic', 'random'])
            },
            'Ridge': {
                'clf__alpha': Continuous(1.0, 1.0),
                'clf__fit_intercept': Categorical([True, False]),
                'clf__tol': Continuous(0.0001, 0.001),
                'clf__solver': Categorical(['auto', 'svd', 'cholesky'])
            },
            'KNeighborsRegressor': {
                'clf__n_neighbors': Integer(3, 7),
                'clf__weights': Categorical(['uniform', 'distance']),
                'clf__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree'])
            },
            'XGBRegressor': {
                'clf__learning_rate': Continuous(0.01, 0.1),
                'clf__n_estimators': Integer(50, 100),
                'clf__max_depth': Integer(3, 5),
                'clf__subsample': Continuous(0.8, 1.0),
                'clf__colsample_bytree': Continuous(0.8, 1.0)
            }
        }

    def get_param_grids_exhaustive(self) -> Dict[str, Dict]:
        """Define los espacios de búsqueda exhaustiva."""
        return {
            'LinearRegression': {
                "clf__copy_X": [True, False],
                "clf__fit_intercept": [True, False],
                "clf__positive": [True, False]
            },
            'DecisionTreeRegressor': {
                "clf__max_depth": [3, 5, 7, 10],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__random_state': [42]
            },
            'RandomForestRegressor': {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [5, 10],
                'clf__min_samples_split': [2, 5],
                'clf__random_state': [42]
            },
            'Lasso': {
                'clf__alpha': [1.0],
                'clf__fit_intercept': [True, False],
                'clf__max_iter': [1000, 2000],
                'clf__tol': [0.0001, 0.001],
                'clf__selection': ['cyclic', 'random']
            },
            'Ridge': {
                'clf__alpha': [1.0],
                'clf__fit_intercept': [True, False],
                'clf__tol': [0.0001, 0.001],
                'clf__solver': ['auto', 'svd', 'cholesky']
            },
            'KNeighborsRegressor': {
                'clf__n_neighbors': [3, 5, 7],
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'XGBRegressor': {
                'clf__learning_rate': [0.01, 0.1],
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [3, 5],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0]
            }
        }

    def genetic_search(self) -> Dict[str, Dict]:
        """Realiza búsqueda genética de hiperparámetros, con manejo de errores."""
        results = {}
        for name, model in self.models.items():
            try:
                lasso_cv = LassoCV(cv=5)
                lasso_cv.fit(self.X_train, self.y_train)
                f_selection = SelectFromModel(lasso_cv)

                X_train_fs = f_selection.transform(self.X_train)
                X_test_fs = f_selection.transform(self.X_test)

                pl = Pipeline([
                    ('fs', f_selection),
                    ('clf', model),
                ])

                print(f"Entrenando {name} con método genético...")

                evolved_estimator = GASearchCV(
                    estimator=pl,
                    cv=5,
                    scoring="neg_mean_squared_error",
                    population_size=10,
                    generations=5,
                    tournament_size=3,
                    elitism=True,
                    crossover_probability=0.8,
                    mutation_probability=0.1,
                    param_grid=self.param_grids_genetic[name],
                    algorithm="eaSimple",
                    n_jobs=-1,
                    error_score='raise',
                    verbose=True
                )

                evolved_estimator.fit(X_train_fs, self.y_train)

                results[name] = {
                    'best_params': evolved_estimator.best_params_,
                    'estimator': evolved_estimator.best_estimator_,
                    'score': evolved_estimator.best_score_
                }
            except Exception as e:
                print(f"Error en búsqueda genética para {name}: {e}")
        return results

    def exhaustive_search(self) -> Dict[str, Dict]:
        """Realiza búsqueda exhaustiva de hiperparámetros, con manejo de errores."""
        results = {}
        for name, model in self.models.items():
            try:
                lasso_cv = LassoCV(cv=5)
                lasso_cv.fit(self.X_train, self.y_train)
                f_selection = SelectFromModel(lasso_cv)

                X_train_fs = f_selection.transform(self.X_train)
                X_test_fs = f_selection.transform(self.X_test)

                pl = Pipeline([
                    ('clf', model),
                ])

                print(f"Entrenando {name} con método exhaustivo...")

                grid_search = GridSearchCV(
                    estimator=pl,
                    param_grid=self.param_grids_exhaustive[name],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train_fs, self.y_train)

                results[name] = {
                    'best_params': grid_search.best_params_,
                    'estimator': grid_search.best_estimator_,
                    'score': grid_search.best_score_
                }
            except Exception as e:
                print(f"Error en búsqueda exhaustiva para {name}: {e}")
        return results


class RegressionEvaluator(ModelEvaluator):
    """Evaluador especializado para modelos de regresión."""

    def get_models(self) -> Dict[str, object]:
        """Modelos específicos para regresión."""
        return {
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'Lasso': Lasso(random_state=42),
            'Ridge': Ridge(random_state=42),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBRegressor': XGBRegressor(random_state=42)
        }

    def evaluate(self, results: Dict[str, Dict]):
        """Evalúa los resultados de la búsqueda de hiperparámetros, con manejo de errores."""
        for name, result in results.items():
            try:
                model = result['estimator']
                y_pred = model.predict(self.X_test)

                print(f"\nEvaluación para {name}:")
                print(f"Mejores parámetros: {result['best_params']}")
                print(f"R2 Score: {r2_score(self.y_test, y_pred):.4f}")
                print(f"MSE: {mean_squared_error(self.y_test, y_pred):.4f}")
            except Exception as e:
                print(f"Error al evaluar modelo de regresión {name}: {e}")


class ClassificationEvaluator(ModelEvaluator):
    """Evaluador especializado para modelos de clasificación."""

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray):
        """Inicializa el evaluador con los datos."""
        super().__init__(X_train, X_test, y_train, y_test)
        self.models = self.get_models()
        self.param_grids_genetic = self.get_param_grids_genetic()
        self.param_grids_exhaustive = self.get_param_grids_exhaustive()

    def get_models(self) -> Dict[str, object]:
        """Modelos específicos para clasificación."""
        return {
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier()
        }

    def get_param_grids_genetic(self) -> Dict[str, Dict]:
        """Espacios de búsqueda genética para clasificación."""
        return {
            'DecisionTreeClassifier': {
                'criterion': Categorical(['gini', 'entropy']),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 20),
                'class_weight': Categorical(['balanced', None])
            },
            'RandomForestClassifier': {
                'n_estimators': Integer(50, 500),
                'criterion': Categorical(['gini', 'entropy']),
                'max_depth': Integer(3, 15),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 10),
                'class_weight': Categorical(['balanced', 'balanced_subsample', None])
            },
            'AdaBoostClassifier': {
                'n_estimators': Integer(50, 500),
                'learning_rate': Continuous(0.01, 1.0),
                'algorithm': Categorical(['SAMME', 'SAMME.R'])
            },
            'GradientBoostingClassifier': {
                'learning_rate': Continuous(0.01, 0.2),
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(3, 10),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 20)
            },
            'KNeighborsClassifier': {
                'n_neighbors': Integer(1, 30),
                'weights': Categorical(['uniform', 'distance']),
                'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])
            }
        }

    def get_param_grids_exhaustive(self) -> Dict[str, Dict]:
        """Espacios de búsqueda exhaustiva para clasificación."""
        return {
            'DecisionTreeClassifier': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            },
            'RandomForestClassifier': {
                'n_estimators': [100, 200, 300],
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'AdaBoostClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            },
            'GradientBoostingClassifier': {
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        }

    def evaluate(self, results: Dict[str, Dict]):
        """Evalúa los resultados de la búsqueda de hiperparámetros, con manejo de errores."""
        for name, result in results.items():
            try:
                model = result['estimator']
                y_pred = model.predict(self.X_test)
                mc = confusion_matrix(self.y_test, y_pred)

                print(f"\nEvaluación para {name}:")
                print(f"Mejores parámetros: {result['best_params']}")
                print("Matriz de Confusión:")
                print(mc)

                precision = np.diag(mc) / np.sum(mc, axis=1)
                print(f"Precisión por clase: {precision}")
                print(
                    f"Precisión global: {np.sum(np.diag(mc)) / np.sum(mc):.4f}")
            except Exception as e:
                print(f"Error al evaluar modelo de clasificación {name}: {e}")
