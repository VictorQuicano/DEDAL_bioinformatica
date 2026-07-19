# Análisis del artículo seleccionado — Trabajo Final (Opción 2.1)

**Curso:** Bioinformática — Escuela Profesional de Ciencia de la Computación (UNSA)
**Docente:** Rolando Jesús Cárdenas Talavera
**Opción:** 2.1 — Trabajo de Investigación e Implementación

---

## 1. Identificación del artículo

**Título:** *Deep embedding and alignment of protein sequences*
**Autores:** Felipe Llinares-López, Quentin Berthet, Mathieu Blondel, Olivier Teboul, Jean-Philippe Vert (Google Research)
**Revista:** Nature Methods (versión final 2023; publicado primero como preprint en 2022)
**Modelo/método:** DEDAL — *Deep Embedding and Differentiable Alignment*

**Enlaces del artículo:**
- Nature Methods (versión final): https://www.nature.com/articles/s41592-022-01700-2
- bioRxiv (preprint, acceso libre): https://www.biorxiv.org/content/10.1101/2021.11.15.468653v2.full

**Enlaces de código y modelo:**
- Repositorio oficial (Google Research monorepo): https://github.com/google-research/google-research/tree/master/dedal
- Modelo preentrenado (TensorFlow Hub): https://tfhub.dev/google/dedal/3

---

## 2. Problema abordado

DEDAL ataca el problema del **alineamiento de secuencias de proteínas y la detección de homología remota**: dado un par de secuencias de aminoácidos, encontrar el mejor alineamiento local entre ellas y determinar si son evolutivamente homólogas, incluso cuando la similitud de secuencia es muy baja (identidad de secuencia < 10%), caso en el que los métodos clásicos de alineamiento fallan con frecuencia.

Este es, conceptualmente, el mismo problema que se trabajó en los laboratorios 02 y 03 del curso (alineamiento local con Smith-Waterman y global con Needleman-Wunsch), pero llevado al límite donde la matriz de sustitución fija (tipo BLOSUM) deja de ser suficiente.

## 3. Cómo se usa la Inteligencia Artificial para resolverlo

La idea central de DEDAL es que **no reemplaza el algoritmo de Smith-Waterman**, sino que lo alimenta con mejores parámetros:

1. Un **transformer encoder** (arquitectura tipo BERT) preentrenado sobre secuencias de proteínas genera un *embedding* contextual de cada aminoácido de las dos secuencias a alinear.
2. A partir de esos embeddings, una capa adicional calcula, **para ese par específico de secuencias**, una matriz de sustitución y unos parámetros de gap (apertura/extensión) *ad hoc* — en lugar de usar una matriz fija global como BLOSUM62.
3. Con esa matriz específica del par, se ejecuta el **algoritmo de Smith-Waterman estándar** (el mismo de `lab02.cpp`/`lab03.cpp`, con match/mismatch/gap y traceback) para obtener el alineamiento local óptimo.
4. Durante el *entrenamiento* (no necesario para nuestra replicación), el Smith-Waterman se reemplaza por una versión diferenciable (suavizada) para poder propagar gradientes y ajustar los pesos del transformer.

En resumen: **SW clásico con matriz fija → SW con matriz aprendida por una red neuronal, específica para cada par de secuencias.**

## 4. Metodología de los autores (detallada)

### 4.1 Datos de entrenamiento
- **UniRef50** (release de marzo de 2018): ~30 millones de secuencias de proteínas no redundantes, usadas para preentrenar el transformer con un objetivo de *masked language modeling* (similar a BERT, pero sobre secuencias de aminoácidos).
- **Pfam-A 34.0 (seed alignments)**: familias de proteínas con alineamientos de referencia curados manualmente, usados para entrenar la tarea de alineamiento y de detección de homología.

### 4.2 Arquitectura
- Encoder tipo transformer que produce embeddings por posición para cada secuencia.
- Una cabeza de "alineamiento" que combina los embeddings de ambas secuencias para producir matrices de score de sustitución y de gap específicas del par.
- Una cabeza de "homología" que produce un score/logit de si el par es homólogo o no.
- Sobre las matrices de score, un módulo de Smith-Waterman (diferenciable en entrenamiento, estándar en inferencia) calcula el alineamiento final.

### 4.3 Entrenamiento (referencial, no se replica)
- Entrenado con **32 núcleos de TPU v3** — fuera del alcance de un grupo de pregrado, y **no es necesario** replicarlo: el modelo preentrenado está disponible públicamente.

### 4.4 Evaluación
- Comparado contra **Smith-Waterman clásico con matriz de sustitución fija**, usando las familias de matrices **BLOSUM**, **VTML** y **PFASUM**. El baseline reportado en el paper usa la mejor combinación encontrada entre más de 1000 combinaciones de matriz + parámetros de gap (ganadora: **PFASUM60**).
- Métrica principal: **AUPRC** (área bajo la curva precisión-recall) para detección de homología remota, y **porcentaje de columnas de alineamiento correctas** respecto al alineamiento de referencia de Pfam, particionado por bandas de identidad de secuencia (PID).

## 5. Resultados obtenidos por los autores

| Métrica | Smith-Waterman + PFASUM60 (mejor baseline clásico) | DEDAL |
|---|---|---|
| AUPRC, dominios Pfam, PID < 0.1 (homólogos remotos) | 0.389 | **0.988** (+154%) |
| AUPRC, dominios extendidos, PID < 0.1 | 0.501 | **0.987** (+97%) |
| Calidad de alineamiento en homólogos remotos | — | mejora de **2× a 3×** respecto a métodos existentes |

**Conclusión de los autores:** la brecha de desempeño entre DEDAL y el Smith-Waterman clásico es pequeña quando las secuencias son muy similares (alta identidad), pero se vuelve muy grande en el régimen de homología remota (baja identidad), que es precisamente el caso más difícil y más relevante en la práctica (por ejemplo, para anotar función de proteínas poco caracterizadas).

## 6. Cómo se puede replicar (guía práctica para el grupo)

**Importante:** no se debe intentar reentrenar DEDAL (requeriría UniRef50 completo + TPUs). La replicación factible y suficiente para el trabajo es la de **inferencia + evaluación comparativa**, que es exactamente lo que pide la consigna del curso.

### Paso 1 — Preparar el entorno
```bash
git clone https://github.com/google-research/google-research.git
cd google-research
pip install -r dedal/requirements.txt
pip install tensorflow tensorflow-hub
```

### Paso 2 — Cargar el modelo preentrenado y correr inferencia
```python
import tensorflow_hub as hub
from dedal import infer

dedal_model = hub.load('https://tfhub.dev/google/dedal/3')

protein_a = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV..."   # secuencia 1 (aminoácidos)
protein_b = "MKTIYIAKQRQISFVKSHFSRQLEERLGLIEV..."   # secuencia 2 (aminoácidos)

inputs = infer.preprocess(protein_a, protein_b)   # padding a 512 tokens
output = dedal_model(inputs)
alignment = infer.Alignment(protein_a, protein_b, *output)
```
La salida incluye: el alineamiento local predicho, el score de homología y las matrices de sustitución/gap específicas del par (esto es justamente lo interesante para comparar contra una matriz fija tipo BLOSUM62).

### Paso 3 — Implementar el baseline propio
Extender el Smith-Waterman ya construido en `lab02.cpp`/`lab03.cpp`:
- Cambiar el esquema simple de match/mismatch por una **matriz BLOSUM62** (matriz de sustitución estándar para proteínas).
- Mantener gap afín (apertura + extensión) y traceback, tal como ya está implementado.

### Paso 4 — Preparar el set de evaluación
- Descargar un subconjunto pequeño y manejable de **Pfam-A seed** (los identificadores usados por los propios autores están enlazados en el repo, vía Google Drive).
- Alternativamente, usar pares de proteínas de benchmarks públicos más pequeños referenciados en el paper (p. ej. Malidup/Malisam), que tienen alineamientos de referencia curados y son mucho más manejables en tamaño para un grupo de pregrado.

### Paso 5 — Correr la comparación
Para cada par de proteínas del set de evaluación:
1. Alinear con el SW propio (BLOSUM62 fijo).
2. Alinear con DEDAL (inferencia).
3. Comparar contra el alineamiento de referencia (ground truth de Pfam/Malidup) y calcular: % de columnas correctas, score de alineamiento, y (si se separan pares homólogos/no homólogos) la curva precisión-recall / AUPRC.
4. Repetir separando los pares por bandas de identidad de secuencia (alta vs. baja identidad) para reproducir, a menor escala, la tabla de resultados del paper.
5. Medir tiempo de ejecución de ambos métodos sobre el mismo hardware.

### Restricciones técnicas a documentar
- El modelo preentrenado hace *padding* de las secuencias a **512 tokens**: secuencias más largas requieren truncamiento — un caso análogo al parámetro `limite` que ya se usó en `lab07.cpp` para acotar secuencias largas.
- La arquitectura transformer puede ser lenta en CPU; para el volumen de pares que un grupo de pregrado necesita evaluar (decenas a un par de cientos), es suficiente un entorno gratuito tipo Google Colab (CPU o GPU gratuita), sin necesidad de infraestructura costosa.

## 7. Plan de comparación (SW propio vs. DEDAL)

| Elemento | Nuestra implementación | DEDAL (autores) |
|---|---|---|
| Algoritmo base | Smith-Waterman clásico (programación dinámica, traceback) | Smith-Waterman clásico (mismo algoritmo) |
| Matriz de sustitución | BLOSUM62 fija para todos los pares | Matriz aprendida, específica por par, generada por transformer |
| Parámetros de gap | Fijos (apertura/extensión definidos manualmente) | Aprendidos, específicos por par |
| Necesidad de entrenamiento | Ninguna | Ninguna para nosotros (se usa el modelo ya preentrenado) |
| Transparencia | Totalmente auditable línea por línea | Caja "semi-negra" (embeddings + red), aunque el paso final sigue siendo SW |
| Dependencias | Solo C++ estándar | TensorFlow + TensorFlow Hub |

**Métricas a reportar:** AUPRC de detección de homología, % de columnas de alineamiento correctas frente a referencia, score bruto de alineamiento, tiempo de ejecución por par.

## 8. Ventajas y desventajas

**Ventajas de nuestra implementación (SW clásico + BLOSUM62):**
- Totalmente transparente y auditable, sin dependencias de frameworks de deep learning.
- Tiempo de ejecución predecible y bajo para secuencias cortas/medianas.
- No requiere GPU ni modelo preentrenado descargado.

**Desventajas frente a DEDAL:**
- La matriz de sustitución fija no captura el contexto evolutivo específico de cada par, lo que se traduce en una caída notable de desempeño en homólogos remotos (la brecha documentada por los autores: AUPRC 0.389 vs. 0.988).
- No generaliza a relaciones no explícitamente reflejadas en los valores fijos de la matriz.

**Ventajas de DEDAL:**
- Score de sustitución/gap adaptativo por par mejora sustancialmente la sensibilidad en homología remota.
- Sigue produciendo un alineamiento interpretable (no es una "caja negra" total, ya que el paso final es Smith-Waterman estándar).

**Desventajas de DEDAL:**
- Requiere preentrenamiento costoso (irreproducible para el grupo, aunque no es necesario reproducirlo).
- Limitación de longitud de secuencia por el padding a 512 tokens.
- Menor transparencia que el SW clásico en la etapa de generación de la matriz de score (esa parte sí es una red neuronal).

## 9. Justificación de la elección

1. Es un artículo de investigación real, verificable y de alto impacto (Nature Methods), publicado dentro del rango 2021-2026 solicitado.
2. Tiene código oficial público y mantenido por los propios autores (Google Research), no de terceros.
3. Usa datasets estándar y públicos de la comunidad de bioinformática (UniRef50, Pfam-A).
4. Es el único candidato evaluado donde el algoritmo central del curso (Smith-Waterman) **es parte literal de la arquitectura del método**, no solo una alternativa elegida por analogía — lo que hace la comparación exacta y fácil de explicar y defender.
5. Permite una replicación completamente factible (solo inferencia, sin entrenar) con resultados cuantitativos claros y ya reportados por los autores para contrastar.
6. No requiere infraestructura costosa: basta con el modelo preentrenado vía TensorFlow Hub y un entorno gratuito tipo Google Colab.

## 10. Evaluación de viabilidad

**9 / 10** — Se resta un punto por la posible fricción de instalación de TensorFlow/TensorFlow Hub y por la necesidad de acotar cuidadosamente el tamaño del subconjunto de evaluación de Pfam-A para no consumir tiempo excesivo en descarga y preprocesamiento de datos.
