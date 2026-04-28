# Proyecto SAR - Buscador de Wikipedia (Curso 2025-2026)

Este proyecto consiste en la implementación de un sistema de recuperación de información eficiente para artículos de la Wikipedia en formato JSON. Es un proyecto coordinado entre las asignaturas de **Sistemas de Almacenamiento y Recuperación de Información (SAR)**.

---

## 📋 Descripción del Proyecto
El objetivo es construir un motor de búsqueda capaz de procesar grandes colecciones de documentos, indexarlos y permitir recuperarlos mediante consultas complejas. El sistema soporta:
* **Consultas Booleanas**: Uso de operadores `AND` y `NOT`.
* **Búsquedas Posicionales**: Localización de frases exactas mediante el uso de comillas.
* **Búsqueda Semántica**: Recuperación basada en el significado y similitud vectorial de los textos.

---

## 🛠️ Tecnologías y Librerías
El proyecto está desarrollado en **Python** y utiliza las siguientes bibliotecas:
* `nltk`: Para la segmentación de frases (`sent_tokenize`).
* `spacy`: Para la generación de embeddings de texto.
* `scikit-learn`: Para la gestión de estructuras de búsqueda rápida KDTree.
* `pickle`: Para la serialización y persistencia de los índices en disco.
* `argparse`: Para la gestión de argumentos por línea de comandos.

---

## 🚀 Guía de Uso

### 1. Indexación de Documentos
Para generar un índice a partir de una carpeta con archivos JSON de Wikipedia:

```bash
python SAR_Indexer.py <dir_articulos> <nombre_indice> [-P] [-S]
```

#### Parámetros y Opciones

- `dir`: Directorio con los artículos en formato JSON.  
- `index`: Nombre del fichero donde se guardará el índice.  

- `-P` / `--positional`: *(Opcional)* Calcula el índice posicional para permitir búsquedas de frases.  
- `-S` / `--semantic`: *(Opcional)* Calcula el índice semántico para búsquedas por similitud.  


### 2. Recuperación (Búsqueda)

Para buscar artículos en el índice creado:

```bash
python SAR_Searcher.py <nombre_indice> [opciones]
```

#### Opciones
* -Q "consulta": Ejecuta una consulta directa

  ej:
  ```bash
  python SAR_Searcher.py index -Q "fin de semana"
  ```
* -L lista.txt: Procesa una lista de consultas desde un fichero.
* -T test.txt: Modo de evaluación para comparar resultados con una referencia.
* -C / --count: Muestra solo el número de artículos recuperados.
* -A / --all: Muestra todos los resultados (por defecto solo se muestran los 10 primeros).
* -R / --semantic_ranking: Ordena los resultados de la búsqueda binaria por similitud semántica.

### 3. Evaluación Oficial (Modo Test)

Este es el método principal para comprobar si la implementación es correcta y coincide con los resultados de referencia.  
```bash
python SAR_Searcher.py mi_indice.bin -T test_list.txt
```
* **Si todo es correcto:** El programa mostrará el mensaje "Parece que todo ha ido bien, buen trabajo!".  
* **Si hay errores:** Se resaltarán las consultas donde el número de artículos recuperados no coincida con la referencia.

---

## 👥 Autores y Reparto de Funcionalidades
Este proyecto ha sido desarrollado por:

### Eva Borrajo 
#### Funcionalidades Obligatorias:
* tokenize(self, text): Limpieza de símbolos y normalización.
* index_file(self, filename): Lectura del JSON, uso de parse_article y construcción del índice.
* get_posting(self, term): Recuperar la posting list.
* and_posting(self, p1, p2): Intersección de listas (merge sin sets).
* show_stats(self): Visualización de estadísticas.
#### Parte de la Ampliación (Semántica):
* update_chuncks(self, txt, artid): División en frases con nltk.

### Alberto Delgado
#### Funcionalidades Obligatorias:
* index_dir(self, root, args): Recorrido recursivo de carpetas y flags.
* reverse_posting(self, p): Operador NOT.
* solve_query(self, query): Procesamiento de la consulta (orden izquierda → derecha).
* solve_and_show(self, query): Formateo de resultados (ID, Título, URL).
#### Parte de la Ampliación (Semántica):
* create_kdtree(self): Generación del índice vectorial (fit del modelo).
* solve_semantic_query(self, query): Búsqueda por similitud con KDTree.
* semantic_reranking(self, query, articles): Reordenación semántica.
