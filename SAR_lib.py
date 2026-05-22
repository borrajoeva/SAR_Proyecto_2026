# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel

import math

## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        #1 - completar
        # Extraemos las frases del artículo
        cks= nltk.sent_tokenize(txt)

        #2 - completar

        # Obtenemos los embeddings de todas las frases de golpe
        embs_del_articulo = self.model.get_embeddings(cks)

        for i, c in enumerate(cks):
            self.chuncks.append(c)
            self.chunck_index.append(artid)
            self.embeddings.append(embs_del_articulo[i])
            
        self.artid_to_emb[artid] = embs_del_articulo

        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")

        # 1. Se llama al método fit pasándole la lista con todos los embeddings de las frases
        # 2. Se almacena opcionalmente el KDTree generado en el atributo de la clase
        if self.embeddings:
            self.kdtree = self.model.fit(self.embeddings)

        print("done!")


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()
        
        # COMPLETAR

        # 1 y 2
        top_k = self.MAX_EMBEDDINGS
        resultados = self.model.query(query, top_k=top_k)

        # 3 y 4
        total_embeddings = len(self.embeddings)

        while (self.semantic_threshold is not None and 
               resultados[-1][0] <= self.semantic_threshold and 
               top_k < total_embeddings):
            
            # Volvemos al paso 2 aumentando top_k
            top_k += self.MAX_EMBEDDINGS
            
            # Volvemos al paso 1 ejecutando la query con el nuevo top_k
            resultados = self.model.query(query, top_k=top_k)
            
            if len(resultados) < top_k:
                break
        # 5
        lista_final=[]
        vistos = set() # Usamos un set para que la búsqueda sea ultrarrápida
        
        for dist, idx_chunk in resultados:
            # Si hay un umbral definido, descartamos estrictamente los que lo superen
            if self.semantic_threshold is not None and dist > self.semantic_threshold:
                continue 
                
            # Recuperamos a qué artículo pertenece esta frase
            id_articulo = self.chunck_index[idx_chunk]
            
            # Añadimos el artículo solo si no lo hemos visto ya (mantenemos el orden de relevancia)
            if id_articulo not in vistos:
                vistos.add(id_articulo)
                lista_final.append(id_articulo)

        return lista_final



    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()
        # COMPLETAR
        
        # Convertimos la lista original a un set para hacer comprobaciones ultra rápidas
        articulos_buscados = set(articles)
        
        # 1 y 2 - Primera consulta con top_k inicial
        top_k = self.MAX_EMBEDDINGS
        resultados = self.model.query(query, top_k=top_k)

        # Extraemos los artículos recuperados hasta el momento
        articulos_recuperados = set()
        for _, idx_chunk in resultados:
            articulos_recuperados.add(self.chunck_index[idx_chunk])

        total_embeddings = len(self.embeddings)

        # 3 - Bucle para asegurar que recuperamos todos los artículos de la lista binaria
        # Condición: "Mientras NO todos los artículos buscados estén dentro de los recuperados..."
        while not articulos_buscados.issubset(articulos_recuperados) and top_k < total_embeddings:
            top_k += self.MAX_EMBEDDINGS
            resultados = self.model.query(query, top_k=top_k)
            
            # Actualizamos nuestro set de control
            articulos_recuperados = set()
            for _, idx_chunk in resultados:
                articulos_recuperados.add(self.chunck_index[idx_chunk])
                
            if len(resultados) < top_k:
                break

        # 4 - Utilizar la lista ordenada del kdtree para reordenar la lista "articles"
        lista_ordenada = []
        vistos = set()

        # Recorremos los resultados del KDTree (que vienen ordenados del más similar al menos)
        for _, idx_chunk in resultados:
            id_art = self.chunck_index[idx_chunk]

            # Solo nos interesan los artículos que nos pasaron por parámetro
            if id_art in articulos_buscados and id_art not in vistos:
                lista_ordenada.append(id_art)
                vistos.add(id_art)

        # Casuística de seguridad extrema: 
        # Si por algún casual de la vida un artículo no generó embeddings (ej: texto vacío),
        # lo añadimos al final para no perder el resultado de la búsqueda booleana.
        for art in articles:
            if art not in vistos:
                lista_ordenada.append(art)

        return lista_ordenada

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################

        # Una vez recorridos e indexados todos los ficheros, 
        # generamos el modelo KDTree si la opción semántica está activa.
        if self.semantic:
            self.create_kdtree()
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """
         #
        # 
        # Solo se debe indexar el contenido self.DEFAULT_FIELD
        #
        #
        #
        #################
        ### COMPLETAR ###
        #################

        #Sacamos tamaño actual de self.docs (numero total de ficheros registrados)
        docid = len(self.docs)
        #Asignamos el ID calculado al fichero y lo registramos
        self.docs[docid] = filename 

        # Recorremos cada linea del archivo, cada linea es un objeto JSON
        for i, line in enumerate(open(filename)): 

            # Con parse_article convertimos esa linea leida en un diccionario con campos
            j = self.parse_article(line)
            url = j['url']

            # Comprobamos si el artículo ya está indexado
            if not self.already_in_index(j):
                id_articulo = len(self.articles)
                self.articles[id_articulo] = {
                'title': j['title'], 
                'url': url,
                'docid': docid,
                'pos': i 
                }

                self.urls.add(url)

                palabras = self.tokenize(j[self.DEFAULT_FIELD])
                palabras_unicas = set(palabras)

                if not self.positional:
                    # Analizamos las palabras si el artículo es nuevo
                    for p in palabras_unicas:
                        if p not in self.index:
                            self.index[p] = [id_articulo]
                        else:
                            self.index[p].append(id_articulo)

                else:
                    for pos_en_texto, k in enumerate(palabras):
                        if k not in self.index:
                            self.index[k] = [[id_articulo, [pos_en_texto]]]

                        else:
                            if self.index[k][-1][0] == id_articulo:
                                self.index[k][-1][1].append(pos_en_texto)
                            else:
                                self.index[k].append([id_articulo, [pos_en_texto]])

                # AMPLIACIÓN SEMÁNTICA: Extrae y guarda las frases del artículo
                if self.semantic:
                    self.update_chuncks(j[self.DEFAULT_FIELD], id_articulo)

       


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        nump=len(self.index) 
        numart=len(self.articles) 
        numdocs=len(self.docs) 
        numpostings= 0

        # Calculamos el gran total de postings
        for p in self.index:
            numpostings += len(self.get_posting(p))

        # Imprimimos los totales globales una sola vez
        print(f"Ficheros indexados: {numdocs}")
        print(f"Atículos indexados: {numart}")
        print(f"Vocabulario: {nump}")
        print(f"Número de postings: {numpostings}")

        # Preparamos la muestra alfabética de las 10 primeras palabras
        listakeys = sorted(self.index.keys())
        listakeys = listakeys[:10]

        # Imprimimos cada palabra de la muestra con su tamaño individual
        for p in listakeys:
            print(f"{p}: {len(self.get_posting(p))}")
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        if query is None or len(query) == 0:
            return [], {}

        # Separamos los tokens respetando lo que esté entre comillas dobles
        tokens = re.findall(r'".*?"|\S+', query)

        res = None
        es_not = False

        for token in tokens:
            if token == "NOT":
                es_not = True
                continue

            # 1. Comprobamos si es una búsqueda posicional (comienza y termina por comillas)
            if token.startswith('"') and token.endswith('"'):
                frase = token[1:-1] # Retiramos las comillas
                terminos_frase = self.tokenize(frase)
                p_actual = self.get_positionals(terminos_frase)
                
            # 2. Si no tiene comillas, es una búsqueda de un solo término
            else:
                terminos_limpios = self.tokenize(token)
                if len(terminos_limpios) > 0:
                    p_actual = self.get_posting(terminos_limpios[0])
                else:
                    p_actual = []

            # 3. Aplicamos el operador NOT si venía precedido por él
            if es_not:
                p_actual = self.reverse_posting(p_actual)
                es_not = False

            # 4. Acumulamos usando AND implícito
            if res is None:
                res = p_actual
            else:
                res = self.and_posting(res, p_actual)

        # 5. AMPLIACIÓN SEMÁNTICA: Si el reranking está activo, reordenamos los resultados
        if self.semantic_ranking and res:
            res = self.semantic_reranking(query, res)

        return res, {}


    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        # Si la palabra existe en el índice invertido, devolvemos su lista de artículos
        if term in self.index:
            if not self.positional:
                # Devolvemos la lista tal cual
                return self.index[term]
            else:
                lista_limpia = []
                for art in self.index[term]:
                    lista_limpia.append(art[0])
                return lista_limpia

        # Si no existe, devolvemos una lista vacía de forma segura
        else:
            return []
        



    def get_positionals(self, terms:str):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        #################################
        ## COMPLETAR PARA POSICIONALES ##
        #################################
        pass



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        res = []
        i = 0 # Puntero para recorrer la lista p (que ya viene ordenada)
        total_articulos = len(self.articles)
        
        # Recorremos todos los IDs de artículos posibles
        for artid in range(total_articulos):
            # Si el artículo actual es igual al que apunta el puntero de p, lo saltamos
            if i < len(p) and artid == p[i]:
                i += 1 
            else:
                # Si no está en p, lo guardamos en el resultado
                res.append(artid)
                
        return res


    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        res = []
        i = 0
        j = 0

        # Calculamos el tamaño del salto para cada lista
        salto_p1 = int(math.sqrt(len(p1))) if len(p1) > 0 else 1
        salto_p2 = int(math.sqrt(len(p2))) if len(p2) > 0 else 1

        while i < len(p1) and j < len(p2):
            
            # CASO 1: Los números en los que están los punteros son iguales
            if p1[i] == p2[j]:
                # Guardamos el número en 'res' 
                res.append(p1[i])
                # Avanzamos el puntero i
                i+=1
                # Avanzamos el puntero j
                j+=1
                

            # CASO 2: El número de p1 es menor que el de p2
            elif p1[i] < p2[j]:
                
                # Avanzamos el puntero i
                if i + salto_p1 < len(p1) and p1[i + salto_p1] <= p2[j]:
                    i+=salto_p1
                else:
                    i+=1
               
                
            # CASO 3: El número de p2 es menor que el de p1
            else:
                # Avanzamos solo el puntero j
                if j + salto_p2 < len(p2) and p2[j + salto_p2] <= p1[i]:
                    j+=salto_p2
                else:
                    j+=1

        return res
        
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################






    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        res = []
        i = 0
        j = 0
        
        # Calculamos el tamaño del salto para cada lista
        salto_p1 = int(math.sqrt(len(p1))) if len(p1) > 0 else 1
        salto_p2 = int(math.sqrt(len(p2))) if len(p2) > 0 else 1

        while i < len(p1) and j < len(p2):
            
            # CASO 1: Son iguales (gato y perro coinciden, así que lo descartamos)
            if p1[i] == p2[j]:
                # Avanzamos el puntero i
                i+=1
                # Avanzamos el puntero j
                j+=1

            elif p1[i] < p2[j]:
                # Guardamos el número en 'res' 
                res.append(p1[i])

                # Avanzamos el puntero i
                if i + salto_p1 < len(p1) and p1[i + salto_p1] <= p2[j]:
                    i+=salto_p1
                else:
                    i+=1

            else:
                # Avanzamos solo el puntero j
                if j + salto_p2 < len(p2) and p2[j + salto_p2] <= p1[i]:
                    j+=salto_p2
                else:
                    j+=1

        if i < len(p1):
            res.extend(p1[i:])

        return res

        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """

        ################
        ## COMPLETAR  ##
        ################
        
        # 1. Ejecutamos la consulta
        res, _ = self.solve_query(query)
        
        print(f"{'='*50}")
        print(f"Query: '{query}'")
        print(f"Number of results: {len(res)}")
        
        # 2. Comprobamos cuántos artículos debemos mostrar
        if self.show_all:
            limite = len(res)
        else:
            limite = min(len(res), self.SHOW_MAX)
            
        # 3. Imprimimos los resultados con el formato requerido
        for i in range(limite):
            artid = res[i]
            info = self.articles[artid]
            # Formato: Número de orden | ID | Título | URL
            print(f"#{i+1} \t({artid}) \t{info['title']} \t{info['url']}")
            
        # Devuelve el número de resultados (necesario para el modo -T de evaluación)
        return len(res)
