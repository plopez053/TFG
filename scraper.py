import requests
from bs4 import BeautifulSoup
import os
import re
from concurrent.futures import ThreadPoolExecutor

# Ruta de salida para los archivos descargados
OUTPUT_DIR = "actas"
# Patrón para limpiar caracteres de los nombres de archivos
RE_LIMPIAR_NOMBRE = re.compile(r'[\\/*?:"<>|]')

def descargar_pdf(session, url_post, payload_base, opcion):
    anio_texto = opcion.text.strip()
    anio_valor = opcion.get('value')
    
    if not anio_valor or not anio_texto.isdigit():
        return
        
    print(f"\nProcesando año {anio_texto}...")
    
    # Preparamos el payload para este año
    payload_anio = payload_base.copy()
    payload_anio['anioId'] = anio_valor
    
    try:
        resp = session.post(url_post, data=payload_anio, timeout=30)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Buscamos la tabla con la clase 'tablalistados'
        tabla = soup.find('table', class_='tablalistados')
        if not tabla:
            print(f"No se encontró tabla para {anio_texto}")
            return

        filas = tabla.find_all('tr')
        print(f"Procesando {len(filas)} filas en {anio_texto}...")
        descargas_count = 0

        for idx, fila in enumerate(filas):
            tds = fila.find_all('td')
            if len(tds) < 6: 
                continue

            # Extraemos metadatos para el nombre del archivo
            fecha = tds[0].get_text(strip=True).replace("/", "-")
            sesion_raw = tds[2].get_text(strip=True).replace("\n", " ").strip()
            sesion_limpia = RE_LIMPIAR_NOMBRE.sub("", sesion_raw)

            # Revisamos cada columna de interés
            for col_idx, doc_tipo in [(5, "Acta"),(3,"Orden"),(4,"Extracto")]:
                link = tds[col_idx].find('a', href=True)
                
                if link and 'pdf' in link['href'].lower():
                    # Aseguramos que existe la carpeta para el año
                    folder_path = os.path.join(OUTPUT_DIR, anio_texto)
                    os.makedirs(folder_path, exist_ok=True)
                    
                    nombre_archivo = f"{fecha}_{sesion_limpia}_{doc_tipo}.pdf"
                    ruta_guardado = os.path.join(folder_path, nombre_archivo)

                    if not os.path.exists(ruta_guardado):
                        href = link['href']
                        url_pdf = f"https://www.bilbao.eus{href}" if href.startswith("/") else href
                        
                        print(f"Detectado {doc_tipo} ({fecha}): {nombre_archivo}")
                        try:
                            # Descarga del contenido binario (PDF)
                            resp_pdf = session.get(url_pdf, timeout=60)
                            if resp_pdf.status_code == 200:
                                with open(ruta_guardado, "wb") as f:
                                    f.write(resp_pdf.content)
                                print(f"Guardado")
                                descargas_count += 1
                            else:
                                print(f"Error HTTP {resp_pdf.status_code}")
                        except Exception as e:
                            print(f"Error en descarga: {e}")

        if descargas_count > 0:
            print(f"Finalizado {anio_texto}: {descargas_count} archivos nuevos.")
        
    except Exception as e:
        print(f"Error procesando año {anio_texto}: {e}")

def obtener_html():
    url = "https://www.bilbao.eus/cs/Satellite?c=Page&cid=3000015482&language=es&pageid=3000015482&pagename=Bilbaonet%2FPage%2FBIO_ListadoSesionesPlenarias"
    session = requests.Session()
    
    print("Obteniendo página principal...")
    resp_inicial = session.get(url)
    soup = BeautifulSoup(resp_inicial.text, 'html.parser')
    
    form = None
    for f in soup.find_all('form'):
        if f.find(attrs={'name': 'anioId'}) or f.find(attrs={'id': 'anioId'}):
            form = f
            break
            
    if not form:
        print("No se encontró el formulario.")
        return

    payload = {}
    for input_tag in form.find_all('input'):
        name = input_tag.get('name')
        value = input_tag.get('value', '')
        if name:
            payload[name] = value

    opciones = form.find_all('option')
    print(f"Encontrados {len(opciones)} años.")

    action = form.get('action')
    url_post = "https://www.bilbao.eus" + action if action and not action.startswith("http") else url

    with ThreadPoolExecutor(max_workers=5) as executor:
        for opcion in opciones:
            executor.submit(descargar_pdf, session, url_post, payload, opcion)

if __name__ == "__main__":
    obtener_html()
    print("Proceso finalizado.")