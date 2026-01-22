import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

def descargar_pdf(session, url_post, payload_base, opcion):
    anio_texto = opcion.text.strip()
    anio_valor = opcion.get('value')
    
    if not anio_valor:
        return
        
    print(f"--- Buscando PDFs en el año: {anio_texto} ---")
    
    payload_anio = payload_base.copy()
    payload_anio['anioId'] = anio_valor
    
    try:
        resp_lista = session.post(url_post, data=payload_anio)
        soup_lista = BeautifulSoup(resp_lista.text, 'html.parser')
        
        enlaces = soup_lista.find_all('a', href=True)
        pdf_urls = []
        for a in enlaces:
            if 'blobheader=application%2Fpdf' in a['href']:
                url_pdf = a['href']
                if not url_pdf.startswith('http'):
                    url_pdf = "https://www.bilbao.eus" + url_pdf
                pdf_urls.append(url_pdf)

        print(f"   -> He encontrado {len(pdf_urls)} documentos para {anio_texto}")

        for i, url_pdf in enumerate(pdf_urls):
            import urllib.parse
            try:
                nombre_real = urllib.parse.unquote(url_pdf.split('filename%3D')[1].split('&')[0])
            except:
                nombre_real = f"archivo_{anio_texto}_{i}.pdf"

            print(f"   -> Descargando: {nombre_real}...")
            
            # Pedimos el archivo. IMPORTANTE: usamos .content (datos binarios) no .text
            resp_pdf = session.get(url_pdf)
            
            # Guardamos con 'wb' (Write Binary) porque un PDF no es texto plano
            with open(nombre_real, "wb") as f:
                f.write(resp_pdf.content)
        
        print(f"✅ Finalizado año {anio_texto}")
        
    except Exception as e:
        print(f"Error procesando {anio_texto}: {e}")
        

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