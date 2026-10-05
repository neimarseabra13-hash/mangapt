from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote
import ipaddress, socket, io, os, re, asyncio, base64
from pathlib import Path
import httpx
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import pytesseract

VERSION = '0.4.0'
app = FastAPI(title='MangaPT Server', version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
ROOT = Path(__file__).resolve().parent
UA = 'MangaPT/0.4 (public-content importer)'
TIMEOUT = 30.0
MAX_HTML = 5 * 1024 * 1024
MAX_IMAGE = 20 * 1024 * 1024
MAX_UPLOAD = 20 * 1024 * 1024
MAX_BATCH = 50
TRANSLATOR_URL = os.getenv('TRANSLATOR_URL', '').strip()
TRANSLATOR_KEY = os.getenv('TRANSLATOR_KEY', '').strip()
TRANSLATOR_MODE = os.getenv('TRANSLATOR_MODE', 'generic').strip().lower()


def validate_public_url(url: str):
    p = urlparse(url)
    if p.scheme not in ('http', 'https') or not p.hostname:
        raise HTTPException(400, 'A URL precisa usar http:// ou https://.')
    try:
        infos = socket.getaddrinfo(p.hostname, None)
        ips = {ipaddress.ip_address(x[4][0]) for x in infos}
        if any(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast for ip in ips):
            raise HTTPException(403, 'Por segurança, endereços locais ou privados não são aceitos.')
    except socket.gaierror:
        raise HTTPException(400, 'Não foi possível resolver o domínio.')


async def http_get(url: str, max_bytes: int, accept: str = '*/*'):
    validate_public_url(url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT, headers={'User-Agent': UA, 'Accept': accept}) as client:
        try:
            r = await client.get(url)
        except httpx.HTTPError as e:
            raise HTTPException(502, f'Não foi possível acessar o recurso: {e}')
    if r.status_code in (401, 403, 429):
        raise HTTPException(424, 'O servidor de origem exige acesso autorizado ou bloqueou a requisição. O MangaPT não tenta contornar essa proteção.')
    if r.status_code >= 400:
        raise HTTPException(502, f'O servidor respondeu HTTP {r.status_code}.')
    if len(r.content) > max_bytes:
        raise HTTPException(413, 'O recurso excede o limite permitido.')
    return r


def extract_images(html: str, base_url: str):
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    attrs = ('src', 'data-src', 'data-lazy-src', 'data-original', 'data-image', 'data-url')
    for img in soup.find_all('img'):
        for attr in attrs:
            value = img.get(attr)
            if value:
                u = urljoin(base_url, value)
                if u not in found and urlparse(u).scheme in ('http', 'https'):
                    found.append(u)
                break
    for tag in soup.find_all('source'):
        for attr in ('src', 'srcset', 'data-srcset'):
            value = tag.get(attr)
            if value:
                for part in value.split(','):
                    u = urljoin(base_url, part.strip().split(' ')[0])
                    if u not in found and urlparse(u).scheme in ('http', 'https'):
                        found.append(u)
    for tag in soup.find_all('script'):
        txt = tag.string or tag.get_text(' ', strip=True)
        for match in re.findall(r'https?://[^"\'\s<>]+\.(?:jpg|jpeg|png|webp)(?:\?[^"\'\s<>]*)?', txt, re.I):
            if match not in found:
                found.append(match)
    return found


async def fetch_image(url: str):
    r = await http_get(url, MAX_IMAGE, 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8')
    media = r.headers.get('content-type', 'application/octet-stream').split(';')[0]
    if not media.startswith('image/'):
        raise HTTPException(415, 'O recurso encontrado não é uma imagem.')
    return r.content, media


@app.get('/')
async def index():
    return FileResponse(ROOT / 'mangapt.html', media_type='text/html')


@app.get('/api/health')
async def health():
    return {
        'ok': True, 'service': 'mangapt', 'version': VERSION,
        'ocr': True, 'translator': bool(TRANSLATOR_URL),
        'translator_mode': TRANSLATOR_MODE,
        'batch': True, 'render': True
    }


@app.get('/api/import')
async def import_chapter(url: str = Query(..., min_length=10)):
    r = await http_get(url, MAX_HTML, 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.5')
    images = extract_images(r.text, str(r.url))
    return {
        'source': str(r.url),
        'pages': [{'url': u, 'proxy': '/api/image?url=' + quote(u, safe='')} for u in images[:100]],
        'count': min(len(images), 100)
    }


@app.get('/api/image')
async def image_proxy(url: str = Query(..., min_length=10)):
    content, media = await fetch_image(url)
    return StreamingResponse(io.BytesIO(content), media_type=media, headers={'Cache-Control': 'public, max-age=3600'})


def image_from_bytes(data: bytes):
    try:
        im = Image.open(io.BytesIO(data)).convert('RGB')
        if im.width * im.height > 25_000_000:
            im.thumbnail((5000, 5000))
        return im
    except Exception:
        raise HTTPException(415, 'Arquivo não é uma imagem válida.')


def preprocess_variants(im: Image.Image):
    gray = ImageOps.grayscale(im)
    scale = 1.5 if max(im.width, im.height) < 1800 else 1.0
    if scale != 1.0:
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.Resampling.LANCZOS)
    # Mantém uma versão normal e uma com contraste/limiar para melhorar texto pequeno.
    boosted = ImageOps.autocontrast(gray)
    return [boosted, boosted.point(lambda p: 255 if p > 190 else 0)]


def ocr_image(im: Image.Image, lang='eng'):
    variants = preprocess_variants(im)
    candidates = []
    for variant in variants:
        try:
            data = pytesseract.image_to_data(variant, lang=lang, config='--psm 11', output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError as e:
            raise HTTPException(500, f'OCR indisponível: {e}')
        scale = variant.width / im.width
        words=[]
        n=len(data.get('text', []))
        for i in range(n):
            txt=(data['text'][i] or '').strip(); raw=str(data['conf'][i]).strip()
            conf=float(raw) if raw not in ('','-1') else -1
            if not txt or conf < 25: continue
            x,y,w,h=[int(data[k][i]) for k in ('left','top','width','height')]
            words.append({'text':txt,'conf':conf,'x':x,'y':y,'w':w,'h':h,'cy':y+h/2})
        words.sort(key=lambda z:(z['cy'],z['x']))
        lines=[]
        for word in words:
            best=None; best_delta=99999
            for line in lines:
                avg_h=sum(x['h'] for x in line)/len(line); avg_cy=sum(x['cy'] for x in line)/len(line)
                delta=abs(word['cy']-avg_cy)
                if delta <= max(12, avg_h*0.7) and delta < best_delta:
                    best=line; best_delta=delta
            if best is None: lines.append([word])
            else: best.append(word)
        for line in lines:
            line.sort(key=lambda x:x['x'])
            text=' '.join(x['text'] for x in line).strip()
            if not text: continue
            x0=min(x['x'] for x in line); y0=min(x['y'] for x in line)
            x1=max(x['x']+x['w'] for x in line); y1=max(x['y']+x['h'] for x in line)
            candidates.append({
                'text':text,
                'confidence':round(sum(x['conf'] for x in line)/len(line)/100,3),
                'box':{'x':int(x0/scale),'y':int(y0/scale),'w':int((x1-x0)/scale),'h':int((y1-y0)/scale)}
            })
    candidates.sort(key=lambda b:(b['box']['y'],b['box']['x']))
    out=[]
    for c in candidates:
        duplicate=False
        for old in out:
            a,b=old['box'],c['box']
            ix=max(0,min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x']))
            iy=max(0,min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']))
            inter=ix*iy; area=min(a['w']*a['h'],b['w']*b['h']) or 1
            if inter/area>.65:
                duplicate=True
                if c['confidence']>old['confidence']: old.update(c)
                break
        if not duplicate: out.append(c)
    return out


class TranslateRequest(BaseModel):
    text: str
    source: str = 'en'
    target: str = 'pt-BR'

class BatchTranslateRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    source: str = 'en'
    target: str = 'pt-BR'

class ProcessRequest(BaseModel):
    imageUrl: str
    source: str = 'en'
    target: str = 'pt-BR'
    render: bool = True


@app.post('/api/ocr')
async def ocr(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, 'Arquivo excede o limite permitido.')
    im = image_from_bytes(data)
    blocks = await asyncio.to_thread(ocr_image, im)
    return {'width': im.width, 'height': im.height, 'blocks': blocks, 'text': '\n'.join(x['text'] for x in blocks)}


async def external_translate(text: str, source: str, target: str):
    if not TRANSLATOR_URL:
        return None
    headers = {'Content-Type': 'application/json'}
    if TRANSLATOR_KEY:
        headers['Authorization'] = f'Bearer {TRANSLATOR_KEY}'
    if TRANSLATOR_MODE in ('openai', 'openai-compatible'):
        payload = {
            'model': os.getenv('TRANSLATOR_MODEL', 'gpt-4.1-mini'),
            'messages': [
                {'role':'system','content':f'Translate manga dialogue from {source} to {target}. Preserve tone, names, honorifics and line meaning. Return only the translation.'},
                {'role':'user','content':text}
            ], 'temperature': 0.2
        }
    else:
        payload = {'q': text, 'source': source, 'target': target, 'format': 'text'}
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            r = await client.post(TRANSLATOR_URL, json=payload, headers=headers)
        except httpx.HTTPError as e:
            raise HTTPException(502, f'Falha ao chamar o tradutor: {e}')
    if r.status_code >= 400:
        raise HTTPException(502, f'O serviço de tradução respondeu HTTP {r.status_code}.')
    try: obj = r.json()
    except Exception: raise HTTPException(502, 'O tradutor retornou uma resposta inválida.')
    if TRANSLATOR_MODE in ('openai', 'openai-compatible'):
        result = (((obj.get('choices') or [{}])[0]).get('message') or {}).get('content')
    else:
        result = obj.get('translatedText') or obj.get('translation') or obj.get('text')
    return (result or '').strip()


@app.post('/api/translate')
async def translate(req: TranslateRequest):
    if not req.text.strip(): return {'translatedText': '', 'configured': bool(TRANSLATOR_URL)}
    result = await external_translate(req.text, req.source, req.target)
    if result is None:
        return {'translatedText': '', 'configured': False, 'message': 'Configure TRANSLATOR_URL para ativar a tradução automática.'}
    return {'translatedText': result, 'configured': True}


@app.post('/api/translate/batch')
async def translate_batch(req: BatchTranslateRequest):
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(413, f'Máximo de {MAX_BATCH} blocos por lote.')
    if not TRANSLATOR_URL:
        return {'configured': False, 'translations': ['' for _ in req.texts], 'message': 'Configure TRANSLATOR_URL para ativar a tradução automática.'}
    async def one(t): return await external_translate(t, req.source, req.target) if t.strip() else ''
    translations = await asyncio.gather(*(one(t) for t in req.texts))
    return {'configured': True, 'translations': translations}


def fit_font(draw, text, max_w, max_h):
    # Fonte comum do container; cai para a fonte padrão se não existir.
    candidates = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    size = max(10, min(42, int(max_h * .42)))
    while size >= 10:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        if draw.textbbox((0,0), 'Ag', font=font)[3] <= max_h and draw.textbbox((0,0), text[:25], font=font)[2] <= max_w * 1.15:
            return font
        size -= 2
    return ImageFont.load_default()


def draw_translations(im: Image.Image, blocks, translations):
    out = im.copy().convert('RGB'); draw = ImageDraw.Draw(out)
    for block, translated in zip(blocks, translations):
        translated = (translated or '').strip()
        if not translated: continue
        b=block['box']; x,y,w,h=b['x'],b['y'],b['w'],b['h']
        pad=max(8, int(min(w,h)*.45))
        x0=max(0,x-pad); y0=max(0,y-pad); x1=min(out.width,x+w+pad); y1=min(out.height,y+h+pad)
        draw.rounded_rectangle((x0,y0,x1,y1), radius=max(6,min(30,pad)), fill='white')
        max_w=max(60,x1-x0-12); max_h=max(20,y1-y0-8)
        font=fit_font(draw, translated, max_w, max_h)
        words=translated.split(); lines=[]; line=''
        for word in words:
            test=(line+' '+word).strip()
            if draw.textbbox((0,0),test,font=font)[2] <= max_w or not line: line=test
            else: lines.append(line); line=word
        if line: lines.append(line)
        bbox=draw.textbbox((0,0),'Ag',font=font); line_h=max(14,bbox[3]-bbox[1]+4)
        total_h=line_h*len(lines)
        ty=y0+max(3,(y1-y0-total_h)//2)
        for ln in lines:
            tw=draw.textbbox((0,0),ln,font=font)[2]
            tx=x0+max(4,(x1-x0-tw)//2)
            draw.text((tx,ty),ln,fill='black',font=font)
            ty+=line_h
    buf=io.BytesIO(); out.save(buf,format='JPEG',quality=92); return buf.getvalue()


class RenderRequest(BaseModel):
    imageUrl: str
    blocks: list
    translations: list

@app.post('/api/render')
async def render(req: RenderRequest):
    data,_=await fetch_image(req.imageUrl); im=image_from_bytes(data)
    if len(req.blocks)!=len(req.translations): raise HTTPException(400,'Quantidade de blocos e traduções não coincide.')
    out=await asyncio.to_thread(draw_translations,im,req.blocks,req.translations)
    return StreamingResponse(io.BytesIO(out),media_type='image/jpeg')


def encode_jpeg(data: bytes): return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')

@app.post('/api/process-url')
async def process_url(req: ProcessRequest):
    data,_=await fetch_image(req.imageUrl); im=image_from_bytes(data)
    ocr_lang = 'eng' if req.source.lower() in ('en','en-us','eng') else req.source
    blocks=await asyncio.to_thread(ocr_image,im,ocr_lang)
    if not TRANSLATOR_URL:
        return {'configured':False,'blocks':blocks,'translations':['' for _ in blocks],'image':None,'message':'OCR concluído. Configure TRANSLATOR_URL para tradução automática.'}
    trans_source = 'en' if req.source.lower() in ('en','en-us','eng') else req.source
    translations=await asyncio.gather(*(external_translate(b['text'],trans_source,req.target) for b in blocks))
    rendered=await asyncio.to_thread(draw_translations,im,blocks,translations) if req.render else None
    return {'configured':True,'blocks':blocks,'translations':translations,'image':encode_jpeg(rendered) if rendered else None}

@app.post('/api/process-upload')
async def process_upload(file: UploadFile=File(...), source: str='eng', target: str='pt-BR', render: bool=True):
    data=await file.read()
    if len(data)>MAX_UPLOAD: raise HTTPException(413,'Arquivo excede o limite permitido.')
    im=image_from_bytes(data); ocr_lang='eng' if source.lower() in ('en','en-us','eng') else source; blocks=await asyncio.to_thread(ocr_image,im,ocr_lang)
    if not TRANSLATOR_URL:
        return {'configured':False,'blocks':blocks,'translations':['' for _ in blocks],'image':None,'message':'OCR concluído. Configure TRANSLATOR_URL para tradução automática.'}
    trans_source = 'en' if source.lower() in ('en','en-us','eng') else source
    translations=await asyncio.gather(*(external_translate(b['text'],trans_source,target) for b in blocks))
    rendered=await asyncio.to_thread(draw_translations,im,blocks,translations) if render else None
    return {'configured':True,'blocks':blocks,'translations':translations,'image':encode_jpeg(rendered) if rendered else None}
