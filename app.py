from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import joblib
import pandas as pd
import numpy as np
import json
import re

app = FastAPI(title="IA Híbrida - UFRJ Consulting Club")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na vida real colocamos "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. CONFIGURAÇÃO DA IA GENERATIVA (O Juiz Semântico)
genai.configure(api_key="AIzaSyAM-LwazhRTEXJNMOx__9JJI3nIvmKIQsw")

modelo_ideal = None
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        if 'flash' in m.name:
            modelo_ideal = m.name
            break
        modelo_ideal = m.name
modelo_gemini = genai.GenerativeModel(modelo_ideal)

# 2. CONFIGURAÇÃO DO MACHINE LEARNING (A Realidade do Clube)
try:
    modelo_xgb = joblib.load('modelo_cp_views.pkl')
    print("✅ XGBoost carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar XGBoost: {e}")

MAPA_CATEGORIAS = {
    "Business": 0, "Consultoria": 1, "Desenvolvimento Pessoal": 2, 
    "Economia / Finanças": 3, "Finanças": 4, "Geopolítica / Sociedade": 5, 
    "Negócios": 6, "Sem Categoria": 7, "Tecnologia / Informação": 8
}
KEYWORDS_IMPACTO = ['ia', 'inteligência artificial', 'eua', 'china', 'trump', 'recessão', 'inflação', 'guerra', 'crise', 'futuro']

class SugestaoPost(BaseModel):
    titulo: str
    categoria: str
    minutos_leitura: int = 8

@app.post("/prever")
def prever_views(dados: SugestaoPost):
    # PASSO 1: A Realidade (Cálculo do XGBoost)
    t = dados.titulo.lower()
    features = pd.DataFrame([{
        'minutosLeitura': dados.minutos_leitura,
        'idade_dias': 30,
        'titulo_n_palavras': len(dados.titulo.split()),
        'titulo_n_chars': len(dados.titulo),
        'titulo_tem_numero': 1 if re.search(r'\d', str(dados.titulo)) else 0,
        'titulo_tem_pergunta': 1 if '?' in str(dados.titulo) else 0,
        'n_keywords_impacto': sum(1 for k in KEYWORDS_IMPACTO if k in t),
        'categoria_enc': MAPA_CATEGORIAS.get(dados.categoria, 7)
    }])
    
    pred_log = modelo_xgb.predict(features)[0]
    views_base = int(np.expm1(pred_log))

    # PASSO 2: O Editor-Chefe (Avaliando a qualidade)
    prompt = f"""
    Você é o Editor-Chefe Sênior do blog do UFRJ Consulting Club.
    
    Analise o seguinte título de postagem: "{dados.titulo}" (Categoria: "{dados.categoria}")
    
    Sua tarefa:
    1. Multiplicador: Avalie a qualidade do título. 
       - 0.1 : Títulos sem sentido, palavras soltas ou vulgares.
       - 0.8 : Títulos muito genéricos ou chatos.
       - 1.0 : Títulos normais e informativos.
       - 1.2 a 1.5 : Títulos excelentes, intrigantes e persuasivos.
    2. Feedback: Uma justificativa curta.
    3. Sugestões: Crie 2 opções de títulos MUITO melhores e mais persuasivos (estilo copywriting).
    
    Responda APENAS com um JSON válido neste exato formato:
    {{
        "multiplicador": 1.2,
        "feedback": "O título é bom, mas falta um gatilho de curiosidade.",
        "sugestoes": ["A Arte da Estratégia Corporativa", "Como a IA está mudando o mercado"]
    }}
    """
    
    try:
        resposta = modelo_gemini.generate_content(prompt)
        match = re.search(r'\{.*\}', resposta.text, re.DOTALL)
        
        if match:
            resultado_llm = json.loads(match.group(0))
            
            # PASSO 3: A FUSÃO MATEMÁTICA (Onde a mágica acontece com segurança)
            multiplicador = resultado_llm.get('multiplicador', 1.0)
            views_finais = int(views_base * multiplicador)
            
            # Devolvemos para o site as views já calculadas + os textos da IA
            return {
                "views_estimadas": views_finais,
                "feedback": resultado_llm.get('feedback', ''),
                "sugestoes": resultado_llm.get('sugestoes', [])
            }
        else:
            return {"views_estimadas": views_base, "feedback": "Análise matemática.", "sugestoes": []}
            
    except Exception as e:
        print(f"Erro: {e}")
        return {"views_estimadas": views_base, "feedback": "Análise matemática.", "sugestoes": []}