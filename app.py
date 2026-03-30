from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import json
import re
import os

# Carrega as variáveis do arquivo .env (esconde a chave da API)
load_dotenv()

app = FastAPI(title="IA Híbrida - UFRJ Consulting Club")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na vida real colocamos "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. CONFIGURAÇÃO DA IA GENERATIVA (O Juiz Semântico)
# A nova biblioteca pega automaticamente a variável GEMINI_API_KEY do ambiente
try:
    client = genai.Client()
    print("✅ Google GenAI configurado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao configurar Google GenAI: {e}. Verifique sua GEMINI_API_KEY.")

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
        # Nova sintaxe da biblioteca google-genai
        resposta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
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
            return {"views_estimadas": views_base, "feedback": "Análise matemática (Erro no formato JSON da IA).", "sugestoes": []}
            
    except Exception as e:
        print(f"Erro na IA Generativa: {e}")
        return {"views_estimadas": views_base, "feedback": "Análise matemática (IA Generativa indisponível).", "sugestoes": []}

if __name__ == "__main__":
    import uvicorn
    # O Render define a variável PORT automaticamente
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)