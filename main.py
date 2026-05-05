from typing import Optional
import shap
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import pickle

from contextlib import asynccontextmanager
from io import BytesIO
import matplotlib
matplotlib.use("Agg")

pipeline = None
modelo = None
scaler = None
features = None
explainer = None
imagem = None
metrics = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, modelo, scaler, explainer, metrics

    with open("pipeline_treinado.pkl", "rb") as f:
        package = pickle.load(f)

    pipeline = package["model"]
    metrics = package["metrics"]

    modelo = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    
    explainer = shap.TreeExplainer(modelo)

    yield

app = FastAPI(lifespan=lifespan)

colunas = [
    "Idade",
    "Peso (Kg)",
    "Altura (cm)",
    "IMC",
    "Grupo Sanguineo",
    "Frequencia Cardiaca (bpm)",
    "Frequencia Respiratoria (breaths/min)",
    "Hemoglobina (g/dl)",
    "Ciclo (Regular /Irregular)",
    "Duracao Ciclo (days)",
    "Duracao Casamento (Anos)",
    "Gravida (S/N)",
    "Numero de abortos",
    "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "FSH/LH",
    "Medida Quadril (polegadas)",
    "Medida Cintura (polegadas)",
    "Relacao Cintura Quadril",
    "TSH (mIU/L)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "Ganho Peso (S/N)",
    "Crescimento Capilar (s/n)",
    "Escurecimento da pele (S/N)",
    "Perda de Cabelo (S/N)",
    "Espinhas (S/N)",
    "Fast food (S/N)",
    "Exercita Regularmente (S/N)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Numero Foliculos (Ovario Esquerdo)",
    "Numero Foliculos (Ovario Direito)",
    "Media Tamanho Foliculo (Ovario Esquerdo) (mm)",
    "Media Tamanho Foliculo (Ovario Direito) (mm)",
    "Espessura do endometrio (mm)"
]

@app.get("/")
def model_info():
    global modelo
    
    if modelo is None:
        return {"error": "Modelo não carregado"}
    
    importances = modelo.feature_importances_

    feature_imp = pd.DataFrame({'Feature': colunas, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    
    return {
        "model_name": type(modelo).__name__,
        "task": "classification",
        "features": colunas,
        "n_features": len(colunas),
        "top_features": feature_imp.head(10).to_dict(orient='records'),
        "metrics": metrics
    }

@app.post("/predict/")
def run_predict(
    Numero_Foliculos_OD: Optional[int] = Form(6),   # média aproximada
    Numero_Foliculos_OE: Optional[int] = Form(6),
    Escurecimento_Pele: Optional[int] = Form(1),  # 1 (Sim) ou 0 (Não)
    Ganho_Peso: Optional[int] = Form(1),  # 1 (Sim) ou 0 (Não)
    Crescimento_Capilar: Optional[int] = Form(0),  # 1 (Sim) ou 0 (Não)
    AMH_ng_mL: Optional[float] = Form(5.62),
    Ciclo_Regular: Optional[int] = Form(2),# 2 4 ou 5
    FSH_LH: Optional[float] = Form(6.9),
    LH_mIU_mL: Optional[float] = Form(64.7),
    Duracao_Ciclo_days: Optional[int] = Form(5),
    Idade: Optional[int] = Form(31),
    FSH_mIU_mL: Optional[float] = Form(14.6),
    IMC: Optional[float] = Form(24.32),
    Duracao_Casamento_Anos: Optional[int] = Form(8),
    Peso_Kg: Optional[float] = Form(59.64),
    VitD3_ng_mL: Optional[float] = Form(49.92),
    Espessura_Endometrio_mm: Optional[float] = Form(8),
    TSH_mIU_L: Optional[float] = Form(2.98),
    Medida_Quadril_polegadas: Optional[float] = Form(37.99),
    Media_Tamanho_Foliculo_OD_mm: Optional[float] = Form(15), 
    PRG_ng_mL: Optional[float] = Form(0.61),
    Relacao_Cintura_Quadril: Optional[float] = Form(0.89),
    PRL_ng_mL: Optional[float] = Form(24.32),
    FastFood: Optional[int] = Form(0),  # 1 (Sim) ou 0 (Não)
    Media_Tamanho_Foliculo_OE_mm: Optional[float] = Form(15), 
    Medida_Cintura_polegadas: Optional[float] = Form(33.84),
    Altura_cm: Optional[int] = Form(156),
    RBS_mg_dl: Optional[float] = Form(99.84),
    Espinhas: Optional[int] = Form(1),  # 1 (Sim) ou 0 (Não)
    Hemoglobina_g_dl: Optional[float] = Form(11.16),
    I_beta_HCG_mIU_mL: Optional[float] = Form(664.55),
    Frequencia_Cardiaca_bpm: Optional[int] = Form(73),
    II_beta_HCG_mIU_mL: Optional[float] = Form(238.23),
    Frequencia_Respiratoria_bpm: Optional[int] = Form(19),
    Grupo_Sanguineo: Optional[int] = Form(14),    # (A+ = 11 / A- = 12 / B+ = 13 / B- = 14 / O+ =15 / O- = 16 / AB+ =17 / AB- = 18)
    BP_Systolic_mmHg: Optional[int] = Form(115),
    Exercita_Regularmente: Optional[int] = Form(0),  # 1 (Sim) ou 0 (Não)
    Perda_Cabelo: Optional[int] = Form(0), # 1 (Sim) ou 0 (Não)
    Numero_Abortos: Optional[int] = Form(0),
    BP_Diastolic_mmHg: Optional[int] = Form(77),
    Gravida: Optional[int] = Form(0) # 1 (Sim) ou 0 (Não)
    ): 
    
    try:
        input_data = [
            Idade, Peso_Kg, Altura_cm, IMC, Grupo_Sanguineo,
            Frequencia_Cardiaca_bpm, Frequencia_Respiratoria_bpm, Hemoglobina_g_dl,
            Ciclo_Regular, Duracao_Ciclo_days, Duracao_Casamento_Anos, Gravida,
            Numero_Abortos, I_beta_HCG_mIU_mL, II_beta_HCG_mIU_mL, FSH_mIU_mL,
            LH_mIU_mL, FSH_LH, Medida_Quadril_polegadas, Medida_Cintura_polegadas,
            Relacao_Cintura_Quadril, TSH_mIU_L, AMH_ng_mL, PRL_ng_mL, VitD3_ng_mL,
            PRG_ng_mL, RBS_mg_dl, Ganho_Peso, Crescimento_Capilar, Escurecimento_Pele,
            Perda_Cabelo, Espinhas, FastFood, Exercita_Regularmente,
            BP_Systolic_mmHg, BP_Diastolic_mmHg, Numero_Foliculos_OE,
            Numero_Foliculos_OD, Media_Tamanho_Foliculo_OE_mm,
            Media_Tamanho_Foliculo_OD_mm, Espessura_Endometrio_mm
        ]
        global imagem
        df_input = pd.DataFrame([input_data], columns=colunas)
        
        df_scaled = scaler.transform(df_input)
        
        prediction = modelo.predict(df_scaled)
        shap_values = explainer(df_scaled)
        
        nomes_abreviados = [
            "Idade", "Peso", "Altura", "IMC", "Grupo",
            "FC", "FR", "Hb",
            "Ciclo", "Dur Ciclo", "Dur Casam", "Grav",
            "Abortos", "HCG I", "HCG II", "FSH",
            "LH", "FSH/LH", "Quadril", "Cintura",
            "RCQ", "TSH", "AMH", "PRL", "VitD",
            "PRG", "RBS", "Ganho Peso", "Capilar",
            "Pele", "Queda", "Espinhas", "FastFood",
            "Exercício", "PA Sist", "PA Diast",
            "Qtd Fol OE", "Qtd Fol OD", "Tam OE",
            "Tam OD", "Endométrio"
        ]
        
        shap.plots.force(
            shap_values.base_values[0][1],
            shap_values.values[0][:, 1],
            feature_names=nomes_abreviados,
            matplotlib=True
        )

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        plt.close()

        buf.seek(0)
        imagem = buf.getvalue()

        return {
            "prediction": int(prediction[0]),
            "input": df_input.to_dict(orient='records')[0],
            "explicacao_visual": "http://127.0.0.1:8000/shap_plot"
        }

    except Exception as e:
        return {"error": str(e)}
    
from fastapi import Response

@app.get(
    "/shap_plot",
    responses={200: {"content": {"image/png": {}}}}
)
def shap_plot():
    if imagem is None:
        return {"error": "Nenhuma imagem gerada ainda"}

    return Response(content=imagem, media_type="image/png")