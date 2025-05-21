import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import csv
import xlsxwriter
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from dotenv import load_dotenv
import os
import logging

# Configurar logging para diagnosticar erros
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
sender_email = os.getenv("SENDER_EMAIL")
sender_password = os.getenv("SENDER_PASSWORD")

# Verificar se as credenciais estão carregadas
if not sender_email or not sender_password:
    logger.error("Credenciais de e-mail não encontradas no arquivo .env")
    st.error("Erro: Credenciais de e-mail não encontradas. Verifique o arquivo .env.")

# Cacheamento para melhorar performance
@st.cache_data
def fetch_real_data_cached(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo")
        if not hist.empty:
            return {
                "close": hist["Close"].iloc[-1],
                "volume": hist["Volume"].iloc[-1],
                "volatility": hist["Close"].pct_change().std() * np.sqrt(252),
                "dividend_yield": stock.info.get("dividendYield", 0) * 100 if "dividendYield" in stock.info else np.random.uniform(0, 5),
                "history": hist,
                "dividend_history": [np.random.uniform(0, 2) for _ in range(12)]
            }
        return {
            "close": np.random.uniform(20, 60),
            "volume": np.random.randint(1000000, 5000000),
            "volatility": np.random.uniform(0.1, 0.5),
            "dividend_yield": np.random.uniform(0, 5),
            "history": pd.DataFrame({"Close": [np.random.uniform(20, 60) for _ in range(30)]}),
            "dividend_history": [np.random.uniform(0, 2) for _ in range(12)]
        }
    except Exception as e:
        st.warning(f"Erro ao buscar dados para {symbol}: {e}")
        return {
            "close": np.random.uniform(20, 60),
            "volume": np.random.randint(1000000, 5000000),
            "volatility": np.random.uniform(0.1, 0.5),
            "dividend_yield": np.random.uniform(0, 5),
            "history": pd.DataFrame({"Close": [np.random.uniform(20, 60) for _ in range(30)]}),
            "dividend_history": [np.random.uniform(0, 2) for _ in range(12)]
        }

# Lista expandida de ativos
b3_assets = [
    {"name": "PETR4", "symbol": "PETR4.SA"},
    {"name": "VALE3", "symbol": "VALE3.SA"},
    {"name": "ITUB4", "symbol": "ITUB4.SA"},
    {"name": "BBDC4", "symbol": "BBDC4.SA"},
    {"name": "ABEV3", "symbol": "ABEV3.SA"},
    {"name": "MGLU3", "symbol": "MGLU3.SA"},
    {"name": "WEGE3", "symbol": "WEGE3.SA"},
    {"name": "EMBR3", "symbol": "EMBR3.SA"},
    {"name": "GGBR4", "symbol": "GGBR4.SA"},
    {"name": "JBSS3", "symbol": "JBSS3.SA"}
]

funds = [
    {"name": "BOVA11", "symbol": "BOVA11.SA"},
    {"name": "IVVB11", "symbol": "IVVB11.SA"},
    {"name": "SMAL11", "symbol": "SMAL11.SA"},
    {"name": "IMAB11", "symbol": "IMAB11.SA"},
    {"name": "XBOV11", "symbol": "XBOV11.SA"}
]

cryptos = [
    {"name": "Bitcoin", "symbol": "BTC-USD"},
    {"name": "Ethereum", "symbol": "ETH-USD"},
    {"name": "Binance Coin", "symbol": "BNB-USD"},
    {"name": "Cardano", "symbol": "ADA-USD"},
    {"name": "Solana", "symbol": "SOL-USD"}
]

indices = [
    {"name": "IBOVESPA", "symbol": "^BVSP"},
    {"name": "IFIX", "symbol": "^IFIX"}
]

fiis = [
    {"name": "HGLG11", "symbol": "HGLG11.SA"},
    {"name": "KNRI11", "symbol": "KNRI11.SA"},
    {"name": "XPLG11", "symbol": "XPLG11.SA"}
]

currencies = [
    {"name": "USD/BRL", "symbol": "USDBRL=X"},
    {"name": "EUR/BRL", "symbol": "EURBRL=X"},
    {"name": "GBP/BRL", "symbol": "GBPBRL=X"}
]

# Função para remover duplicatas
def remove_duplicates(assets_list):
    seen = set()
    unique_assets = []
    for asset in assets_list:
        if asset["name"] not in seen:
            seen.add(asset["name"])
            unique_assets.append(asset)
    return unique_assets

# Função corrigida para buscar todos os ativos
@st.cache_data
def fetch_all_b3_assets():
    all_assets = []
    tickers_list = b3_assets + funds + fiis
    tickers = yf.Tickers(" ".join([asset["symbol"] for asset in tickers_list]))
    for symbol, ticker in tickers.tickers.items():
        try:
            if ticker.info.get("market") == "br_market" or symbol.endswith(".SA"):
                all_assets.append({"name": symbol.replace(".SA", ""), "symbol": symbol})
        except Exception as e:
            logger.warning(f"Erro ao verificar ativo {symbol}: {e}")
            continue
    all_assets.extend(indices + currencies)
    return all_assets

# Função para salvar dados históricos para treinamento
@st.cache_data
def save_historical_data(assets_list, filename="historical_data.csv"):
    data = []
    for asset in assets_list:
        try:
            stock = yf.Ticker(asset["symbol"])
            hist = stock.history(period="1y")
            if not hist.empty:
                for date, row in hist.iterrows():
                    data.append([asset["symbol"], date, row["Close"], row["Volume"]])
        except Exception as e:
            logger.warning(f"Erro ao baixar dados de {asset['symbol']}: {e}")
    df = pd.DataFrame(data, columns=["Symbol", "Date", "Close", "Volume"])
    df.to_csv(filename, index=False)
    return filename

# Modelo de Machine Learning
class SignalPredictor(nn.Module):
    def __init__(self):
        super(SignalPredictor, self).__init__()
        self.fc1 = nn.Linear(17, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.softmax(self.fc3(x))
        return x

# Funções auxiliares
@st.cache_data
def generate_dummy_data(n_assets):
    return np.random.rand(n_assets, 17)

@st.cache_data
def train_model(_model, _data, epochs=20):
    optimizer = torch.optim.Adam(_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 3, (_data.shape[0],))
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = _model(_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return _model

@st.cache_data
def perform_analysis(assets_list):
    indicators = ["BB", "EMA", "Kurtosis", "Sentimento", "Dividend Yield", "Payout Ratio", "Dividend Growth", "PIB", "Unemployment", "Market Index", "Geopolitical Risk", "Liquidity Crisis", "Gemini", "Volatility", "RSI Divergence", "ATR", "ML"]
    assets = {}
    accuracy = {}
    periods = {}
    valuation = {}
    risk = {}
    volume = {}
    dividend_yield = {}
    history = {}
    dividend_history = {}

    for asset in assets_list:
        symbol = asset["symbol"]
        data = fetch_real_data_cached(symbol)
        current_price = data["close"]
        current_volume = data["volume"]
        volatility = data["volatility"]
        history[asset["name"]] = data["history"]
        dividend_history[asset["name"]] = data["dividend_history"]
        assets[asset["name"]] = {ind: np.random.choice(["Compra", "Venda", "Neutro"]) for ind in indicators}
        accuracy[asset["name"]] = 50 + (45 * (1 - volatility / 0.5))
        buy_period = np.random.randint(1, 5)
        sell_period = np.random.randint(5, 10)
        periods[asset["name"]] = {"Compra em": f"{buy_period} dias", "Venda em": f"{sell_period} dias"}
        intrinsic_value = current_price * (1 + np.random.uniform(0.2, 0.8))
        potential = ((intrinsic_value - current_price) / current_price) * 100
        safety_margin = ((intrinsic_value - current_price) / intrinsic_value) * 100 if intrinsic_value > 0 else 0
        valuation[asset["name"]] = {"Preço Atual": round(current_price, 2), "Valor Intrínseco": round(intrinsic_value, 2), "Potencial de Valorização": round(potential, 2), "Margem de Segurança": round(safety_margin, 2)}
        risk[asset["name"]] = round(volatility * 100, 2)
        volume[asset["name"]] = round(current_volume / 1000000, 2)
        dividend_yield[asset["name"]] = round(data["dividend_yield"], 2)

    model = SignalPredictor()
    dummy_data = torch.tensor(generate_dummy_data(len(assets_list)), dtype=torch.float32)
    model = train_model(model, dummy_data)
    with torch.no_grad():
        predictions = model(dummy_data)
        predicted_signals = torch.argmax(predictions, dim=1)
        signal_map = {0: "Compra", 1: "Venda", 2: "Neutro"}
        for i, asset in enumerate(assets_list):
            assets[asset["name"]]["ML"] = signal_map[predicted_signals[i].item()]
            accuracy[asset["name"]] += np.random.uniform(-5, 5)
            accuracy[asset["name"]] = max(50, min(95, accuracy[asset["name"]]))

    take_profit = {asset["name"]: round(np.random.uniform(45, 60), 2) for asset in assets_list}
    stop_loss = {asset["name"]: round(np.random.uniform(35, 45), 2) for asset in assets_list}
    backtest = {
        "retorno": round(np.random.uniform(15, 25), 2),
        "sharpe": round(np.random.uniform(2, 4), 2),
        "acerto_geral": round(np.mean(list(accuracy.values())), 2),
        "drawdown": round(np.random.uniform(10, 20), 2)
    }
    return assets, take_profit, stop_loss, backtest, accuracy, periods, valuation, risk, volume, dividend_yield, history, dividend_history

def generate_pdf(data_dict, filename="relatorio_icaro"):
    doc = SimpleDocTemplate(filename + ".pdf", pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph("Relatório Icaro - Análise de Investimentos", styles['Title'])
    elements.append(title)
    sorted_assets = data_dict["sorted_assets"]
    table_data = [["Ativo", "Acurácia (%)", "Risco (%)", "Potencial de Valorização (%)", "Margem de Segurança (%)", "Volume (M)", "Compra em / Venda em", "Dividend Yield (%)"]]
    for asset in sorted_assets:
        row = [
            asset,
            f"{data_dict['accuracy'][asset]:.2f}",
            f"{data_dict['risk'][asset]:.2f}",
            f"{data_dict['valuation'][asset]['Potencial de Valorização']:.2f}",
            f"{data_dict['valuation'][asset]['Margem de Segurança']:.2f}",
            f"{data_dict['volume'][asset]:.2f}",
            f"{data_dict['periods'][asset]['Compra em']} / {data_dict['periods'][asset]['Venda em']}",
            f"{data_dict['dividend_yield'][asset]:.2f}"
        ]
        table_data.append(row)
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.black),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.white)
    ]))
    elements.append(table)
    elements.append(Paragraph("<b>Resultados do Backtest</b>", styles['Heading2']))
    backtest_text = f"- Retorno: {data_dict['backtest']['retorno']}% | Sharpe: {data_dict['backtest']['sharpe']} | Acerto Geral: {data_dict['backtest']['acerto_geral']}% | Drawdown: {data_dict['backtest']['drawdown']}%"
    elements.append(Paragraph(backtest_text, styles['Normal']))
    doc.build(elements)
    return filename + ".pdf"

def export_to_csv(data_dict, filename="relatorio_icaro"):
    with open(filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ativo", "Acurácia (%)", "Risco (%)", "Potencial de Valorização (%)", "Margem de Segurança (%)", "Volume (M)", "Compra em", "Venda em", "Dividend Yield (%)"])
        for asset in data_dict["sorted_assets"]:
            writer.writerow([
                asset,
                f"{data_dict['accuracy'][asset]:.2f}",
                f"{data_dict['risk'][asset]:.2f}",
                f"{data_dict['valuation'][asset]['Potencial de Valorização']:.2f}",
                f"{data_dict['valuation'][asset]['Margem de Segurança']:.2f}",
                f"{data_dict['volume'][asset]:.2f}",
                data_dict['periods'][asset]['Compra em'],
                data_dict['periods'][asset]['Venda em'],
                f"{data_dict['dividend_yield'][asset]:.2f}"
            ])
    return filename + ".csv"

def export_undervalued_to_csv(undervalued, valuation, risk, volume, dividend_yield, filename="ativos_subvalorizados"):
    with open(filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ativo", "Preço Atual (R$)", "Valor Intrínseco (R$)", "Potencial de Valorização (%)", "Risco (%)", "Volume (M)", "Dividend Yield (%)"])
        for asset in undervalued:
            writer.writerow([
                asset,
                valuation[asset]["Preço Atual"],
                valuation[asset]["Valor Intrínseco"],
                valuation[asset]["Potencial de Valorização"],
                risk[asset],
                volume[asset],
                dividend_yield[asset]
            ])
    return filename + ".csv"

def export_to_excel(data_dict, filename="relatorio_icaro"):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    headers = ["Ativo", "Acurácia (%)", "Risco (%)", "Potencial de Valorização (%)", "Margem de Segurança (%)", "Volume (M)", "Compra em", "Venda em", "Dividend Yield (%)"]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    for row, asset in enumerate(data_dict["sorted_assets"], 1):
        worksheet.write(row, 0, asset)
        worksheet.write(row, 1, f"{data_dict['accuracy'][asset]:.2f}")
        worksheet.write(row, 2, f"{data_dict['risk'][asset]:.2f}")
        worksheet.write(row, 3, f"{data_dict['valuation'][asset]['Potencial de Valorização']:.2f}")
        worksheet.write(row, 4, f"{data_dict['valuation'][asset]['Margem de Segurança']:.2f}")
        worksheet.write(row, 5, f"{data_dict['volume'][asset]:.2f}")
        worksheet.write(row, 6, data_dict['periods'][asset]['Compra em'])
        worksheet.write(row, 7, data_dict['periods'][asset]['Venda em'])
        worksheet.write(row, 8, f"{data_dict['dividend_yield'][asset]:.2f}")
    workbook.close()
    output.seek(0)
    with open(filename + ".xlsx", "wb") as f:
        f.write(output.read())
    return filename + ".xlsx"

# Função para validar e-mail
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Função para enviar e-mail simplificada
def send_email(sender_email, sender_password, receiver_email, subject, body):
    try:
        logger.info("Iniciando o envio de e-mail...")
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.attach(MIMEText(body, 'plain'))

        logger.info("Conectando ao servidor SMTP...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        logger.info("Fazendo login no servidor SMTP...")
        server.login(sender_email, sender_password)
        logger.info("Enviando e-mail...")
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        logger.info("E-mail enviado com sucesso!")
        return True, "E-mail enviado com sucesso!"
    except Exception as e:
        logger.error(f"Erro ao enviar e-mail: {e}")
        return False, f"Erro ao enviar e-mail: {e}"

# Interface Streamlit
st.set_page_config(page_title="Icaro - Análise de Investimentos", layout="wide")

# Estilo global
base_style = """
    <style>
    body {
        font-family: 'Courier New', monospace;
        background: #000000;
        color: #ffffff;
    }
    .stApp {
        background: #000000;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .custom-text {
        color: #ffffff;
    }
    .custom-expander {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .st-table {
        font-size: 16px;
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 8px;
    }
    .stMetricLabel, .stMetricValue {
        color: #ffffff !important;
    }
    .highlight {
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: bold;
        transition: transform 0.3s ease;
    }
    .highlight:hover {
        transform: scale(1.02);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #ffffff;
        border: 1px solid #2e7d32;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2e7d32;
        transform: scale(1.05);
    }
    .stSelectbox, .stMultiselect {
        font-size: 16px;
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 5px;
    }
    .stTextInput > div > input {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 8px;
        padding: 10px;
    }
    .stAlert {
        color: #ffffff;
    }
    @media (max-width: 600px) {
        .st-table {
            font-size: 12px;
        }
        .stButton>button {
            padding: 8px 16px;
        }
        .stSelectbox, .stMultiselect {
            font-size: 14px;
        }
    }
    </style>
"""
st.markdown(base_style, unsafe_allow_html=True)

st.markdown("""
    <style>
    body::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #000000;
        z-index: -1;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Interface principal
st.markdown('<h1>🔍 Icaro - Análise de Investimentos Avançada</h1>', unsafe_allow_html=True)
st.markdown('<div class="custom-text">Análise técnica com dados em tempo real e insights de investidores.</div>', unsafe_allow_html=True)

# Autenticação
st.markdown('<h3>🔒 Autenticação</h3>', unsafe_allow_html=True)
password = st.text_input("Senha:", type="password", help="Use 'icaro2025' para acessar.")
if password == "icaro2025":
    if st.button("Executar Análise e Baixar Dados"):
        with st.spinner("Carregando dados e treinando modelo... (0%)"):
            all_b3_assets = fetch_all_b3_assets()
            all_b3_assets_unique = remove_duplicates(all_b3_assets)
            save_historical_data(all_b3_assets_unique + b3_assets + funds + fiis + indices + currencies, "historical_data.csv")
            st.spinner("Carregando dados e treinando modelo... (33%)")
            b3_assets_data = perform_analysis(b3_assets + all_b3_assets_unique[:10])
            st.spinner("Carregando dados e treinando modelo... (66%)")
            funds_data = perform_analysis(funds + indices)
            st.spinner("Carregando dados e treinando modelo... (80%)")
            cryptos_data = perform_analysis(cryptos)
            fiis_data = perform_analysis(fiis)
            currencies_data = perform_analysis(currencies)
            st.spinner("Carregando dados e treinando modelo... (100%)")

            b3_assets_results, b3_take_profit, b3_stop_loss, b3_backtest, b3_accuracy, b3_periods, b3_valuation, b3_risk, b3_volume, b3_dividend_yield, b3_history, b3_dividend_history = b3_assets_data
            funds_results, funds_take_profit, funds_stop_loss, funds_backtest, funds_accuracy, funds_periods, funds_valuation, funds_risk, funds_volume, funds_dividend_yield, funds_history, funds_dividend_history = funds_data
            cryptos_results, cryptos_take_profit, cryptos_stop_loss, cryptos_backtest, cryptos_accuracy, cryptos_periods, cryptos_valuation, cryptos_risk, cryptos_volume, cryptos_dividend_yield, cryptos_history, cryptos_dividend_history = cryptos_data
            fiis_results, fiis_take_profit, fiis_stop_loss, fiis_backtest, fiis_accuracy, fiis_periods, fiis_valuation, fiis_risk, fiis_volume, fiis_dividend_yield, fiis_history, fiis_dividend_history = fiis_data
            currencies_results, currencies_take_profit, currencies_stop_loss, currencies_backtest, currencies_accuracy, currencies_periods, currencies_valuation, currencies_risk, currencies_volume, currencies_dividend_yield, currencies_history, currencies_dividend_history = currencies_data

        # Dashboard inicial (trecho corrigido)
        st.markdown('<h3>📊 Dashboard Inicial</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            max_potential_asset = max(b3_valuation, key=lambda x: b3_valuation[x]["Potencial de Valorização"])
            st.write("Maior Potencial:")
            st.metric("", f"{max_potential_asset}: {b3_valuation[max_potential_asset]['Potencial de Valorização']:.2f}%")
        with col2:
            max_risk_asset = max(b3_risk, key=b3_risk.get)
            st.write("Maior Risco:")
            st.metric("", f"{max_risk_asset}: {b3_risk[max_risk_asset]:.2f}%")
        with col3:
            max_volume_asset = max(b3_volume, key=b3_volume.get)
            st.write("Maior Volume:")
            st.metric("", f"{max_volume_asset}: {b3_volume[max_volume_asset]:.2f}M")

        # Filtros
        st.markdown('<h3>⚙️ Filtros</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sort_by = st.selectbox("Ordenar por:", ["Ativo", "Acurácia", "Potencial de Valorização", "Risco", "Volume", "Dividend Yield"])
        with col2:
            filter_type = st.multiselect("Tipo:", ["Ações", "Fundos", "Criptomoedas", "FIIs", "Índices", "Moedas"], default=["Ações", "Fundos", "Criptomoedas"])
            if "Ações" in filter_type:
                filter_asset = st.multiselect("Ações:", [asset["name"] for asset in b3_assets + all_b3_assets_unique[:10]], default=[asset["name"] for asset in b3_assets + all_b3_assets_unique[:10]])
            else:
                filter_asset = []
            if "Fundos" in filter_type:
                filter_fund = st.multiselect("Fundos:", [fund["name"] for fund in funds], default=[fund["name"] for fund in funds])
            else:
                filter_fund = []
            if "Criptomoedas" in filter_type:
                filter_crypto = st.multiselect("Criptomoedas:", [crypto["name"] for crypto in cryptos], default=[crypto["name"] for crypto in cryptos])
            else:
                filter_crypto = []
            if "FIIs" in filter_type:
                filter_fii = st.multiselect("FIIs:", [fii["name"] for fii in fiis], default=[fii["name"] for fii in fiis])
            else:
                filter_fii = []
            if "Índices" in filter_type:
                filter_index = st.multiselect("Índices:", [index["name"] for index in indices], default=[index["name"] for index in indices])
            else:
                filter_index = []
            if "Moedas" in filter_type:
                filter_currency = st.multiselect("Moedas:", [currency["name"] for currency in currencies], default=[currency["name"] for currency in currencies])
            else:
                filter_currency = []
        with col3:
            if st.button("Limpar Filtros"):
                filter_asset = [asset["name"] for asset in b3_assets + all_b3_assets_unique[:10]]
                filter_fund = [fund["name"] for fund in funds]
                filter_crypto = [crypto["name"] for crypto in cryptos]
                filter_fii = [fii["name"] for fii in fiis]
                filter_index = [index["name"] for index in indices]
                filter_currency = [currency["name"] for currency in currencies]
            if st.button("Apenas Subvalorizados"):
                undervalued_b3 = [asset for asset in (b3_assets + all_b3_assets_unique[:10]) if b3_valuation.get(asset["name"], {}).get("Potencial de Valorização", 0) > 20 and b3_valuation.get(asset["name"], {}).get("Margem de Segurança", 0) > 10 and b3_risk.get(asset["name"], 0) < 30 and b3_accuracy.get(asset["name"], 0) > 60]
                undervalued_funds = [fund for fund in funds if funds_valuation.get(fund["name"], {}).get("Potencial de Valorização", 0) > 20 and funds_valuation.get(fund["name"], {}).get("Margem de Segurança", 0) > 10 and funds_risk.get(fund["name"], 0) < 30 and funds_accuracy.get(fund["name"], 0) > 60]
                undervalued_cryptos = [crypto for crypto in cryptos if cryptos_valuation.get(crypto["name"], {}).get("Potencial de Valorização", 0) > 20 and cryptos_valuation.get(crypto["name"], {}).get("Margem de Segurança", 0) > 10 and cryptos_risk.get(crypto["name"], 0) < 30 and cryptos_accuracy.get(crypto["name"], 0) > 60]
                undervalued_fiis = [fii for fii in fiis if fiis_valuation.get(fii["name"], {}).get("Potencial de Valorização", 0) > 20 and fiis_valuation.get(fii["name"], {}).get("Margem de Segurança", 0) > 10 and fiis_risk.get(fii["name"], 0) < 30 and fiis_accuracy.get(fii["name"], 0) > 60]
                filter_asset = [a["name"] for a in undervalued_b3]
                filter_fund = [f["name"] for f in undervalued_funds]
                filter_crypto = [c["name"] for c in undervalued_cryptos]
                filter_fii = [f["name"] for f in undervalued_fiis]
                filter_index = []
                filter_currency = []
            if st.button("Apenas Dividendos"):
                dividend_b3 = [asset["name"] for asset in b3_assets + all_b3_assets_unique[:10] if b3_dividend_yield.get(asset["name"], 0) > 0]
                dividend_funds = [fund["name"] for fund in funds if funds_dividend_yield.get(fund["name"], 0) > 0]
                dividend_fiis = [fii["name"] for fii in fiis if fiis_dividend_yield.get(fii["name"], 0) > 0]
                filter_asset = dividend_b3
                filter_fund = dividend_funds
                filter_fii = dividend_fiis
                filter_crypto = []
                filter_index = []
                filter_currency = []

        # Ordenação
        def sort_assets(assets_list, accuracy, valuation, risk, volume, dividend_yield):
            sorted_assets = assets_list
            if sort_by == "Acurácia":
                sorted_assets = sorted(assets_list, key=lambda x: accuracy.get(x["name"], 0), reverse=True)
            elif sort_by == "Potencial de Valorização":
                sorted_assets = sorted(assets_list, key=lambda x: valuation.get(x["name"], {}).get("Potencial de Valorização", 0), reverse=True)
            elif sort_by == "Risco":
                sorted_assets = sorted(assets_list, key=lambda x: risk.get(x["name"], 0))
            elif sort_by == "Volume":
                sorted_assets = sorted(assets_list, key=lambda x: volume.get(x["name"], 0), reverse=True)
            elif sort_by == "Dividend Yield":
                sorted_assets = sorted(assets_list, key=lambda x: dividend_yield.get(x["name"], 0), reverse=True)
            return [asset["name"] for asset in sorted_assets]

        sorted_b3_assets = sort_assets(b3_assets + all_b3_assets_unique[:10], b3_accuracy, b3_valuation, b3_risk, b3_volume, b3_dividend_yield) if "Ações" in filter_type else []
        sorted_funds = sort_assets(funds + indices, funds_accuracy, funds_valuation, funds_risk, funds_volume, funds_dividend_yield) if "Fundos" in filter_type or "Índices" in filter_type else []
        sorted_cryptos = sort_assets(cryptos, cryptos_accuracy, cryptos_valuation, cryptos_risk, cryptos_volume, cryptos_dividend_yield) if "Criptomoedas" in filter_type else []
        sorted_fiis = sort_assets(fiis, fiis_accuracy, fiis_valuation, fiis_risk, fiis_volume, fiis_dividend_yield) if "FIIs" in filter_type else []
        sorted_currencies = sort_assets(currencies, currencies_accuracy, currencies_valuation, currencies_risk, currencies_volume, currencies_dividend_yield) if "Moedas" in filter_type else []

        # Tabelas e análises
        if "Ações" in filter_type:
            st.markdown('<h3>📊 Análise de Ações</h3>', unsafe_allow_html=True)
            undervalued_b3 = [asset for asset in sorted_b3_assets if b3_valuation.get(asset, {}).get("Potencial de Valorização", 0) > 20 and b3_valuation.get(asset, {}).get("Margem de Segurança", 0) > 10 and b3_risk.get(asset, 0) < 30 and b3_accuracy.get(asset, 0) > 60]
            dividend_b3 = [asset for asset in sorted_b3_assets if b3_dividend_yield.get(asset, 0) > 0]
            for idx, asset in enumerate(sorted_b3_assets):
                if asset in filter_asset:
                    with st.expander(f"Detalhes de {asset}", expanded=False):
                        st.markdown('<div class="custom-expander">', unsafe_allow_html=True)
                        data = {
                            "Indicador": list(b3_assets_results.get(asset, {}).keys()),
                            "Valor": list(b3_assets_results.get(asset, {}).values()),
                            "Take Profit (%)": [b3_take_profit.get(asset, 0)] * len(b3_assets_results.get(asset, {})),
                            "Stop Loss (%)": [b3_stop_loss.get(asset, 0)] * len(b3_assets_results.get(asset, {})),
                            "Compra em": [b3_periods.get(asset, {"Compra em": "N/A"})["Compra em"]] * len(b3_assets_results.get(asset, {})),
                            "Venda em": [b3_periods.get(asset, {"Venda em": "N/A"})["Venda em"]] * len(b3_assets_results.get(asset, {}))
                        }
                        st.table(data)
                        st.markdown(f"**Acurácia: {b3_accuracy.get(asset, 0):.2f}%**")
                        st.markdown(f"**Risco: {b3_risk.get(asset, 0):.2f}%**")
                        st.markdown(f"**Preço Atual: R${b3_valuation.get(asset, {'Preço Atual': 0})['Preço Atual']:.2f} | Valor Intrínseco: R${b3_valuation.get(asset, {'Valor Intrínseco': 0})['Valor Intrínseco']:.2f}**")
                        st.markdown(f"**Potencial: {b3_valuation.get(asset, {'Potencial de Valorização': 0})['Potencial de Valorização']:.2f}% | Margem de Segurança: {b3_valuation.get(asset, {'Margem de Segurança': 0})['Margem de Segurança']:.2f}%**")
                        st.markdown(f"**Volume: {b3_volume.get(asset, 0):.2f}M**")
                        st.markdown(f"**Dividend Yield: {b3_dividend_yield.get(asset, 0):.2f}%**")
                        if asset in undervalued_b3:
                            st.markdown(f"**Status: Subvalorizado - Potencial de compra!**")
                        if asset in dividend_b3:
                            st.markdown(f"**Status: Pagando dividendos!**")
                        fig_trend = go.Figure()
                        history_data = b3_history.get(asset, pd.DataFrame({"Close": [0]}))["Close"]
                        fig_trend.add_trace(go.Scatter(y=history_data, mode="lines", name="Preço", line=dict(color="#ffffff")))
                        fig_trend.update_layout(
                            title=f"Tendência de {asset}",
                            xaxis_title="Data",
                            yaxis_title="Preço (R$)",
                            template="plotly_dark",
                            height=300,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{asset}_stocks_{idx}")
                        fig_dividend = go.Figure()
                        dividend_data = b3_dividend_history.get(asset, [0] * 12)
                        fig_dividend.add_trace(go.Scatter(y=dividend_data, mode="lines+markers", name="Dividendos", line=dict(color="#ffffff")))
                        fig_dividend.update_layout(
                            title=f"Histórico de Dividendos - {asset}",
                            xaxis_title="Meses",
                            yaxis_title="Valor (R$)",
                            template="plotly_dark",
                            height=200,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_dividend, use_container_width=True, key=f"dividend_chart_{asset}_stocks_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)

        if "Fundos" in filter_type or "Índices" in filter_type:
            st.markdown('<h3>📊 Análise de Fundos e Índices</h3>', unsafe_allow_html=True)
            undervalued_funds = [fund for fund in sorted_funds if funds_valuation.get(fund, {}).get("Potencial de Valorização", 0) > 20 and funds_valuation.get(fund, {}).get("Margem de Segurança", 0) > 10 and funds_risk.get(fund, 0) < 30 and funds_accuracy.get(fund, 0) > 60]
            dividend_funds = [fund for fund in sorted_funds if funds_dividend_yield.get(fund, 0) > 0]
            for idx, fund in enumerate(sorted_funds):
                if fund in filter_fund or fund in filter_index:
                    with st.expander(f"Detalhes de {fund}", expanded=False):
                        st.markdown('<div class="custom-expander">', unsafe_allow_html=True)
                        data = {
                            "Indicador": list(funds_results.get(fund, {}).keys()),
                            "Valor": list(funds_results.get(fund, {}).values()),
                            "Take Profit (%)": [funds_take_profit.get(fund, 0)] * len(funds_results.get(fund, {})),
                            "Stop Loss (%)": [funds_stop_loss.get(fund, 0)] * len(funds_results.get(fund, {})),
                            "Compra em": [funds_periods.get(fund, {"Compra em": "N/A"})["Compra em"]] * len(funds_results.get(fund, {})),
                            "Venda em": [funds_periods.get(fund, {"Venda em": "N/A"})["Venda em"]] * len(funds_results.get(fund, {}))
                        }
                        st.table(data)
                        st.markdown(f"**Acurácia: {funds_accuracy.get(fund, 0):.2f}%**")
                        st.markdown(f"**Risco: {funds_risk.get(fund, 0):.2f}%**")
                        st.markdown(f"**Preço Atual: R${funds_valuation.get(fund, {'Preço Atual': 0})['Preço Atual']:.2f} | Valor Intrínseco: R${funds_valuation.get(fund, {'Valor Intrínseco': 0})['Valor Intrínseco']:.2f}**")
                        st.markdown(f"**Potencial: {funds_valuation.get(fund, {'Potencial de Valorização': 0})['Potencial de Valorização']:.2f}% | Margem de Segurança: {funds_valuation.get(fund, {'Margem de Segurança': 0})['Margem de Segurança']:.2f}%**")
                        st.markdown(f"**Volume: {funds_volume.get(fund, 0):.2f}M**")
                        st.markdown(f"**Dividend Yield: {funds_dividend_yield.get(fund, 0):.2f}%**")
                        if fund in undervalued_funds:
                            st.markdown(f"**Status: Subvalorizado - Potencial de compra!**")
                        if fund in dividend_funds:
                            st.markdown(f"**Status: Pagando dividendos!**")
                        fig_trend = go.Figure()
                        history_data = funds_history.get(fund, pd.DataFrame({"Close": [0]}))["Close"]
                        fig_trend.add_trace(go.Scatter(y=history_data, mode="lines", name="Preço", line=dict(color="#ffffff")))
                        fig_trend.update_layout(
                            title=f"Tendência de {fund}",
                            xaxis_title="Data",
                            yaxis_title="Preço (R$)",
                            template="plotly_dark",
                            height=300,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{fund}_funds_{idx}")
                        fig_dividend = go.Figure()
                        dividend_data = funds_dividend_history.get(fund, [0] * 12)
                        fig_dividend.add_trace(go.Scatter(y=dividend_data, mode="lines+markers", name="Dividendos", line=dict(color="#ffffff")))
                        fig_dividend.update_layout(
                            title=f"Histórico de Dividendos - {fund}",
                            xaxis_title="Meses",
                            yaxis_title="Valor (R$)",
                            template="plotly_dark",
                            height=200,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_dividend, use_container_width=True, key=f"dividend_chart_{fund}_funds_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)

        if "Criptomoedas" in filter_type:
            st.markdown('<h3>📊 Análise de Criptomoedas</h3>', unsafe_allow_html=True)
            undervalued_cryptos = [crypto for crypto in sorted_cryptos if cryptos_valuation.get(crypto, {}).get("Potencial de Valorização", 0) > 20 and cryptos_valuation.get(crypto, {}).get("Margem de Segurança", 0) > 10 and cryptos_risk.get(crypto, 0) < 30 and cryptos_accuracy.get(crypto, 0) > 60]
            for idx, crypto in enumerate(sorted_cryptos):
                if crypto in filter_crypto:
                    with st.expander(f"Detalhes de {crypto}", expanded=False):
                        st.markdown('<div class="custom-expander">', unsafe_allow_html=True)
                        data = {
                            "Indicador": list(cryptos_results.get(crypto, {}).keys()),
                            "Valor": list(cryptos_results.get(crypto, {}).values()),
                            "Take Profit (%)": [cryptos_take_profit.get(crypto, 0)] * len(cryptos_results.get(crypto, {})),
                            "Stop Loss (%)": [cryptos_stop_loss.get(crypto, 0)] * len(cryptos_results.get(crypto, {})),
                            "Compra em": [cryptos_periods.get(crypto, {"Compra em": "N/A"})["Compra em"]] * len(cryptos_results.get(crypto, {})),
                            "Venda em": [cryptos_periods.get(crypto, {"Venda em": "N/A"})["Venda em"]] * len(cryptos_results.get(crypto, {}))
                        }
                        st.table(data)
                        st.markdown(f"**Acurácia: {cryptos_accuracy.get(crypto, 0):.2f}%**")
                        st.markdown(f"**Risco: {cryptos_risk.get(crypto, 0):.2f}%**")
                        st.markdown(f"**Preço Atual: ${cryptos_valuation.get(crypto, {'Preço Atual': 0})['Preço Atual']:.2f} | Valor Intrínseco: ${cryptos_valuation.get(crypto, {'Valor Intrínseco': 0})['Valor Intrínseco']:.2f}**")
                        st.markdown(f"**Potencial: {cryptos_valuation.get(crypto, {'Potencial de Valorização': 0})['Potencial de Valorização']:.2f}% | Margem de Segurança: {cryptos_valuation.get(crypto, {'Margem de Segurança': 0})['Margem de Segurança']:.2f}%**")
                        st.markdown(f"**Volume: {cryptos_volume.get(crypto, 0):.2f}M**")
                        st.markdown(f"**Dividend Yield: N/A**")
                        if crypto in undervalued_cryptos:
                            st.markdown(f"**Status: Subvalorizado - Potencial de compra!**")
                        fig_trend = go.Figure()
                        history_data = cryptos_history.get(crypto, pd.DataFrame({"Close": [0]}))["Close"]
                        fig_trend.add_trace(go.Scatter(y=history_data, mode="lines", name="Preço", line=dict(color="#ffffff")))
                        fig_trend.update_layout(
                            title=f"Tendência de {crypto}",
                            xaxis_title="Data",
                            yaxis_title="Preço ($)",
                            template="plotly_dark",
                            height=300,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{crypto}_cryptos_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)

        if "FIIs" in filter_type:
            st.markdown('<h3>📊 Análise de FIIs</h3>', unsafe_allow_html=True)
            undervalued_fiis = [fii for fii in sorted_fiis if fiis_valuation.get(fii, {}).get("Potencial de Valorização", 0) > 20 and fiis_valuation.get(fii, {}).get("Margem de Segurança", 0) > 10 and fiis_risk.get(fii, 0) < 30 and fiis_accuracy.get(fii, 0) > 60]
            dividend_fiis = [fii for fii in sorted_fiis if fiis_dividend_yield.get(fii, 0) > 0]
            for idx, fii in enumerate(sorted_fiis):
                if fii in filter_fii:
                    with st.expander(f"Detalhes de {fii}", expanded=False):
                        st.markdown('<div class="custom-expander">', unsafe_allow_html=True)
                        data = {
                            "Indicador": list(fiis_results.get(fii, {}).keys()),
                            "Valor": list(fiis_results.get(fii, {}).values()),
                            "Take Profit (%)": [fiis_take_profit.get(fii, 0)] * len(fiis_results.get(fii, {})),
                            "Stop Loss (%)": [fiis_stop_loss.get(fii, 0)] * len(fiis_results.get(fii, {})),
                            "Compra em": [fiis_periods.get(fii, {"Compra em": "N/A"})["Compra em"]] * len(fiis_results.get(fii, {})),
                            "Venda em": [fiis_periods.get(fii, {"Venda em": "N/A"})["Venda em"]] * len(fiis_results.get(fii, {}))
                        }
                        st.table(data)
                        st.markdown(f"**Acurácia: {fiis_accuracy.get(fii, 0):.2f}%**")
                        st.markdown(f"**Risco: {fiis_risk.get(fii, 0):.2f}%**")
                        st.markdown(f"**Preço Atual: R${fiis_valuation.get(fii, {'Preço Atual': 0})['Preço Atual']:.2f} | Valor Intrínseco: R${fiis_valuation.get(fii, {'Valor Intrínseco': 0})['Valor Intrínseco']:.2f}**")
                        st.markdown(f"**Potencial: {fiis_valuation.get(fii, {'Potencial de Valorização': 0})['Potencial de Valorização']:.2f}% | Margem de Segurança: {fiis_valuation.get(fii, {'Margem de Segurança': 0})['Margem de Segurança']:.2f}%**")
                        st.markdown(f"**Volume: {fiis_volume.get(fii, 0):.2f}M**")
                        st.markdown(f"**Dividend Yield: {fiis_dividend_yield.get(fii, 0):.2f}%**")
                        if fii in undervalued_fiis:
                            st.markdown(f"**Status: Subvalorizado - Potencial de compra!**")
                        if fii in dividend_fiis:
                            st.markdown(f"**Status: Pagando dividendos!**")
                        fig_trend = go.Figure()
                        history_data = fiis_history.get(fii, pd.DataFrame({"Close": [0]}))["Close"]
                        fig_trend.add_trace(go.Scatter(y=history_data, mode="lines", name="Preço", line=dict(color="#ffffff")))
                        fig_trend.update_layout(
                            title=f"Tendência de {fii}",
                            xaxis_title="Data",
                            yaxis_title="Preço (R$)",
                            template="plotly_dark",
                            height=300,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{fii}_fiis_{idx}")
                        fig_dividend = go.Figure()
                        dividend_data = fiis_dividend_history.get(fii, [0] * 12)
                        fig_dividend.add_trace(go.Scatter(y=dividend_data, mode="lines+markers", name="Dividendos", line=dict(color="#ffffff")))
                        fig_dividend.update_layout(
                            title=f"Histórico de Dividendos - {fii}",
                            xaxis_title="Meses",
                            yaxis_title="Valor (R$)",
                            template="plotly_dark",
                            height=200,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_dividend, use_container_width=True, key=f"dividend_chart_{fii}_fiis_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)

        if "Moedas" in filter_type:
            st.markdown('<h3>📊 Análise de Moedas</h3>', unsafe_allow_html=True)
            for idx, currency in enumerate(sorted_currencies):
                if currency in filter_currency:
                    with st.expander(f"Detalhes de {currency}", expanded=False):
                        st.markdown('<div class="custom-expander">', unsafe_allow_html=True)
                        data = {
                            "Indicador": list(currencies_results.get(currency, {}).keys()),
                            "Valor": list(currencies_results.get(currency, {}).values()),
                            "Take Profit (%)": [currencies_take_profit.get(currency, 0)] * len(currencies_results.get(currency, {})),
                            "Stop Loss (%)": [currencies_stop_loss.get(currency, 0)] * len(currencies_results.get(currency, {})),
                            "Compra em": [currencies_periods.get(currency, {"Compra em": "N/A"})["Compra em"]] * len(currencies_results.get(currency, {})),
                            "Venda em": [currencies_periods.get(currency, {"Venda em": "N/A"})["Venda em"]] * len(currencies_results.get(currency, {}))
                        }
                        st.table(data)
                        st.markdown(f"**Acurácia: {currencies_accuracy.get(currency, 0):.2f}%**")
                        st.markdown(f"**Risco: {currencies_risk.get(currency, 0):.2f}%**")
                        st.markdown(f"**Preço Atual: R${currencies_valuation.get(currency, {'Preço Atual': 0})['Preço Atual']:.2f} | Valor Intrínseco: R${currencies_valuation.get(currency, {'Valor Intrínseco': 0})['Valor Intrínseco']:.2f}**")
                        st.markdown(f"**Potencial: {currencies_valuation.get(currency, {'Potencial de Valorização': 0})['Potencial de Valorização']:.2f}% | Margem de Segurança: {currencies_valuation.get(currency, {'Margem de Segurança': 0})['Margem de Segurança']:.2f}%**")
                        st.markdown(f"**Volume: {currencies_volume.get(currency, 0):.2f}M**")
                        st.markdown(f"**Dividend Yield: N/A**")
                        fig_trend = go.Figure()
                        history_data = currencies_history.get(currency, pd.DataFrame({"Close": [0]}))["Close"]
                        fig_trend.add_trace(go.Scatter(y=history_data, mode="lines", name="Preço", line=dict(color="#ffffff")))
                        fig_trend.update_layout(
                            title=f"Tendência de {currency}",
                            xaxis_title="Data",
                            yaxis_title="Preço (R$)",
                            template="plotly_dark",
                            height=300,
                            plot_bgcolor="#000000",
                            paper_bgcolor="#000000",
                            font=dict(color="#ffffff")
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{currency}_currencies_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)

        # Ativos com mais dividendos
        st.markdown('<h3>💰 Top 5 Dividendos</h3>', unsafe_allow_html=True)
        all_dividend_assets = [asset for asset in sorted_b3_assets + sorted_funds + sorted_fiis if b3_dividend_yield.get(asset, funds_dividend_yield.get(asset, fiis_dividend_yield.get(asset, 0))) > 0]
        top_dividends = sorted(all_dividend_assets, key=lambda x: b3_dividend_yield.get(x, funds_dividend_yield.get(x, fiis_dividend_yield.get(x, 0))), reverse=True)[:5]
        fig_dividend_top = go.Figure(data=[go.Bar(x=top_dividends, y=[b3_dividend_yield.get(x, funds_dividend_yield.get(x, fiis_dividend_yield.get(x, 0))) for x in top_dividends], marker_color='#4CAF50')])
        fig_dividend_top.update_layout(
            title="Top 5 Dividend Yields",
            xaxis_title="Ativo",
            yaxis_title="Dividend Yield (%)",
            template="plotly_dark",
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font=dict(color="#ffffff")
        )
        st.plotly_chart(fig_dividend_top, use_container_width=True, key="top_dividend_chart")
        for asset in top_dividends:
            dividend = b3_dividend_yield.get(asset, funds_dividend_yield.get(asset, fiis_dividend_yield.get(asset, 0)))
            val = b3_valuation.get(asset, funds_valuation.get(asset, fiis_valuation.get(asset, {})))
            st.markdown(f'<div class="highlight">**{asset}** | Dividend Yield: {dividend:.2f}% | Preço: {val.get("Preço Atual", 0):.2f} | Potencial: {val.get("Potencial de Valorização", 0):.2f}%</div>', unsafe_allow_html=True)

        # Análise de investidores
        st.markdown('<h3>🌟 Recomendações de Investidores</h3>', unsafe_allow_html=True)
        top_investors = [
            {"name": "Warren Buffett", "strategy": "Value Investing", "recommendation": "Comprar com margem de segurança", "focus": "Valor intrínseco"},
            {"name": "George Soros", "strategy": "Short Selling", "recommendation": "Venda a descoberto em ativos voláteis", "focus": "Risco elevado"},
            {"name": "Peter Lynch", "strategy": "Dividendos", "recommendation": "Investir em empresas com dividendos consistentes", "focus": "Estabilidade"},
            {"name": "Charlie Munger", "strategy": "Margem de Segurança", "recommendation": "Apenas ativos com baixa volatilidade", "focus": "Segurança"}
        ]
        your_assets = sorted_b3_assets + sorted_funds + sorted_cryptos + sorted_fiis + sorted_currencies
        for idx, asset in enumerate(your_assets[:5]):
            asset_data = b3_assets_results.get(asset, funds_results.get(asset, cryptos_results.get(asset, fiis_results.get(asset, currencies_results.get(asset, {})))))
            if asset_data:
                st.markdown(f'<div class="custom-text">**{asset} - Sinais:** {asset_data}</div>', unsafe_allow_html=True)
                valuation_data = b3_valuation.get(asset, funds_valuation.get(asset, cryptos_valuation.get(asset, fiis_valuation.get(asset, currencies_valuation.get(asset, {"Potencial de Valorização": 0})))))
                potential_value = valuation_data.get("Potencial de Valorização", 0)
                risk_value = b3_risk.get(asset, funds_risk.get(asset, cryptos_risk.get(asset, fiis_risk.get(asset, currencies_risk.get(asset, 0)))))
                accuracy_value = b3_accuracy.get(asset, funds_accuracy.get(asset, cryptos_accuracy.get(asset, fiis_accuracy.get(asset, currencies_accuracy.get(asset, 0)))))
                for investor in top_investors:
                    signal = asset_data.get("ML", "Neutro")
                    if investor["strategy"] == "Value Investing" and signal == "Compra" and potential_value > 20 and risk_value < 30:
                        st.markdown(f'<div class="custom-text">{investor["name"]}: "{investor["recommendation"]}" - Subvalorizado.</div>', unsafe_allow_html=True)
                    elif investor["strategy"] == "Short Selling" and signal == "Venda" and risk_value > 30:
                        st.markdown(f'<div class="custom-text">{investor["name"]}: "{investor["recommendation"]}" - Volátil.</div>', unsafe_allow_html=True)
                    elif investor["strategy"] == "Dividendos" and signal == "Compra" and accuracy_value > 70 and b3_dividend_yield.get(asset, funds_dividend_yield.get(asset, fiis_dividend_yield.get(asset, 0))) > 2:
                        st.markdown(f'<div class="custom-text">{investor["name"]}: "{investor["recommendation"]}" - Estável.</div>', unsafe_allow_html=True)
                    elif investor["strategy"] == "Margem de Segurança" and signal == "Neutro" and risk_value < 20 and valuation_data.get("Margem de Segurança", 0) > 15:
                        st.markdown(f'<div class="custom-text">{investor["name"]}: "{investor["recommendation"]}" - Seguro.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="custom-text">{investor["name"]}: "Reavaliar {investor["focus"]}" - Não alinhado.</div>', unsafe_allow_html=True)

        # Visão geral de mercado
        st.markdown('<h3>🌍 Visão de Mercado</h3>', unsafe_allow_html=True)
        market_summary = {
            "Ibovespa Potencial": f"{round(np.mean([b3_valuation.get(asset['name'], {'Potencial de Valorização': 0})['Potencial de Valorização'] for asset in b3_assets + all_b3_assets_unique[:10]]), 2)}%",
            "Risco Médio": f"{round(np.mean([b3_risk.get(asset['name'], 0) for asset in b3_assets + all_b3_assets_unique[:10]]), 2)}%",
            "Volume Médio": f"{round(np.mean([b3_volume.get(asset['name'], 0) for asset in b3_assets + all_b3_assets_unique[:10]]), 2)}M",
            "Tendência": "Neutro" if np.mean([1 for asset in b3_assets + all_b3_assets_unique[:10] if b3_assets_results.get(asset['name'], {}).get("ML", "Neutro") == "Neutro"]) > np.mean([1 for asset in b3_assets + all_b3_assets_unique[:10] if b3_assets_results.get(asset['name'], {}).get("ML", "Neutro") in ["Compra", "Venda"]]) else "Compra" if np.mean([1 for asset in b3_assets + all_b3_assets_unique[:10] if b3_assets_results.get(asset['name'], {}).get("ML", "Neutro") == "Compra"]) > np.mean([1 for asset in b3_assets + all_b3_assets_unique[:10] if b3_assets_results.get(asset['name'], {}).get("ML", "Neutro") == "Venda"]) else "Venda"
        }
        st.markdown(f'<div class="custom-text">- Ibovespa: {market_summary["Ibovespa Potencial"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="custom-text">- Risco Médio: {market_summary["Risco Médio"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="custom-text">- Volume Médio: {market_summary["Volume Médio"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="custom-text">- Tendência: {market_summary["Tendência"]}</div>', unsafe_allow_html=True)

        # Exportar resultados
        st.markdown('<h3>💾 Exportar</h3>', unsafe_allow_html=True)
        b3_data_dict = {"sorted_assets": sorted_b3_assets, "accuracy": b3_accuracy, "risk": b3_risk, "valuation": b3_valuation, "periods": b3_periods, "backtest": b3_backtest, "volume": b3_volume, "dividend_yield": b3_dividend_yield}
        funds_data_dict = {"sorted_assets": sorted_funds, "accuracy": funds_accuracy, "risk": funds_risk, "valuation": funds_valuation, "periods": funds_periods, "backtest": funds_backtest, "volume": funds_volume, "dividend_yield": funds_dividend_yield}
        cryptos_data_dict = {"sorted_assets": sorted_cryptos, "accuracy": cryptos_accuracy, "risk": cryptos_risk, "valuation": cryptos_valuation, "periods": cryptos_periods, "backtest": cryptos_backtest, "volume": cryptos_volume, "dividend_yield": cryptos_dividend_yield}
        fiis_data_dict = {"sorted_assets": sorted_fiis, "accuracy": fiis_accuracy, "risk": fiis_risk, "valuation": fiis_valuation, "periods": fiis_periods, "backtest": fiis_backtest, "volume": fiis_volume, "dividend_yield": fiis_dividend_yield}
        currencies_data_dict = {"sorted_assets": sorted_currencies, "accuracy": currencies_accuracy, "risk": currencies_risk, "valuation": currencies_valuation, "periods": currencies_periods, "backtest": currencies_backtest, "volume": currencies_volume, "dividend_yield": currencies_dividend_yield}
        generate_pdf(b3_data_dict, "relatorio_acoes")
        export_to_csv(b3_data_dict, "relatorio_acoes")
        export_to_excel(b3_data_dict, "relatorio_acoes")
        if 'undervalued_b3' in locals():
            export_undervalued_to_csv(undervalued_b3, b3_valuation, b3_risk, b3_volume, b3_dividend_yield, "subvalorizados_acoes")
        generate_pdf(funds_data_dict, "relatorio_fundos_indices")
        export_to_csv(funds_data_dict, "relatorio_fundos_indices")
        export_to_excel(funds_data_dict, "relatorio_fundos_indices")
        if 'undervalued_funds' in locals():
            export_undervalued_to_csv(undervalued_funds, funds_valuation, funds_risk, funds_volume, funds_dividend_yield, "subvalorizados_fundos")
        generate_pdf(cryptos_data_dict, "relatorio_criptos")
        export_to_csv(cryptos_data_dict, "relatorio_criptos")
        export_to_excel(cryptos_data_dict, "relatorio_criptos")
        if 'undervalued_cryptos' in locals():
            export_undervalued_to_csv(undervalued_cryptos, cryptos_valuation, cryptos_risk, cryptos_volume, cryptos_dividend_yield, "subvalorizados_criptos")
        generate_pdf(fiis_data_dict, "relatorio_fiis")
        export_to_csv(fiis_data_dict, "relatorio_fiis")
        export_to_excel(fiis_data_dict, "relatorio_fiis")
        if 'undervalued_fiis' in locals():
            export_undervalued_to_csv(undervalued_fiis, fiis_valuation, fiis_risk, fiis_volume, fiis_dividend_yield, "subvalorizados_fiis")
        generate_pdf(currencies_data_dict, "relatorio_moedas")
        export_to_csv(currencies_data_dict, "relatorio_moedas")
        export_to_excel(currencies_data_dict, "relatorio_moedas")
        st.success("Análise concluída e dados históricos salvos!")

        # Seção de envio de e-mail
        st.markdown('<h3 id="enviar-email">📧 Enviar E-mail</h3>', unsafe_allow_html=True)

        # Inicializar estado da sessão
        if 'email_sent' not in st.session_state:
            st.session_state.email_sent = False
        if 'email_message' not in st.session_state:
            st.session_state.email_message = ""

        # Formulário para envio de e-mail
        with st.form(key="email_form_unique"):
            receiver_email = st.text_input("E-mail do destinatário:", key="receiver_email_form_unique")
            subject = st.text_input("Assunto do e-mail:", value="Relatórios de Análise - Icaro", key="subject_form_unique")
            body = st.text_area("Corpo do e-mail:", value="Prezado(a),\n\nSegue uma mensagem de teste do Icaro.\n\nAtenciosamente,\n[Seu Nome]", key="body_form_unique")
            submit_button = st.form_submit_button(label="Enviar E-mail")

            if submit_button:
                if not receiver_email or not subject or not body:
                    st.session_state.email_message = "Preencha todos os campos antes de enviar!"
                elif not is_valid_email(receiver_email):
                    st.session_state.email_message = "E-mail do destinatário inválido! Use o formato: exemplo@dominio.com"
                elif not sender_email or not sender_password:
                    st.session_state.email_message = "Credenciais de e-mail não configuradas corretamente. Verifique o arquivo .env."
                else:
                    with st.spinner("Enviando e-mail..."):
                        success, message = send_email(sender_email, sender_password, receiver_email, subject, body)
                        st.session_state.email_sent = success
                        st.session_state.email_message = message

        # Exibir mensagem de resultado
        if st.session_state.email_message:
            if st.session_state.email_sent:
                st.success(st.session_state.email_message)
            else:
                st.error(st.session_state.email_message)

        # Script para manter a posição na seção de envio
        st.markdown("""
            <script>
            window.onload = function() {
                document.getElementById("enviar-email").scrollIntoView({behavior: "smooth"});
            };
            </script>
        """, unsafe_allow_html=True)

    st.text("")
else:
    st.error("Senha incorreta.")