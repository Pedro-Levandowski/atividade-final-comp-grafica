# Trabalho final de Computação Gráfica - Tema 2 (Limiarização)
# pra rodar: streamlit run app.py

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# fizemos o Otsu na mão pra conferir se dá o mesmo T que o OpenCV
def otsu(img):
    # p guarda a chance de cada tom de cinza aparecer na imagem
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()

    # pra cada t: quanto da imagem fica no fundo (w0) e a média acumulada (mu)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_total = mu[-1]

    # fórmula da variância entre as classes, a mesma do slide
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_total * w0 - mu) ** 2 / (w0 * (1 - w0))
    # nas pontas a conta divide por zero, então zeramos esses valores
    sigma_b = np.nan_to_num(sigma_b, nan=0, posinf=0)

    # o T escolhido é onde essa variância é a maior
    return int(np.argmax(sigma_b)), sigma_b


def histograma(img, T):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.hist(img.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(T, color="red", label=f"T = {T}")
    ax.set_xlim(0, 255)
    ax.legend()
    return fig


st.set_page_config(page_title="Limiarização", layout="wide")
st.title("Limiarização e Segmentação")

# usa as imagens da pasta ou uma que a pessoa enviar
pasta = Path(__file__).parent / "imagens"
exemplos = sorted(p.name for p in pasta.glob("*.png"))
escolha = st.sidebar.selectbox("Imagem", exemplos + ["Enviar arquivo"])

if escolha == "Enviar arquivo":
    arquivo = st.sidebar.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])
    if arquivo is None:
        st.stop()
    dados = np.frombuffer(arquivo.read(), np.uint8)
else:
    # o cv2.imread não abria a imagem por causa do acento em "Área de Trabalho", por isso lemos assim
    dados = np.fromfile(pasta / escolha, np.uint8)

# já carrega em tons de cinza, porque a limiarização compara um único valor por pixel com o T
img = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)

aba_global, aba_otsu, aba_adapt = st.tabs(["Global", "Otsu", "Adaptativo"])

with aba_global:
    T = st.slider("Limiar T", 0, 255, 127)
    # pixel maior que T vira branco (255), o resto vira preto
    _, binaria = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Original")
    c2.image(binaria, caption=f"Global com T = {T}")
    c3.pyplot(histograma(img, T))

with aba_otsu:
    k = st.select_slider("Desfoque gaussiano antes do Otsu (0 = sem)",
                         options=[0, 3, 5, 7, 9, 11, 15, 21])
    # borrar antes ajuda bastante quando a imagem tem muito ruído
    entrada = cv2.GaussianBlur(img, (k, k), 0) if k > 0 else img

    T_nosso, sigma_b = otsu(entrada)
    # com THRESH_OTSU o 0 que passamos é ignorado, o OpenCV calcula o T sozinho
    T_cv, binaria = cv2.threshold(entrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.write(f"T da nossa função: **{T_nosso}** | T do OpenCV: **{int(T_cv)}**")

    c1, c2 = st.columns(2)
    c1.pyplot(histograma(entrada, T_nosso))
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(sigma_b)
    ax.axvline(T_nosso, color="red")
    ax.set_title("Variância entre classes para cada t")
    c2.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Original")
    c2.image(entrada, caption=f"Desfoque {k}x{k}" if k else "Sem desfoque")
    c3.image(binaria, caption=f"Otsu, T = {int(T_cv)}")

with aba_adapt:
    c1, c2, c3 = st.columns(3)
    metodo = c1.radio("Método", ["Média", "Gaussiana"])
    bloco = c2.slider("Tamanho do bloco (ímpar)", 3, 151, 25, step=2)
    C = c3.slider("Constante C", -10, 30, 10)

    # na média todos os vizinhos pesam igual, na gaussiana os mais perto do pixel pesam mais
    if metodo == "Média":
        tipo = cv2.ADAPTIVE_THRESH_MEAN_C
        media = cv2.blur(img, (bloco, bloco))
    else:
        tipo = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        media = cv2.GaussianBlur(img, (bloco, bloco), 0)

    # aqui cada pixel é comparado com a vizinhança dele (bloco x bloco), não com um T único
    binaria = cv2.adaptiveThreshold(img, 255, tipo, cv2.THRESH_BINARY, bloco, C)

    # só pra visualizar: o T de cada pixel é a média dos vizinhos menos C,
    # dá pra ver ele ficando mais escuro onde tem sombra
    limiar_local = np.clip(media.astype(int) - C, 0, 255).astype(np.uint8)

    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Original")
    c2.image(limiar_local, caption="Limiar de cada pixel T(x, y)")
    c3.image(binaria, caption=f"Adaptativo, bloco {bloco}, C = {C}")
