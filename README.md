# Limiarização e Segmentação

Trabalho de Computação Gráfica, Tema 2: limiarização global, algoritmo de Otsu e limiarização adaptativa, com OpenCV e Streamlit.

## Como rodar

```
pip install -r requirements.txt
streamlit run app.py
```

O app tem uma aba para cada técnica, com sliders para mudar os parâmetros (T, desfoque, tamanho do bloco e C). As imagens de exemplo ficam na pasta `imagens/`, mas também dá para enviar outra imagem.

Na aba do Otsu, o limiar é calculado por uma implementação própria (função `otsu`) e comparado com o do OpenCV.
