from plotly import express as px
import streamlit as st
from PIL import Image
import os
from components.sidebar import sidebar
import numpy as np
from cai.models import load_kereas_model
from keras.models import load_model
from cai import layers

def clear_submit():
    st.session_state["submit"] = False

im = Image.open("C:\\Users\\stan_\\lidc-idri-preproc\\diploma_work\\EfficientNet\\ct_icon.ico")
st.set_page_config(page_title="Классификация узелков", page_icon=im, layout="wide")
st.header("Классификация узелков")


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

#sidebar()

uploaded_file = st.file_uploader(
    "Загрузить .npy файл",
    type=["npy"],
    help="Поддерживаются файлы формата .npy размера 32х32",
    on_change=clear_submit,
)



loaded_model = load_kereas_model('C:\\Users\\stan_\\lidc-idri-preproc\\diploma_work\\EfficientNet\\kEffNetV2-1-best_result.hdf5')

def pr():
    f = np.load("C:\\Users\\stan_\\lidc-idri-preproc\\diploma_work\\EfficientNet\\X_tr1.npy")
    return np.argmax(loaded_model.predict(f[123][:, :].reshape(1, f[123][:, :].shape[0], f[123][:, :].shape[1], 1)), axis=-1)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".npy"):
        #doc = parse_pdf(uploaded_file)
        print(uploaded_file, type(uploaded_file))
        f = np.load(uploaded_file)
        f = f[123]
        print(f.shape)
        if f.shape == (32, 32) or f.shape == (32, 32, 1):
            #fig = px.imshow(f,aspect='equal')
            #st.pyplot(f)
            st.image(f, caption='Фрагмент среза КТ легких', width = 200)
            print(f)
            f = f.reshape(1, 32, 32, 1)
            print(f.shape)
            
            #prc = pr()
            prc = [1]
            if prc[0] == 0:
                st.markdown("###Предсказанный класс: Узелок диаметра более 3 мм")
            elif prc[0] == 1:
                st.markdown("### Предсказанный класс: Узелок диаметра менее 3 мм")
            else:
                st.markdown("### Предсказанный класс: Не узелок")
            print('asdsd')
        else:
            raise ValueError("Форма массива должна быть равна (32, 32)!")
        
    else:
        raise ValueError("Данный тип файлов не поддерживается!")
#    #try:
#    #    with st.spinner("Indexing document... This may take a while⏳"):
#    #        index = embed_docs(text)
#    #    st.session_state["api_key_configured"] = True
#    #except OpenAIError as e:
#    #    st.error(e._message)