from flask import Flask, request, redirect, render_template, url_for
import numpy as np
from keras.models import load_model
import re
import spacy
nlp = spacy.load('es_core_news_sm')

def corregir_texto(texto):
    # Patrones a buscar y sus reemplazos
    patrones_reemplazos = [
        (r'<(?:3)+', ' amor '),      # Para "jajaf", "jajajaf", "jajajajaf", ...
        (r'xd+', ' burla '),  # Para "xd", "xdd", "xddd", ...
        (r'(?:ja)+\b', ' burla '),      # Para "jaja", "jajaja", "jajajaja", ...
        (r'cringe', 'avergonzado'),
        (r'hvd', 'avergonzado'),
        (r'huev\S*', 'avergonzado'),
        (r'webd\S*', 'avergonzado'),
        (r'y(?:y)+', 'y'),
        (r'u(?:u)+', 'u'),
        (r'o(?:o)+', 'o'),
        (r'a(?:a)+', 'a'),
        (r'i(?:i)+', 'i'),
        (r'e(?:e)+', 'e'),
        (r's(?:s)+', 's'),
        (r'amix\S*', 'amigo'),
        (r'ag\S*', 'asco'),
        (r'obvi+', 'obvio'),
        (r'manyad\S*', 'adinerado'),
        (r'pituc\S*', 'adinerado'),
        (r'lind\S*', 'lindo'),
        (r'csmr', ' mierda '),  # no encontré un términom más adecuado
        (r'tmr', ' mierda '),   # no encontré un términom más adecuado
        (r'shit', ' mierda '),    # no encontré un términom más adecuado
        (r'amazing', ' asombroso '),
        (r'ala', 'vaya'),
        (r'bail\S*', 'baile'),
        (r'piel\S*', ' ')
    ]
    # Aplicar los patrones de búsqueda y reemplazo
    for patron, reemplazo in patrones_reemplazos:
        texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)

    return texto

def tokenization(cadena):
    text=nlp(cadena)
    text=[word.text.strip() and word.text.lower() for word in text ]
    return " ".join(text)

def remove_words(cadena):
    text=nlp(cadena)
    text=[word.text.strip() for word in text if not word.is_punct and not word.is_stop and not word.text.startswith("@") ]
    return " ".join(text)

def lemmatization(cadena):
    text=nlp(cadena)
    text=[word.lemma_ for word in text]
    return " ".join(text)
def filter_words(cadena):
    pos_validos = {"NOUN", "VERB", "ADJ", "ADV"}
    text=nlp(cadena)
    text=[word.text for word in text if word.pos_ in pos_validos]
    test=[word for word in text if word]
    return " ".join(text)

def norm(cadena):
    cadena=corregir_texto(cadena)
    cadena=tokenization(cadena)
    cadena=remove_words(cadena)
    cadena=lemmatization(cadena)
    cadena=filter_words(cadena)
    return cadena

model = load_model('modeloIA.keras')



app = Flask(__name__, template_folder="templates/")

@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():

    if request.method=="POST":

        msg=request.form.get("miValor")
        msg=norm(msg)
        mytest=[msg]

        mytest=np.array(mytest)
        mymodel=load_model("modeloIA.keras")
        predictions = mymodel.predict(mytest)

        # Convertir las predicciones a etiquetas binarias (0 o 1)
        binary_predictions = (predictions > 0.5).astype(int)

        # Imprimir algunas predicciones
        op = ""
        if binary_predictions[0] == 1: op = "POSITIVO" 
        else : op ="NEGATIVO"
        return render_template("predicted.html", msg=op)
    return redirect("/")



if __name__ == "__main__":
    app.run(debug=True)