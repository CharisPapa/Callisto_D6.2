from transformers import AutoTokenizer, AutoModelForTokenClassification
from gr_nlp_toolkit import Pipeline
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
import re
import unidecode
import time

import flask
from flask import request, jsonify
import requests
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

start_time = time.time()
# ---------------MODELS----------------
#   -----Mulitlingual (fr)
tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
multinlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

#   -----es flair
# load tagger
taggerES = SequenceTagger.load("flair/ner-spanish-large")

#   -----de flair
# load tagger
taggerDE = SequenceTagger.load("flair/ner-german-large")

#   -----nl flair
# load tagger
taggerNL = SequenceTagger.load("flair/ner-dutch-large")

#   -----gr nlpaueb-gr-nlp-toolkit
grnlp = Pipeline("ner")

#   -----en flair
# load tagger
taggerEN = SequenceTagger.load("flair/ner-english-large")

print("--- %s seconds ---" % (time.time() - start_time))

@app.route('/', methods=['GET'])
def home():
    # na to kanw na ginetai unchecked ta radios
    return '''
    <h1>Named Entity Recognition service (multilingual tagging).</h1>
    <h2>Also, the language can be selected for language specific models (en, fr, es, de, nl, gr)</h2>
    <p>Input the sentence.</p>
    <form action="/action_page.php">
        <label for="txt">Sentence:</label><br>
        <input type="text" id="txt" name="txt" value=""><br>
        
        <input type="radio" id="en" name="lang" value="en">
        <label for="en">English</label><br>
        <input type="radio" id="fr" name="lang" value="fr">
        <label for="en">French</label><br>
        <input type="radio" id="es" name="lang" value="es">
        <label for="es">Spanish</label><br>
        <input type="radio" id="de" name="lang" value="de">
        <label for="de">German</label><br>
        <input type="radio" id="nl" name="lang" value="nl">
        <label for="nl">Dutch</label><br>
        <input type="radio" id="gr" name="lang" value="gr">
        <label for="gr">Greek</label><br>

        <input type="submit" formaction="/api/v1/resources/ner" value="Submit"><br>
    </form> '''


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.route('/api/v1/resources/ner', methods=['GET'])
def api_filter():
    def openStreetMapCall(locs):
        output_txt = '<p> '
        for loc in locs:
            query = {"q":loc, "format":"jsonv2"}
            response = requests.get('https://nominatim.openstreetmap.org/search.php', params=query)
            if response.json():
                best_result = response.json()
                best_result = best_result[0]
                output_txt += loc + '<br>'
                output_txt += 'lat: ' + best_result['lat'] + '<br>' + 'lon: ' + best_result['lon'] + '<br>'
        output_txt += '\n</p>'
        return output_txt
        
    query_parameters = request.args

    txt = query_parameters.get('txt')
    lang = query_parameters.get('lang')

    if txt:
        start_time = time.time()
        txt = re.sub('@', '', txt)
        txt = re.sub('#', '', txt)
        txt = re.sub(r'http\S+', '', txt)
        unitxt = unidecode.unidecode(txt)
        locs = []
        if lang == 'fr':
            # predict NER tags
            results = multinlp(unitxt)
            print(results)
            for result in results:
                if 'LOC' in result['entity_group']:
                    locs.append(result['word'])
        elif lang == 'es':
            sentence = Sentence(unitxt)
            # predict NER tags
            taggerES.predict(sentence)
            for entity in sentence.get_spans('ner'):
                if 'LOC' in entity.get_label("ner").value:
                    locs.append(entity.text)
        elif lang == 'de':
            sentence = Sentence(unitxt)
            # predict NER tags
            taggerDE.predict(sentence)
            for entity in sentence.get_spans('ner'):
                if 'LOC' in entity.get_label("ner").value:
                    locs.append(entity.text)
        elif lang == 'nl':
            sentence = Sentence(unitxt)
            # predict NER tags
            taggerNL.predict(sentence)
            for entity in sentence.get_spans('ner'):
                if 'LOC' in entity.get_label("ner").value:
                    locs.append(entity.text)
        elif lang == 'gr':
            # predict NER tags
            results = grnlp(txt)
            for result in results.tokens:
                if 'LOC' in result.ner or 'GPE' in result.ner or 'NORP' in result.ner or 'FAC' in result.ner:
                    locs.append(result.text)
        elif lang == 'en':
            sentence = Sentence(unitxt)
            # predict NER tags
            taggerEN.predict(sentence)
            for entity in sentence.get_spans('ner'):
                if 'LOC' in entity.get_label("ner").value:
                    locs.append(entity.text)
        else:
            # predict NER tags
            results = multinlp(unitxt)
            for result in results:
                if 'LOC' in result['entity_group']:
                    locs.append(result['word'])
        
        print("--- %s seconds ---" % (time.time() - start_time))
        return openStreetMapCall(locs)

app.run(host='0.0.0.0')