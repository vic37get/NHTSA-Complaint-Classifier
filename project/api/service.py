from flask import Flask
import traceback
from flask_classful import FlaskView, route, request
import os
import regex as re
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

model = None
tokenizer = None
clf_pipeline = None

class ClassifySectionsService(FlaskView):
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Remove caracteres indesejados do texto."""
        text = re.sub(r'([•●▪•_·□»«#£¢¿&^~´`¨\t])', ' ', text)
        text = re.sub(r'(-)+', '-', text)
        text = re.sub(r'(\.)+', '.', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s\.$', '.', text)
        return text.lower().strip()
    
    
    @route('/', methods=['GET'])
    def index(self):
        return {'message': 'Classify Compliants', 'status': 0}, 200
    
    
    @route('/status', methods=['GET'])
    def status(self):
        return {'message': 'success', 'status': 0}, 200
    
    
    @route('/load', methods=['POST'])
    def load(self):
        global model, tokenizer, clf_pipeline
        try:
            data = request.get_json()
            if data.get('model'):
                logging.info("Carregando modelo e tokenizador..")
                
                model_name = data.get('model')
                model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, token=os.getenv('TOKEN_HF'))
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=os.getenv("TOKEN_HF"))
                clf_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer)
                
                logging.info("Modelo e tokenizador carregados com sucesso.")
                
                return {'message': 'success', 'status': 0}, 200
            return {'message': 'the model parameter is missing', 'status': 1}, 400
        except Exception as e:
            print(e)
            return {
                'message': 'failure! Expected a json',
                'status': 1
            }, 400 


    @route('/classify_complaints', methods=['POST'])
    def classify_sections(self):
        global model, tokenizer, clf_pipeline
        data = request.get_json()
        
        logging.info('Classificando reclamações...')
        
        if 'complaint' not in data:
            return {'message': 'Missing parameter: complaint', 'status': 1}, 400
        
        if not data['complaint']:
            return {'message': 'Empty parameter: complaint', 'status': 1}, 400
        
        try:
            complaint_text = data.get('complaint')
            if isinstance(complaint_text, str):
                cleaned_text = self.clean_text(complaint_text)
                classification = clf_pipeline(cleaned_text, max_length = 512, truncation = True)
                return {'message': 'success', 'status': 0, 'output': classification}, 200
            return {'message': 'data type mismatch, expected string', 'status': 1}, 400
        
        except:
            stacktrace = traceback.format_exc()
            return {
                'message': 'fail',
                'stacktrace': stacktrace,
                'status': 1
            }, 500


app = Flask(__name__)
ClassifySectionsService.register(app, route_base='/')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5009)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    print('Running Compiler Service...')
    print(f'Host: {args.host}')
    print(f'Port: {args.port}')
    print(f'Debug: {args.debug}')
    app.run(debug=args.debug, port=args.port, host=args.host)