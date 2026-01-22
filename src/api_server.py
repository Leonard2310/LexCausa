#!/usr/bin/env python3
"""
LexCausa - Flask API Server (per frontend)

REST API server da aggiungere al progetto esistente.
Espone gli endpoint necessari per il frontend React.

Salva questo file come: src/api_server.py
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Setup paths (mantieni la stessa logica del main.py)
src_path = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(src_path) == "src":
    project_root = os.path.dirname(src_path)
else:
    project_root = src_path
    src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
os.chdir(project_root)

load_dotenv()

from services.legal_search import LegalSearchPipeline
from services.claim_classifier import ClaimClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Pipeline globale (lazy initialization)
pipeline = None
classifier = None


def get_pipeline():
    """Lazy load della pipeline."""
    global pipeline
    if pipeline is None:
        print("🔧 Inizializzazione pipeline...")
        pipeline = LegalSearchPipeline()
        print("✅ Pipeline pronta!")
    return pipeline


def get_classifier():
    """Lazy load del classificatore."""
    global classifier
    if classifier is None:
        print("🔧 Inizializzazione classificatore...")
        classifier = ClaimClassifier()
        print("✅ Classificatore pronto!")
    return classifier


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'LexCausa API',
        'version': '0.1.0'
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint principale per il chatbot.
    
    Request: {"message": "claim legale", "top_k": 5}
    Response: {"response": "testo formattato", ...}
    """
    try:
        pipe = get_pipeline()
        
        data = request.get_json()
        claim = data.get('message', '').strip()
        top_k = data.get('top_k', 5)
        
        if not claim:
            return jsonify({'error': 'Campo "message" obbligatorio'}), 400
        
        # Esegui ricerca
        result = pipe.search(claim, top_k=top_k)
        
        # Formatta risposta
        response_text = format_search_result(result)
        
        return jsonify({
            'response': response_text,
            'classification': {
                'categories': result.classification.categories,
                'descriptions': result.classification.descriptions,
                'libro_mappings': result.classification.libro_mappings
            },
            'articles': [
                {
                    'source': art.source,
                    'articolo': art.articolo,
                    'titolo': art.titolo,
                    'testo': art.testo,
                    'libro': art.libro,
                    'score': art.score
                }
                for art in result.articles
            ],
            'precedents': [
                {
                    'materia': prec.materia,
                    'estremi': prec.estremi,
                    'massima': prec.massima,
                    'score': prec.score
                }
                for prec in result.precedents
            ]
        })
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def format_search_result(result) -> str:
    """Formatta il risultato in markdown."""
    lines = []
    
    # Classificazione
    lines.append("## 📋 Classificazione\n")
    for i, (cat, desc) in enumerate(
        zip(result.classification.categories, result.classification.descriptions), 1
    ):
        source, libro = result.classification.libro_mappings[i - 1]
        source_label = "Codice Civile" if "civile" in source else "Codice Penale"
        lines.append(f"**{i}. {cat}**")
        lines.append(f"- {desc}")
        lines.append(f"- 📚 [{source_label}] {libro}\n")
    
    # Articoli
    if result.articles:
        lines.append("\n---\n")
        lines.append("## 📖 Articoli Rilevanti\n")
        for i, art in enumerate(result.articles, 1):
            src = "CC" if "civile" in art.source else "CP"
            lines.append(f"### {i}. [{src}] Art. {art.articolo} - {art.titolo}")
            lines.append(f"**Rilevanza:** {art.score:.1%}")
            lines.append(f"**Libro:** {art.libro}\n")
            preview = art.testo[:300] + "..." if len(art.testo) > 300 else art.testo
            lines.append(f"{preview}\n")
    
    # Precedenti
    if result.precedents:
        lines.append("\n---\n")
        lines.append("## ⚖️ Precedenti Giurisprudenziali\n")
        for i, prec in enumerate(result.precedents[:3], 1):
            lines.append(f"### {i}. {prec.estremi}")
            lines.append(f"**Rilevanza:** {prec.score:.1%}")
            lines.append(f"**Materia:** {prec.materia}\n")
            preview = prec.massima[:300] + "..." if len(prec.massima) > 300 else prec.massima
            lines.append(f"{preview}\n")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  🚀 LexCausa API Server")
    print("=" * 70)
    print()
    print("  Server in ascolto su: http://localhost:8000")
    print()
    print("  Endpoints:")
    print("  • GET  /health       - Health check")
    print("  • POST /api/chat     - Ricerca legale")
    print()
    print("=" * 70)
    print()
    
    # Usa porta 8000 per non confliggere con Neo4j su 7474
    port = int(os.getenv("API_PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)