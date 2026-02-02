import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Brain, Scale, Search, FileText, CheckCircle2, XCircle } from 'lucide-react';
import './App.css';

// Tab types
const TABS = {
  SEARCH: 'search',
  REASON: 'reason',
  PIPELINE: 'pipeline',
};

// API base URL - uses proxy in development
const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState(TABS.SEARCH);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Ciao! Sono LexCausa, il tuo assistente per ricerche legali nel Codice Civile e Penale italiano. Descrivimi il tuo caso e ti aiuterò a trovare gli articoli e i precedenti più rilevanti.',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [reasoningResult, setReasoningResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, reasoningResult, pipelineResult]);

  const handleSearchSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');

    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, top_k: 5 }),
      });

      if (!response.ok) throw new Error('Errore nella risposta del server');

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          metadata: {
            classification: data.classification,
            articles: data.articles,
            precedents: data.precedents,
          },
        },
      ]);
    } catch (error) {
      console.error('Errore:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Mi dispiace, si è verificato un errore. Riprova più tardi.',
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleReasonSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const claim = input.trim();
    setInput('');
    setIsLoading(true);
    setReasoningResult(null);

    try {
      const response = await fetch(`${API_BASE}/reason`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          claim,
          include_precedents: true,
          use_context: false,
        }),
      });

      if (!response.ok) throw new Error('Errore nella risposta del server');

      const data = await response.json();
      setReasoningResult(data);
    } catch (error) {
      console.error('Errore:', error);
      setReasoningResult({ error: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handlePipelineSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const claim = input.trim();
    setInput('');
    setIsLoading(true);
    setPipelineResult(null);

    try {
      // Chiama l'endpoint /api/pipeline che gestisce tutto il flusso nel backend
      const response = await fetch(`${API_BASE}/pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          claim,
          include_precedents: true,
        }),
      });

      if (!response.ok) throw new Error('Errore nella pipeline');
      const data = await response.json();

      // Il backend restituisce: { claim, reasoner: {...}, counter_reasoner: {...} }
      setPipelineResult(data);
    } catch (error) {
      console.error('Errore pipeline:', error);
      setPipelineResult({ error: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    switch (activeTab) {
      case TABS.SEARCH:
        handleSearchSubmit();
        break;
      case TABS.REASON:
        handleReasonSubmit();
        break;
      case TABS.PIPELINE:
        handlePipelineSubmit();
        break;
      default:
        handleSearchSubmit();
    }
  };

  const getPlaceholder = () => {
    switch (activeTab) {
      case TABS.SEARCH:
        return 'Descrivi il tuo caso legale...';
      case TABS.REASON:
        return 'Inserisci un claim da analizzare...';
      case TABS.PIPELINE:
        return 'Inserisci un claim per la pipeline completa...';
      default:
        return 'Scrivi un messaggio...';
    }
  };

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <div className="header-content">
          <div className="avatar-icon assistant-avatar">
            <Scale size={24} />
          </div>
          <div>
            <h1 className="header-title">LexCausa AI</h1>
            <p className="header-subtitle">Assistente Legale con Ragionamento Causale</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs-container">
        <button
          className={`tab-button ${activeTab === TABS.SEARCH ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.SEARCH)}
        >
          <Search size={16} />
          <span>Ricerca</span>
        </button>
        <button
          className={`tab-button ${activeTab === TABS.REASON ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.REASON)}
        >
          <Brain size={16} />
          <span>Ragionamento</span>
        </button>
        <button
          className={`tab-button ${activeTab === TABS.PIPELINE ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.PIPELINE)}
        >
          <FileText size={16} />
          <span>Pipeline Completa</span>
        </button>
      </div>

      {/* Content Area */}
      <div className="messages-area">
        <div className="messages-container">
          {activeTab === TABS.SEARCH && (
            <>
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`message ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}
                >
                  <div className={`message-avatar ${msg.role === 'user' ? 'user-avatar' : 'assistant-avatar'}`}>
                    {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                  </div>
                  <div className={`message-bubble ${msg.role === 'user' ? 'bubble-user' : 'bubble-assistant'}`}>
                    <p className="message-text">{msg.content}</p>
                  </div>
                </div>
              ))}
            </>
          )}

          {activeTab === TABS.REASON && reasoningResult && (
            <div className="result-card">
              <h3 className="result-title">
                <Brain size={20} />
                Risultato Ragionamento
              </h3>

              {reasoningResult.error ? (
                <p className="error-text">Errore: {reasoningResult.error}</p>
              ) : (
                <>
                  <div className="result-section">
                    <h4>Claim Analizzato</h4>
                    <p>{reasoningResult.claim}</p>
                  </div>

                  {reasoningResult.causality && (
                    <div className="result-section">
                      <h4>Classificazione Causalità</h4>
                      <pre className="code-block">
                        {JSON.stringify(reasoningResult.causality, null, 2)}
                      </pre>
                    </div>
                  )}

                  {reasoningResult.raw_response && (
                    <div className="result-section">
                      <h4>Risposta Completa</h4>
                      <div className="raw-response">{reasoningResult.raw_response}</div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {activeTab === TABS.PIPELINE && pipelineResult && (
            <div className="result-card">
              <h3 className="result-title">
                <FileText size={20} />
                Risultato Pipeline Completa
              </h3>

              {pipelineResult.error ? (
                <p className="error-text">Errore: {pipelineResult.error}</p>
              ) : (
                <>
                  <div className="result-section">
                    <h4>Claim Analizzato</h4>
                    <p>{pipelineResult.claim}</p>
                  </div>

                  {/* SEZIONE REASONER */}
                  <div className="result-section pipeline-section">
                    <h3 className="section-header">
                      <CheckCircle2 size={20} style={{ color: '#10b981' }} />
                      1. REASONER - Argomenti a Favore
                    </h3>

                    {pipelineResult.reasoner?.causality && (
                      <div className="subsection">
                        <h4>Classificazione Causalità</h4>
                        <pre className="code-block">
                          {JSON.stringify(pipelineResult.reasoner.causality, null, 2)}
                        </pre>
                      </div>
                    )}

                    {pipelineResult.reasoner?.statutes && pipelineResult.reasoner.statutes.length > 0 && (
                      <div className="subsection">
                        <h4>Articoli Trovati ({pipelineResult.reasoner.statutes.length})</h4>
                        <ul className="articles-list">
                          {pipelineResult.reasoner.statutes.map((art, idx) => (
                            <li key={idx}>
                              <strong>Art. {art.articolo || art.statute_id}</strong>
                              {art.source && ` (${art.source === 'codice_civile' ? 'c.c.' : 'c.p.'})`}
                              {art.titolo && ` - ${art.titolo}`}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {pipelineResult.reasoner?.precedents && pipelineResult.reasoner.precedents.length > 0 && (
                      <div className="subsection">
                        <h4>Precedenti Trovati ({pipelineResult.reasoner.precedents.length})</h4>
                        <ul className="precedents-list">
                          {pipelineResult.reasoner.precedents.map((prec, idx) => (
                            <li key={idx}>
                              <strong>{prec.title || `Precedente ${idx + 1}`}</strong>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {pipelineResult.reasoner?.raw_response && (
                      <div className="subsection">
                        <h4>Risposta Completa</h4>
                        <div className="raw-response">{pipelineResult.reasoner.raw_response}</div>
                      </div>
                    )}
                  </div>

                  {/* SEZIONE COUNTER-REASONER */}
                  <div className="result-section pipeline-section">
                    <h3 className="section-header">
                      <XCircle size={20} style={{ color: '#ef4444' }} />
                      2. COUNTER-REASONER - Argomenti Contrari
                    </h3>

                    {pipelineResult.counter_reasoner?.reasoner_causality && (
                      <div className="subsection">
                        <h4>Causalità del Reasoner (da Attaccare)</h4>
                        <pre className="code-block">
                          {JSON.stringify(pipelineResult.counter_reasoner.reasoner_causality, null, 2)}
                        </pre>
                      </div>
                    )}

                    {pipelineResult.counter_reasoner?.warrant_info && (
                      <div className="subsection">
                        <h4>Warrant e Causalità Attaccanti</h4>
                        <pre className="code-block">
                          {JSON.stringify(pipelineResult.counter_reasoner.warrant_info, null, 2)}
                        </pre>
                      </div>
                    )}

                    {pipelineResult.counter_reasoner?.statutes && pipelineResult.counter_reasoner.statutes.length > 0 && (
                      <div className="subsection">
                        <h4>Articoli Trovati (Contro-Tesi) ({pipelineResult.counter_reasoner.statutes.length})</h4>
                        <ul className="articles-list">
                          {pipelineResult.counter_reasoner.statutes.map((art, idx) => (
                            <li key={idx}>
                              <strong>Art. {art.articolo || art.statute_id}</strong>
                              {art.source && ` (${art.source === 'codice_civile' ? 'c.c.' : 'c.p.'})`}
                              {art.titolo && ` - ${art.titolo}`}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {pipelineResult.counter_reasoner?.precedents && pipelineResult.counter_reasoner.precedents.length > 0 && (
                      <div className="subsection">
                        <h4>Precedenti Trovati (Contro-Tesi) ({pipelineResult.counter_reasoner.precedents.length})</h4>
                        <ul className="precedents-list">
                          {pipelineResult.counter_reasoner.precedents.map((prec, idx) => (
                            <li key={idx}>
                              <strong>{prec.title || `Precedente ${idx + 1}`}</strong>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {pipelineResult.counter_reasoner?.raw_response && (
                      <div className="subsection">
                        <h4>Risposta Completa</h4>
                        <div className="raw-response">{pipelineResult.counter_reasoner.raw_response}</div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === TABS.REASON && !reasoningResult && !isLoading && (
            <div className="empty-state">
              <Brain size={48} className="empty-icon" />
              <p>Inserisci un claim legale per analizzare la catena causale</p>
            </div>
          )}

          {activeTab === TABS.PIPELINE && !pipelineResult && !isLoading && (
            <div className="empty-state">
              <FileText size={48} className="empty-icon" />
              <p>Inserisci un claim per eseguire la pipeline completa: Reasoner → Counter-Reasoner</p>
            </div>
          )}

          {isLoading && (
            <div className="message message-assistant">
              <div className="message-avatar assistant-avatar">
                <Loader2 size={20} className="loading-spinner" />
              </div>
              <div className="message-bubble bubble-assistant">
                <p>
                  {activeTab === TABS.PIPELINE
                    ? 'Esecuzione pipeline completa (Reasoner + Counter-Reasoner)...'
                    : 'Elaborazione in corso...'}
                </p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="input-area">
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder={getPlaceholder()}
              rows={1}
              className="input-textarea"
              disabled={isLoading}
            />
          </div>
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="send-button"
          >
            <Send size={20} />
          </button>
        </div>
        <p className="input-hint">
          Premi Invio per inviare, Shift+Invio per andare a capo
        </p>
      </div>
    </div>
  );
}