import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Brain, Scale, Search, FileText, CheckCircle2, XCircle, AlertTriangle, ClipboardCheck, Wrench, Settings, GitBranch, Swords } from 'lucide-react';
import './App.css';
import AspicMetagraph from './AspicMetagraph';
import AttackTextDetails from './AttackTextDetails';

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
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [pipelineSettings, setPipelineSettings] = useState({
    reasoner_model: '',
    counter_model: '',
    llm_temperature: 0.3,
    llm_max_tokens: 8192,
    search_top_k_default: 100,
    search_use_top_n_libri: 3,
    precedents_limit_default: 5,
    aqa_alpha: 0.3,
    aqa_beta: 0.4,
    aqa_gamma: 0.3,
    aqa_min_semantic_overlap: 0.5,
    aqa_min_strength_ratio: 1.2,
    aqa_damage_factor: 0.5,
    aqa_allow_factual_attacks: true,
    aqa_allow_cross_codice: true,
  });
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, reasoningResult, pipelineResult]);

  // Load default settings from backend on mount
  useEffect(() => {
    fetch(`${API_BASE}/settings`)
      .then((res) => res.json())
      .then((data) => {
        setAvailableModels(data.models || []);
        const d = data.defaults || {};
        setPipelineSettings((prev) => ({
          ...prev,
          reasoner_model: d.groq_model || prev.reasoner_model,
          counter_model: d.groq_model || prev.counter_model,
          llm_temperature: d.llm_temperature ?? prev.llm_temperature,
          llm_max_tokens: d.llm_max_tokens ?? prev.llm_max_tokens,
          search_top_k_default: d.search_top_k_default ?? prev.search_top_k_default,
          search_use_top_n_libri: d.search_use_top_n_libri ?? prev.search_use_top_n_libri,
          precedents_limit_default: d.precedents_limit_default ?? prev.precedents_limit_default,
          aqa_alpha: d.aqa_alpha ?? prev.aqa_alpha,
          aqa_beta: d.aqa_beta ?? prev.aqa_beta,
          aqa_gamma: d.aqa_gamma ?? prev.aqa_gamma,
          aqa_min_semantic_overlap: d.aqa_min_semantic_overlap ?? prev.aqa_min_semantic_overlap,
          aqa_min_strength_ratio: d.aqa_min_strength_ratio ?? prev.aqa_min_strength_ratio,
          aqa_damage_factor: d.aqa_damage_factor ?? prev.aqa_damage_factor,
          aqa_allow_factual_attacks: d.aqa_allow_factual_attacks ?? prev.aqa_allow_factual_attacks,
          aqa_allow_cross_codice: d.aqa_allow_cross_codice ?? prev.aqa_allow_cross_codice,
        }));
      })
      .catch((err) => console.warn('Could not load settings:', err));
  }, []);

  const updateSetting = (key, value) => {
    setPipelineSettings((prev) => ({ ...prev, [key]: value }));
  };

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
          max_statutes: pipelineSettings.search_top_k_default,
          max_precedents: pipelineSettings.precedents_limit_default,
          settings: {
            reasoner_model: pipelineSettings.reasoner_model,
            counter_model: pipelineSettings.counter_model,
            llm_temperature: pipelineSettings.llm_temperature,
            llm_max_tokens: pipelineSettings.llm_max_tokens,
            search_use_top_n_libri: pipelineSettings.search_use_top_n_libri,
            aqa_alpha: pipelineSettings.aqa_alpha,
            aqa_beta: pipelineSettings.aqa_beta,
            aqa_gamma: pipelineSettings.aqa_gamma,
            aqa_min_semantic_overlap: pipelineSettings.aqa_min_semantic_overlap,
            aqa_min_strength_ratio: pipelineSettings.aqa_min_strength_ratio,
            aqa_damage_factor: pipelineSettings.aqa_damage_factor,
            aqa_allow_factual_attacks: pipelineSettings.aqa_allow_factual_attacks,
            aqa_allow_cross_codice: pipelineSettings.aqa_allow_cross_codice,
          },
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

  const aqaReport = pipelineResult?.evaluation?.aqa_report;
  const aqaProScore = aqaReport?.net_plausibility?.pro ?? 0;
  const aqaContraScore = aqaReport?.net_plausibility?.contra ?? 0;
  const aqaFinalScore = aqaReport?.net_plausibility?.final ?? 0;
  const aqaProLinks = aqaReport?.links?.pro ?? [];
  const aqaContraLinks = aqaReport?.links?.contra ?? [];
  const aqaVerdict = aqaReport?.verdict ?? 'uncertain';
  const aqaVerdictLabel = aqaVerdict === 'plausible'
    ? 'Plausibile'
    : aqaVerdict === 'implausible'
      ? 'Implausibile'
      : 'Incerto';
  const aqaVerdictClass = aqaVerdict === 'plausible'
    ? 'aqa-verdict-positive'
    : aqaVerdict === 'implausible'
      ? 'aqa-verdict-negative'
      : 'aqa-verdict-uncertain';

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

      {/* Settings Panel (collapsible) */}
      <div className="settings-panel">
        <button className="settings-toggle" onClick={() => setSettingsOpen(!settingsOpen)}>
          <Settings size={16} />
          <span>Impostazioni Pipeline</span>
          <span className={`settings-chevron ${settingsOpen ? 'open' : ''}`}>▸</span>
        </button>

        {settingsOpen && (
          <div className="settings-body">
            {/* Models per step */}
            <fieldset className="settings-group">
              <legend>🤖 Modelli LLM per Step</legend>
              <div className="settings-row">
                <label>
                  <span>Reasoner</span>
                  <select
                    value={pipelineSettings.reasoner_model}
                    onChange={(e) => updateSetting('reasoner_model', e.target.value)}
                  >
                    {availableModels.map((m) => (
                      <option key={m} value={m}>{m.split('/').pop()}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Counter-Reasoner</span>
                  <select
                    value={pipelineSettings.counter_model}
                    onChange={(e) => updateSetting('counter_model', e.target.value)}
                  >
                    {availableModels.map((m) => (
                      <option key={m} value={m}>{m.split('/').pop()}</option>
                    ))}
                  </select>
                </label>
              </div>
            </fieldset>

            {/* Temperature & Max Tokens */}
            <fieldset className="settings-group">
              <legend>🎛️ Parametri LLM</legend>
              <div className="settings-row">
                <label>
                  <span>Temperature <strong>{pipelineSettings.llm_temperature.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pipelineSettings.llm_temperature}
                    onChange={(e) => updateSetting('llm_temperature', parseFloat(e.target.value))}
                  />
                </label>
                <label>
                  <span>Max Tokens</span>
                  <input
                    type="number"
                    min="256"
                    max="32768"
                    step="256"
                    value={pipelineSettings.llm_max_tokens}
                    onChange={(e) => updateSetting('llm_max_tokens', parseInt(e.target.value, 10))}
                  />
                </label>
              </div>
            </fieldset>

            {/* Search & Retrieval */}
            <fieldset className="settings-group">
              <legend>🔍 Ricerca e Retrieval</legend>
              <div className="settings-row">
                <label>
                  <span>Top-K Articoli</span>
                  <input
                    type="number"
                    min="10"
                    max="500"
                    step="10"
                    value={pipelineSettings.search_top_k_default}
                    onChange={(e) => updateSetting('search_top_k_default', parseInt(e.target.value, 10))}
                  />
                </label>
                <label>
                  <span>Top-N Libri</span>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={pipelineSettings.search_use_top_n_libri}
                    onChange={(e) => updateSetting('search_use_top_n_libri', parseInt(e.target.value, 10))}
                  />
                </label>
                <label>
                  <span>Max Precedenti</span>
                  <input
                    type="number"
                    min="0"
                    max="20"
                    value={pipelineSettings.precedents_limit_default}
                    onChange={(e) => updateSetting('precedents_limit_default', parseInt(e.target.value, 10))}
                  />
                </label>
              </div>
            </fieldset>

            {/* AQA Weights */}
            <fieldset className="settings-group">
              <legend>⚖️ Pesi AQA (α + β + γ = 1)</legend>
              <div className="settings-row">
                <label>
                  <span>α Cogency <strong>{pipelineSettings.aqa_alpha.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pipelineSettings.aqa_alpha}
                    onChange={(e) => updateSetting('aqa_alpha', parseFloat(e.target.value))}
                  />
                </label>
                <label>
                  <span>β NormSupport <strong>{pipelineSettings.aqa_beta.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pipelineSettings.aqa_beta}
                    onChange={(e) => updateSetting('aqa_beta', parseFloat(e.target.value))}
                  />
                </label>
                <label>
                  <span>γ Semantics <strong>{pipelineSettings.aqa_gamma.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pipelineSettings.aqa_gamma}
                    onChange={(e) => updateSetting('aqa_gamma', parseFloat(e.target.value))}
                  />
                </label>
              </div>
              <div className="aqa-weight-sum">
                Somma: {(pipelineSettings.aqa_alpha + pipelineSettings.aqa_beta + pipelineSettings.aqa_gamma).toFixed(2)}
                {Math.abs(pipelineSettings.aqa_alpha + pipelineSettings.aqa_beta + pipelineSettings.aqa_gamma - 1) > 0.01 && (
                  <span className="aqa-weight-warning"> ⚠️ Dovrebbe essere 1.00</span>
                )}
              </div>
            </fieldset>

            {/* AQA Cross-Attack Parameters */}
            <fieldset className="settings-group">
              <legend>⚔️ Parametri Cross-Attack</legend>
              <div className="settings-row">
                <label>
                  <span>Min Overlap Semantico <strong>{pipelineSettings.aqa_min_semantic_overlap.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pipelineSettings.aqa_min_semantic_overlap}
                    onChange={(e) => updateSetting('aqa_min_semantic_overlap', parseFloat(e.target.value))}
                  />
                </label>
                <label>
                  <span>Min Strength Ratio <strong>{pipelineSettings.aqa_min_strength_ratio.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0.5"
                    max="2.0"
                    step="0.05"
                    value={pipelineSettings.aqa_min_strength_ratio}
                    onChange={(e) => updateSetting('aqa_min_strength_ratio', parseFloat(e.target.value))}
                  />
                </label>
                <label>
                  <span>Damage Factor <strong>{pipelineSettings.aqa_damage_factor.toFixed(2)}</strong></span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pipelineSettings.aqa_damage_factor}
                    onChange={(e) => updateSetting('aqa_damage_factor', parseFloat(e.target.value))}
                  />
                </label>
              </div>
              <div className="settings-row">
                <label className="settings-toggle-label">
                  <input
                    type="checkbox"
                    checked={pipelineSettings.aqa_allow_factual_attacks}
                    onChange={(e) => updateSetting('aqa_allow_factual_attacks', e.target.checked)}
                  />
                  <span>Attacchi Fattuali su Norme</span>
                </label>
                <label className="settings-toggle-label">
                  <input
                    type="checkbox"
                    checked={pipelineSettings.aqa_allow_cross_codice}
                    onChange={(e) => updateSetting('aqa_allow_cross_codice', e.target.checked)}
                  />
                  <span>Cross-Codice (doppia rilevanza)</span>
                </label>
              </div>
            </fieldset>
          </div>
        )}
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

                  {reasoningResult.aspic_ir && (
                    <div className="result-section">
                      <details className="ir-toggle">
                        <summary>ASPIC+ IR</summary>
                        <pre className="code-block">
                          {JSON.stringify(reasoningResult.aspic_ir, null, 2)}
                        </pre>
                      </details>
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

                    {pipelineResult.reasoner?.aspic_ir && (
                      <div className="subsection">
                        <details className="ir-toggle">
                          <summary>ASPIC+ IR (Reasoner)</summary>
                          <pre className="code-block">
                            {JSON.stringify(pipelineResult.reasoner.aspic_ir, null, 2)}
                          </pre>
                        </details>
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

                    {pipelineResult.counter_reasoner?.aspic_ir && (
                      <div className="subsection">
                        <details className="ir-toggle">
                          <summary>ASPIC+ IR (Counter-Reasoner)</summary>
                          <pre className="code-block">
                            {JSON.stringify(pipelineResult.counter_reasoner.aspic_ir, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}

                    {pipelineResult.counter_reasoner?.raw_response && (
                      <div className="subsection">
                        <h4>Risposta Completa</h4>
                        <div className="raw-response">{pipelineResult.counter_reasoner.raw_response}</div>
                      </div>
                    )}
                  </div>

                  {/* SEZIONE EVALUATOR - Verifica Consistenza */}
                  {pipelineResult.evaluation?.consistency_report && (
                    <div className="result-section pipeline-section">
                      <h3 className="section-header">
                        <ClipboardCheck size={20} style={{ color: '#8b5cf6' }} />
                        3. EVALUATOR - Verifica Consistenza
                      </h3>

                      {/* Reasoner Consistency */}
                      {pipelineResult.evaluation.consistency_report.reasoner && (
                        <div className="subsection">
                          <h4>
                            Reasoner - Score: {(pipelineResult.evaluation.consistency_report.reasoner.consistency_score * 100).toFixed(0)}%
                          </h4>
                          <div className="consistency-stats">
                            <span className="stat-item stat-valid">
                              ✅ Valide: {pipelineResult.evaluation.consistency_report.reasoner.valid_citations}/{pipelineResult.evaluation.consistency_report.reasoner.total_citations}
                            </span>
                            <span className="stat-item stat-text">
                              📝 Testo match: {pipelineResult.evaluation.consistency_report.reasoner.text_matches}/{pipelineResult.evaluation.consistency_report.reasoner.text_matches + pipelineResult.evaluation.consistency_report.reasoner.text_mismatches}
                            </span>
                          </div>

                          {pipelineResult.evaluation.consistency_report.reasoner.citation_checks?.length > 0 && (
                            <div className="citation-checks-list">
                              {pipelineResult.evaluation.consistency_report.reasoner.citation_checks.map((check, idx) => (
                                <div key={idx} className={`citation-check-item ${check.found_in_kb ? 'check-valid' : 'check-invalid'}`}>
                                  <div className="check-header">
                                    {check.found_in_kb ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                                    <strong>{check.citation}</strong>
                                    {check.text_verified && (
                                      <span className={`text-badge ${check.llm_validated ? 'badge-llm-validated' : check.text_match ? 'badge-match' : 'badge-mismatch'}`}>
                                        {check.llm_validated
                                          ? `🤖 Validato LLM (sim. ${(check.text_similarity * 100).toFixed(0)}%)`
                                          : check.text_match ? '✅ Testo OK' : '⚠️ Testo diverso'}
                                        {!check.llm_validated && check.text_similarity > 0 && ` (${(check.text_similarity * 100).toFixed(0)}%)`}
                                      </span>
                                    )}
                                  </div>
                                  <div className="check-details">{check.details}</div>

                                  {check.text_verified && (check.cited_text || check.db_text_preview) && (
                                    <details className="text-comparison">
                                      <summary>Confronta Testi</summary>
                                      <div className="text-comparison-content">
                                        {check.cited_text && (
                                          <div className="text-block cited-text">
                                            <h5>📖 Testo Citato nella Catena:</h5>
                                            <p>{check.cited_text}</p>
                                          </div>
                                        )}
                                        {check.db_text_preview && (
                                          <div className="text-block db-text">
                                            <h5>🗄️ Testo nel Database:</h5>
                                            <p>{check.db_text_preview}</p>
                                          </div>
                                        )}
                                      </div>
                                    </details>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                          {pipelineResult.evaluation.consistency_report.reasoner.issues?.length > 0 && (
                            <div className="issues-list">
                              <h5><AlertTriangle size={14} /> Problemi Rilevati:</h5>
                              <ul>
                                {pipelineResult.evaluation.consistency_report.reasoner.issues.map((issue, idx) => (
                                  <li key={idx}>{issue}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Counter-Reasoner Consistency */}
                      {pipelineResult.evaluation.consistency_report.counter_reasoner && (
                        <div className="subsection">
                          <h4>
                            Counter-Reasoner - Score: {(pipelineResult.evaluation.consistency_report.counter_reasoner.consistency_score * 100).toFixed(0)}%
                          </h4>
                          <div className="consistency-stats">
                            <span className="stat-item stat-valid">
                              ✅ Valide: {pipelineResult.evaluation.consistency_report.counter_reasoner.valid_citations}/{pipelineResult.evaluation.consistency_report.counter_reasoner.total_citations}
                            </span>
                            <span className="stat-item stat-text">
                              📝 Testo match: {pipelineResult.evaluation.consistency_report.counter_reasoner.text_matches}/{pipelineResult.evaluation.consistency_report.counter_reasoner.text_matches + pipelineResult.evaluation.consistency_report.counter_reasoner.text_mismatches}
                            </span>
                          </div>

                          {pipelineResult.evaluation.consistency_report.counter_reasoner.citation_checks?.length > 0 && (
                            <div className="citation-checks-list">
                              {pipelineResult.evaluation.consistency_report.counter_reasoner.citation_checks.map((check, idx) => (
                                <div key={idx} className={`citation-check-item ${check.found_in_kb ? 'check-valid' : 'check-invalid'}`}>
                                  <div className="check-header">
                                    {check.found_in_kb ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                                    <strong>{check.citation}</strong>
                                    {check.text_verified && (
                                      <span className={`text-badge ${check.llm_validated ? 'badge-llm-validated' : check.text_match ? 'badge-match' : 'badge-mismatch'}`}>
                                        {check.llm_validated
                                          ? `🤖 Validato LLM (sim. ${(check.text_similarity * 100).toFixed(0)}%)`
                                          : check.text_match ? '✅ Testo OK' : '⚠️ Testo diverso'}
                                        {!check.llm_validated && check.text_similarity > 0 && ` (${(check.text_similarity * 100).toFixed(0)}%)`}
                                      </span>
                                    )}
                                  </div>
                                  <div className="check-details">{check.details}</div>

                                  {check.text_verified && (check.cited_text || check.db_text_preview) && (
                                    <details className="text-comparison">
                                      <summary>Confronta Testi</summary>
                                      <div className="text-comparison-content">
                                        {check.cited_text && (
                                          <div className="text-block cited-text">
                                            <h5>📖 Testo Citato nella Catena:</h5>
                                            <p>{check.cited_text}</p>
                                          </div>
                                        )}
                                        {check.db_text_preview && (
                                          <div className="text-block db-text">
                                            <h5>🗄️ Testo nel Database:</h5>
                                            <p>{check.db_text_preview}</p>
                                          </div>
                                        )}
                                      </div>
                                    </details>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                          {pipelineResult.evaluation.consistency_report.counter_reasoner.issues?.length > 0 && (
                            <div className="issues-list">
                              <h5><AlertTriangle size={14} /> Problemi Rilevati:</h5>
                              <ul>
                                {pipelineResult.evaluation.consistency_report.counter_reasoner.issues.map((issue, idx) => (
                                  <li key={idx}>{issue}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Summary */}
                      {pipelineResult.evaluation.summary && (
                        <div className="subsection">
                          <h4>Riepilogo</h4>
                          <div className="raw-response">{pipelineResult.evaluation.summary}</div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* SEZIONE CATENE RIPARATE */}
                  {pipelineResult.evaluation && (pipelineResult.evaluation.repaired_reasoner_chain || pipelineResult.evaluation.repaired_counter_chain) && (
                    <div className="result-section pipeline-section">
                      <h3 className="section-header">
                        <Wrench size={20} style={{ color: '#f59e0b' }} />
                        4. CATENE DI RAGIONAMENTO RIPARATE
                      </h3>

                      {/* Repaired Reasoner Chain */}
                      {pipelineResult.evaluation.repaired_reasoner_chain && (
                        <div className="subsection">
                          <h4>
                            <CheckCircle2 size={16} style={{ color: '#10b981' }} />
                            Reasoner - Catena Riparata
                          </h4>
                          <div className="raw-response repaired-chain">
                            {pipelineResult.evaluation.repaired_reasoner_chain}
                          </div>

                          {/* Show repaired ASPIC IR if available */}
                          {pipelineResult.evaluation.repaired_reasoner_aspic_ir && Object.keys(pipelineResult.evaluation.repaired_reasoner_aspic_ir).length > 0 && (
                            <details className="ir-toggle">
                              <summary>ASPIC+ IR Riparato (Reasoner)</summary>
                              <pre className="code-block">
                                {JSON.stringify(pipelineResult.evaluation.repaired_reasoner_aspic_ir, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      )}

                      {/* Repaired Counter-Reasoner Chain */}
                      {pipelineResult.evaluation.repaired_counter_chain && (
                        <div className="subsection">
                          <h4>
                            <XCircle size={16} style={{ color: '#ef4444' }} />
                            Counter-Reasoner - Catena Riparata
                          </h4>
                          <div className="raw-response repaired-chain">
                            {pipelineResult.evaluation.repaired_counter_chain}
                          </div>

                          {/* Show repaired ASPIC IR if available */}
                          {pipelineResult.evaluation.repaired_counter_aspic_ir && Object.keys(pipelineResult.evaluation.repaired_counter_aspic_ir).length > 0 && (
                            <details className="ir-toggle">
                              <summary>ASPIC+ IR Riparato (Counter-Reasoner)</summary>
                              <pre className="code-block">
                                {JSON.stringify(pipelineResult.evaluation.repaired_counter_aspic_ir, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      )}

                      {/* Repair Statistics */}
                      {pipelineResult.evaluation.consistency_report && (
                        <div className="subsection repair-stats">
                          <h4>Statistiche Riparazione</h4>
                          <div className="consistency-stats">
                            {pipelineResult.evaluation.consistency_report.reasoner && (
                              <>
                                <span className="stat-item stat-repaired">
                                  🔧 Reasoner: {pipelineResult.evaluation.consistency_report.reasoner.repaired_citations || 0} riparate, {pipelineResult.evaluation.consistency_report.reasoner.dropped_citations || 0} scartate
                                </span>
                              </>
                            )}
                            {pipelineResult.evaluation.consistency_report.counter_reasoner && (
                              <>
                                <span className="stat-item stat-repaired">
                                  🔧 Counter: {pipelineResult.evaluation.consistency_report.counter_reasoner.repaired_citations || 0} riparate, {pipelineResult.evaluation.consistency_report.counter_reasoner.dropped_citations || 0} scartate
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* SEZIONE AQA - Valutazione Argomentativa (sulle catene riparate) */}
                  {aqaReport && (
                    <div className="result-section pipeline-section">
                      <h3 className="section-header">
                        <Scale size={20} style={{ color: '#06b6d4' }} />
                        5. AQA - Valutazione Argomentativa
                      </h3>
                      <p className="aqa-note">Valutazione effettuata sulle catene di ragionamento riparate</p>
                      {aqaReport.enabled ? (
                        <>
                          <div className="aqa-stats">
                            <span className={`aqa-badge ${aqaVerdictClass}`}>
                              Verdetto: {aqaVerdictLabel}
                            </span>
                            <span className="stat-item stat-valid">
                              Pro: {(aqaProScore * 100).toFixed(0)}%
                            </span>
                            <span className="stat-item stat-text">
                              Contro: {(aqaContraScore * 100).toFixed(0)}%
                            </span>
                            <span className="stat-item stat-repaired">
                              Finale: {(aqaFinalScore * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="aqa-meta">
                            <span>Link pro: {aqaProLinks.length}</span>
                            <span>Link contro: {aqaContraLinks.length}</span>
                            {aqaReport.weights && (
                              <span>
                                Pesi: α {aqaReport.weights.alpha.toFixed(2)}, β {aqaReport.weights.beta.toFixed(2)}, γ {aqaReport.weights.gamma.toFixed(2)}
                              </span>
                            )}
                            {aqaReport.notes?.attacks_enabled === false && (
                              <span>Attacchi: disabilitati</span>
                            )}
                            {aqaReport.notes?.attacks_enabled === true && (
                              <span style={{ color: '#f59e0b' }}>⚔️ Attacchi: abilitati</span>
                            )}
                          </div>

                          {/* Chain-level averages */}
                          {aqaReport.chain_scores && (
                            <div className="aqa-meta" style={{ marginTop: '0.5rem' }}>
                              <span className="stat-item stat-valid">
                                📚 Norm Pro: {((aqaReport.chain_scores.pro?.norm_support_avg ?? 0) * 100).toFixed(0)}%
                              </span>
                              <span className="stat-item stat-text">
                                📚 Norm Contro: {((aqaReport.chain_scores.contra?.norm_support_avg ?? 0) * 100).toFixed(0)}%
                              </span>
                              {aqaReport.chain_scores.pro?.attacks_avg > 0 && (
                                <span className="stat-item stat-repaired">
                                  ⚔️ Attacchi Pro: {((aqaReport.chain_scores.pro?.attacks_avg ?? 0)).toFixed(3)}
                                </span>
                              )}
                              {aqaReport.chain_scores.contra?.attacks_avg > 0 && (
                                <span className="stat-item stat-repaired">
                                  ⚔️ Attacchi Contro: {((aqaReport.chain_scores.contra?.attacks_avg ?? 0)).toFixed(3)}
                                </span>
                              )}
                            </div>
                          )}

                          {aqaReport.notes?.weakest_links?.length > 0 && (
                            <div className="aqa-notes">
                              <h5>Link più deboli</h5>
                              <ul className="aqa-list">
                                {aqaReport.notes.weakest_links.map((link, idx) => (
                                  <li key={idx}>
                                    {link.link_id || `Link ${idx + 1}`} - nesso {(link.nesso_plausibility ?? 0).toFixed(2)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {aqaReport.notes?.dominant_attacks?.length > 0 && (
                            <div className="aqa-notes">
                              <h5>Attacchi dominanti</h5>
                              <ul className="aqa-list">
                                {aqaReport.notes.dominant_attacks.map((attack, idx) => {
                                  const attackerLabel = attack.attacker_role === 'contra' ? 'C' : 'P';
                                  const targetLabel = attack.target_role === 'counter' ? 'C' : 'P';
                                  return (
                                    <li key={idx}>
                                      <span className={`role-tag ${attack.attacker_role === 'contra' ? 'role-contra' : 'role-pro'}`}>{attackerLabel}</span>
                                      {attack.attacker}
                                      {' → '}
                                      <span className={`role-tag ${attack.target_role === 'counter' ? 'role-contra' : 'role-pro'}`}>{targetLabel}</span>
                                      {attack.target}
                                      {' — val '}{(attack.value ?? 0).toFixed(3)} (overlap {((attack.overlap ?? 0) * 100).toFixed(0)}%)
                                    </li>
                                  );
                                })}
                              </ul>
                            </div>
                          )}

                          {aqaReport.notes?.precedent_swings?.length > 0 && (
                            <div className="aqa-notes">
                              <h5>Impatto precedenti</h5>
                              <ul className="aqa-list">
                                {aqaReport.notes.precedent_swings.map((item, idx) => (
                                  <li key={idx}>
                                    {item.link_id || `Link ${idx + 1}`} - Δ {(item.delta ?? 0).toFixed(2)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {(aqaProLinks.length > 0 || aqaContraLinks.length > 0) && (
                            <div className="aqa-link-breakdown">
                              <h5>Dettaglio valutazioni per link</h5>
                              {aqaProLinks.length > 0 && (
                                <div className="aqa-link-group">
                                  <h6>Reasoner (Pro)</h6>
                                  <div className="aqa-link-list">
                                    {aqaProLinks.map((link, idx) => (
                                      <div key={idx} className="aqa-link-card">
                                        <div className="aqa-link-header">
                                          <strong>{link.link_id || `Pro ${idx + 1}`}</strong>
                                          <span className="aqa-link-score">
                                            Nesso {(link.nesso_plausibility ?? 0).toFixed(2)}
                                          </span>
                                        </div>
                                        <div className="aqa-link-metrics">
                                          <span>Base {(link.base_score ?? 0).toFixed(2)}</span>
                                          <span>Norm {(link.norm_support ?? 0).toFixed(2)}</span>
                                          <span>Cogency {(link.cogency ?? 0).toFixed(2)}</span>
                                          <span>Sem {(link.semantics ?? 0).toFixed(2)}</span>
                                          {(link.attacks_sum ?? 0) > 0 && (
                                            <span style={{ color: '#f59e0b' }}>⚔️ {(link.attacks_sum ?? 0).toFixed(3)}</span>
                                          )}
                                          <span>Δ prec {(link.precedent_delta ?? 0).toFixed(2)}</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {aqaContraLinks.length > 0 && (
                                <div className="aqa-link-group">
                                  <h6>Counter-Reasoner (Contro)</h6>
                                  <div className="aqa-link-list">
                                    {aqaContraLinks.map((link, idx) => (
                                      <div key={idx} className="aqa-link-card">
                                        <div className="aqa-link-header">
                                          <strong>{link.link_id || `Contro ${idx + 1}`}</strong>
                                          <span className="aqa-link-score">
                                            Nesso {(link.nesso_plausibility ?? 0).toFixed(2)}
                                          </span>
                                        </div>
                                        <div className="aqa-link-metrics">
                                          <span>Base {(link.base_score ?? 0).toFixed(2)}</span>
                                          <span>Norm {(link.norm_support ?? 0).toFixed(2)}</span>
                                          <span>Cogency {(link.cogency ?? 0).toFixed(2)}</span>
                                          <span>Sem {(link.semantics ?? 0).toFixed(2)}</span>
                                          {(link.attacks_sum ?? 0) > 0 && (
                                            <span style={{ color: '#f59e0b' }}>⚔️ {(link.attacks_sum ?? 0).toFixed(3)}</span>
                                          )}
                                          <span>Δ prec {(link.precedent_delta ?? 0).toFixed(2)}</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          <details className="ir-toggle">
                            <summary>Dettagli AQA (JSON)</summary>
                            <pre className="code-block">
                              {JSON.stringify(aqaReport, null, 2)}
                            </pre>
                          </details>
                        </>
                      ) : (
                        <div className="aqa-disabled">AQA disabilitata</div>
                      )}
                    </div>
                  )}

                  {/* SEZIONE METAGRAFO ASPIC+ */}
                  {aqaReport && aqaReport.enabled && (aqaProLinks.length > 0 || aqaContraLinks.length > 0) && (
                    <div className="result-section pipeline-section">
                      <h3 className="section-header">
                        <GitBranch size={20} style={{ color: '#7c3aed' }} />
                        6. ASPIC+ Metagrafo — Attacchi Incrociati
                      </h3>
                      <p style={{ fontSize: '0.82rem', color: '#6b7280', marginBottom: '0.75rem' }}>
                        Clicca su un nodo per vedere i dettagli. Scroll per zoomare, trascina per spostare il grafo.
                      </p>
                      <AspicMetagraph aqaReport={aqaReport} />
                    </div>
                  )}

                  {/* SEZIONE DETTAGLIO TESTUALE ATTACCHI */}
                  {aqaReport && aqaReport.enabled && (aqaProLinks.length > 0 || aqaContraLinks.length > 0) && (
                    <div className="result-section pipeline-section">
                      <h3 className="section-header">
                        <Swords size={20} style={{ color: '#ef4444' }} />
                        7. Dettaglio Testuale Attacchi
                      </h3>
                      <p style={{ fontSize: '0.82rem', color: '#6b7280', marginBottom: '0.75rem' }}>
                        Clicca su un attacco per espandere il testo completo dei link coinvolti.
                      </p>
                      <AttackTextDetails aqaReport={aqaReport} />
                    </div>
                  )}
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
