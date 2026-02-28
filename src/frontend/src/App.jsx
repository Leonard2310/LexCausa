import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Send, Bot, User, Loader2, Brain, Scale, Search, FileText, CheckCircle2, XCircle, AlertTriangle, ClipboardCheck, Wrench, Settings, GitBranch, Swords, Download, Square, Plus } from 'lucide-react';
import './App.css';
import AspicMetagraph from './AspicMetagraph';
import AttackTextDetails from './AttackTextDetails';

// Tab types
const TABS = {
  SEARCH: 'search',
  REASON: 'reason',
  PIPELINE: 'pipeline',
  DOE: 'doe',
};

// API base URL - uses proxy in development
const API_BASE = '/api';

const TAB_WELCOME_MESSAGES = {
  [TABS.SEARCH]:
    'Ciao! Sono LexCausa. Ti aiuto a cercare norme e precedenti rilevanti per il tuo claim.',
  [TABS.REASON]:
    'Ciao! Sono LexCausa. Ti aiuto a costruire la tesi principale con ragionamento giuridico strutturato.',
  [TABS.PIPELINE]:
    'Ciao! Sono LexCausa. Ti aiuto con la pipeline completa: retrieval, Reasoner, Counter-Reasoner e Polisher-Evaluator.',
  [TABS.DOE]:
    'Ciao! Sono LexCausa. Ti aiuto a eseguire il DoE con confronto A/B automatico del Counter-Reasoner.',
};

const PIPELINE_PHASES = [
  { key: 'context_setup', label: 'Preparazione contesto' },
  { key: 'support', label: 'Argomentazione principale' },
  { key: 'counter', label: 'Argomentazione contraria' },
  { key: 'final_evaluation', label: 'Verifica finale' },
];

const ATTACK_LABELS_IT = {
  but_for_fails: 'Controfattuale fallisce',
  no_covering_law_or_low_support: 'Legge di copertura assente/debole',
  alternative_causal_path: 'Percorso causale alternativo',
  duty_to_act_missing_for_omission: "Manca l'obbligo giuridico di agire",
  abnormal_or_atypical_chain: 'Catena causale atipica/anomala',
  sole_sufficient_cause: 'Causa sopravvenuta da sola sufficiente',
  intervening_cause_breaks_chain: 'Fattore sopravvenuto interrompe il nesso',
  force_majeure_filter: 'Filtro caso fortuito/forza maggiore',
  damage_is_indirect: 'Danno indiretto/non immediato',
  damage_not_foreseeable: 'Danno non prevedibile ex ante',
  creditor_contributed: 'Concorso del creditore',
  creditor_failed_to_mitigate: 'Mancata mitigazione del danno',
  quantification_uncertain: 'Quantificazione danno incerta/speculativa',
  competence_or_procedure_regular: 'Competenza e procedura regolari',
  motivation_is_sufficient: 'Motivazione sufficiente',
  participation_not_essential_or_not_denied: 'Partecipazione non decisiva/garanzie rispettate',
  silence_rule_not_applicable: 'Regola su silenzio/termini non applicabile',
  vizio_non_invalidante_21_octies: 'Vizio non invalidante (art. 21-octies)',
  event_was_avoidable: 'Evento evitabile con diligenza',
  event_was_foreseeable: 'Evento prevedibile',
  risk_was_assumed_or_controllable: 'Rischio assunto o controllabile',
};

const sourceShortLabel = (source) => {
  if (source === 'codice_civile') return 'c.c.';
  if (source === 'codice_penale') return 'c.p.';
  if (source === 'codice_amministrativo') return 'L. 241/1990';
  return source || 'codice';
};

const createLivePipelineResult = (claim) => ({
  claim,
  retrieval_context: {
    statutes: [],
    precedents: [],
    memory: {
      enabled: false,
      overwrite: false,
      hit: false,
    },
  },
  reasoner: {
    raw_response: '',
    statutes: [],
    precedents: [],
  },
  counter_reasoner: {
    raw_response: '',
    statutes: [],
    precedents: [],
  },
  evaluation: {},
  _stream: {
    phases: {
      context_setup: 'pending',
      support: 'pending',
      counter: 'pending',
      final_evaluation: 'pending',
    },
    phase_progress: {
      context_setup: 0,
      support: 0,
      counter: 0,
      final_evaluation: 0,
    },
    phase_details: {
      context_setup: '',
      support: '',
      counter: '',
      final_evaluation: '',
    },
    support_max_step: 0,
    counter_max_step: 0,
    evaluation_checks_processed: 0,
    evaluation_expected_checks_by_agent: {
      reasoner: 0,
      counter_reasoner: 0,
    },
    evaluation_live_log: [],
    support_steps: {},
    counter_steps: {},
    support_conclusion_live: '',
  },
});

/** Collapsible list: shows first `limit` items, then a "Mostra tutti" button */
function CollapsibleList({ items, limit = 5, renderItem }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? items : items.slice(0, limit);
  return (
    <ul className="articles-list">
      {visible.map((item, idx) => renderItem(item, idx))}
      {items.length > limit && !expanded && (
        <li className="show-more-btn">
          <button onClick={() => setExpanded(true)}>
            Mostra tutti ({items.length - limit} rimanenti)
          </button>
        </li>
      )}
      {items.length > limit && expanded && (
        <li className="show-more-btn">
          <button onClick={() => setExpanded(false)}>
            Mostra meno
          </button>
        </li>
      )}
    </ul>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState(TABS.PIPELINE);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: TAB_WELCOME_MESSAGES[TABS.SEARCH],
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [reasoningResult, setReasoningResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineHistory, setPipelineHistory] = useState([]);
  const [reasonMessages, setReasonMessages] = useState([]);
  const [pipelineMessages, setPipelineMessages] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [pipelineSettings, setPipelineSettings] = useState({
    reasoner_model: 'gpt_oss_120b',
    counter_model: 'gpt_oss_120b',
    reasoner_temperature: 0,
    counter_temperature: 0.3,
    llm_max_tokens: 8192,
    search_top_k_default: 100,
    search_min_kept_statutes: 8,
    search_use_top_n_libri: 3,
    precedents_limit_default: 5,
    include_precedents: true,
    chain_min_steps: 3,
    chain_max_steps: 10,
    aqa_alpha: 0.3,
    aqa_beta: 0.4,
    aqa_gamma: 0.3,
    aqa_min_semantic_overlap: 0.5,
    aqa_min_strength_ratio: 1.2,
    aqa_damage_factor: 0.5,
    aqa_allow_factual_attacks: true,
    aqa_allow_cross_codice: true,
    reasoner_enable_causality: true,
    counter_enable_causality: true,
    counter_pass_causal_identity: true,
    counter_pass_taxonomy_attacks: true,
    counter_pass_norms: true,
    claim_context_memory_enabled: true,
    claim_context_memory_overwrite: false,
  });
  const [doeSettings, setDoeSettings] = useState({
    reasoner_enable_causality: true,
    counter_pass_causal_identity: false,
    counter_pass_taxonomy_attacks: false,
    counter_pass_norms: true,
    claim_context_memory_enabled: true,
    claim_context_memory_overwrite: false,
  });
  const messagesEndRef = useRef(null);
  const messagesAreaRef = useRef(null);
  const shouldAutoScrollRef = useRef(true);
  const inputRef = useRef(null);
  const pipelineAbortControllerRef = useRef(null);
  const activePipelineRunIdRef = useRef(null);
  const manualPipelineStopRef = useRef(false);
  const currentPipelinePdfRef = useRef(null);
  const historyPipelinePdfRefs = useRef({});
  const [exportingPdfKey, setExportingPdfKey] = useState(null);
  const [isStoppingPipeline, setIsStoppingPipeline] = useState(false);
  const [claimMemoryMenuOpen, setClaimMemoryMenuOpen] = useState(false);
  const [doeDisplaySetup, setDoeDisplaySetup] = useState('treatment');

  const isNearBottom = () => {
    const el = messagesAreaRef.current;
    if (!el) return true;
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    return distanceToBottom <= 120;
  };

  const scrollToBottom = (behavior = 'smooth') => {
    if (!shouldAutoScrollRef.current) return;
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  const handleMessagesScroll = () => {
    shouldAutoScrollRef.current = isNearBottom();
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, reasonMessages, pipelineMessages, reasoningResult, pipelineResult]);

  useEffect(() => {
    if (activeTab !== TABS.PIPELINE) {
      if (activeTab !== TABS.DOE) {
        setClaimMemoryMenuOpen(false);
      }
    }
  }, [activeTab]);

  // Load default settings from backend on mount
  useEffect(() => {
    fetch(`${API_BASE}/settings`)
      .then((res) => res.json())
      .then((data) => {
        setAvailableModels(data.models || []);
        const d = data.defaults || {};
        setPipelineSettings((prev) => ({
          ...prev,
          reasoner_model: d.reasoner_model || prev.reasoner_model,
          counter_model: d.counter_model || prev.counter_model,
          reasoner_temperature:
            d.reasoner_temperature ?? d.llm_temperature ?? prev.reasoner_temperature,
          counter_temperature:
            d.counter_temperature ?? d.llm_temperature ?? prev.counter_temperature,
          llm_max_tokens: d.llm_max_tokens ?? prev.llm_max_tokens,
          search_top_k_default: d.search_top_k_default ?? prev.search_top_k_default,
          search_min_kept_statutes: d.search_min_kept_statutes ?? prev.search_min_kept_statutes,
          search_use_top_n_libri: d.search_use_top_n_libri ?? prev.search_use_top_n_libri,
          precedents_limit_default: d.precedents_limit_default ?? prev.precedents_limit_default,
          include_precedents: d.include_precedents ?? prev.include_precedents,
          chain_min_steps: d.chain_min_steps ?? prev.chain_min_steps,
          chain_max_steps: d.chain_max_steps ?? prev.chain_max_steps,
          aqa_alpha: d.aqa_alpha ?? prev.aqa_alpha,
          aqa_beta: d.aqa_beta ?? prev.aqa_beta,
          aqa_gamma: d.aqa_gamma ?? prev.aqa_gamma,
          aqa_min_semantic_overlap: d.aqa_min_semantic_overlap ?? prev.aqa_min_semantic_overlap,
          aqa_min_strength_ratio: d.aqa_min_strength_ratio ?? prev.aqa_min_strength_ratio,
          aqa_damage_factor: d.aqa_damage_factor ?? prev.aqa_damage_factor,
          aqa_allow_factual_attacks: d.aqa_allow_factual_attacks ?? prev.aqa_allow_factual_attacks,
          aqa_allow_cross_codice: d.aqa_allow_cross_codice ?? prev.aqa_allow_cross_codice,
          // Pipeline defaults: full causality enabled
          reasoner_enable_causality: true,
          counter_enable_causality: true,
          counter_pass_causal_identity: true,
          counter_pass_taxonomy_attacks: true,
          counter_pass_norms: true,
        }));
        setDoeSettings((prev) => ({
          ...prev,
          // DoE defaults: baseline with only taxonomy norms pass-through enabled
          reasoner_enable_causality: true,
          counter_pass_causal_identity: false,
          counter_pass_taxonomy_attacks: false,
          counter_pass_norms: true,
        }));
      })
      .catch((err) => console.warn('Could not load settings:', err));
  }, []);

  const updateSetting = (key, value) => {
    setPipelineSettings((prev) => ({ ...prev, [key]: value }));
  };

  const updateDoeSetting = (key, value) => {
    setDoeSettings((prev) => ({ ...prev, [key]: value }));
  };

  const applyDoeViewSelection = (viewKey) => {
    const normalized = viewKey === 'baseline' ? 'baseline' : 'treatment';
    setDoeDisplaySetup(normalized);
  };

  const handleSearchSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    shouldAutoScrollRef.current = true;

    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          top_k: pipelineSettings.search_top_k_default,
        }),
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
    shouldAutoScrollRef.current = true;
    setIsLoading(true);
    setReasonMessages((prev) => [...prev, { role: 'user', content: claim }]);

    try {
      const response = await fetch(`${API_BASE}/reason`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          claim,
          include_precedents: true,
          use_context: false,
          settings: {
            reasoner_model: pipelineSettings.reasoner_model,
            reasoner_temperature: pipelineSettings.reasoner_temperature,
            reasoner_enable_causality: pipelineSettings.reasoner_enable_causality,
            llm_max_tokens: pipelineSettings.llm_max_tokens,
          },
        }),
      });

      if (!response.ok) throw new Error('Errore nella risposta del server');

      const data = await response.json();
      setReasoningResult(data);
      setReasonMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            data.raw_response ||
            'Ho completato l’analisi del ragionamento. Qui sotto trovi la risposta con i dettagli.',
        },
      ]);
    } catch (error) {
      console.error('Errore:', error);
      setReasoningResult({ error: error.message });
      setReasonMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Mi dispiace, si è verificato un errore durante il ragionamento: ${error.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopPipeline = async () => {
    if (!isLoading || (activeTab !== TABS.PIPELINE && activeTab !== TABS.DOE)) return;

    manualPipelineStopRef.current = true;
    setIsStoppingPipeline(true);
    setPipelineResult((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            final_evaluation: 'Interruzione richiesta...',
          },
        },
      };
    });

    const runId = activePipelineRunIdRef.current;
    try {
      await fetch(`${API_BASE}/pipeline/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(runId ? { run_id: runId } : {}),
      });
    } catch (err) {
      console.warn('Stop pipeline request failed:', err);
    }

    try {
      pipelineAbortControllerRef.current?.abort();
    } catch (err) {
      console.warn('Abort stream failed:', err);
    }
  };

  const isCompletedPipelineRun = (run) => Boolean(
    run
    && !run.error
    && run.evaluation
    && (
      run._stream?.phases?.final_evaluation === 'done'
      || run.evaluation?.aqa_report
      || run.evaluation?.summary
    ),
  );

  const buildPipelinePayload = (claim, activeSettings) => ({
    claim,
    include_precedents: activeSettings.include_precedents,
    max_statutes: activeSettings.search_top_k_default,
    max_precedents: activeSettings.precedents_limit_default,
    claim_context_memory_enabled: !!activeSettings.claim_context_memory_enabled,
    claim_context_memory_overwrite:
      !!activeSettings.claim_context_memory_enabled &&
      !!activeSettings.claim_context_memory_overwrite,
    settings: {
      reasoner_model: activeSettings.reasoner_model,
      counter_model: activeSettings.counter_model,
      reasoner_temperature: activeSettings.reasoner_temperature,
      counter_temperature: activeSettings.counter_temperature,
      llm_max_tokens: activeSettings.llm_max_tokens,
      search_min_kept_statutes: activeSettings.search_min_kept_statutes,
      search_use_top_n_libri: activeSettings.search_use_top_n_libri,
      chain_min_steps: activeSettings.chain_min_steps,
      chain_max_steps: activeSettings.chain_max_steps,
      aqa_alpha: activeSettings.aqa_alpha,
      aqa_beta: activeSettings.aqa_beta,
      aqa_gamma: activeSettings.aqa_gamma,
      aqa_min_semantic_overlap: activeSettings.aqa_min_semantic_overlap,
      aqa_min_strength_ratio: activeSettings.aqa_min_strength_ratio,
      aqa_damage_factor: activeSettings.aqa_damage_factor,
      aqa_allow_factual_attacks: activeSettings.aqa_allow_factual_attacks,
      aqa_allow_cross_codice: activeSettings.aqa_allow_cross_codice,
      reasoner_enable_causality: activeSettings.reasoner_enable_causality,
      counter_enable_causality: activeSettings.counter_enable_causality,
      counter_pass_causal_identity: activeSettings.counter_pass_causal_identity,
      counter_pass_taxonomy_attacks: activeSettings.counter_pass_taxonomy_attacks,
      counter_pass_norms: activeSettings.counter_pass_norms,
    },
  });

  const extractDoeMetrics = (run = {}) => {
    const aqa = run?.evaluation?.aqa_report || {};
    const consistency = run?.evaluation?.consistency_report || {};
    return {
      verdict: aqa?.verdict || 'n/a',
      final_score: Number(aqa?.net_plausibility?.final ?? NaN),
      pro_score: Number(aqa?.net_plausibility?.pro ?? NaN),
      contra_score: Number(aqa?.net_plausibility?.contra ?? NaN),
      pro_links: Array.isArray(aqa?.links?.pro) ? aqa.links.pro.length : 0,
      contra_links: Array.isArray(aqa?.links?.contra) ? aqa.links.contra.length : 0,
      reasoner_consistency: Number(consistency?.reasoner?.consistency_score ?? NaN),
      counter_consistency: Number(consistency?.counter_reasoner?.consistency_score ?? NaN),
    };
  };

  const computeDoeDelta = (baselineMetrics = {}, treatmentMetrics = {}) => {
    const deltaNumber = (a, b) => {
      const left = Number(a);
      const right = Number(b);
      if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
      return right - left;
    };
    return {
      final_score: deltaNumber(baselineMetrics.final_score, treatmentMetrics.final_score),
      pro_score: deltaNumber(baselineMetrics.pro_score, treatmentMetrics.pro_score),
      contra_score: deltaNumber(baselineMetrics.contra_score, treatmentMetrics.contra_score),
      reasoner_consistency: deltaNumber(
        baselineMetrics.reasoner_consistency,
        treatmentMetrics.reasoner_consistency,
      ),
      counter_consistency: deltaNumber(
        baselineMetrics.counter_consistency,
        treatmentMetrics.counter_consistency,
      ),
      pro_links: (treatmentMetrics.pro_links ?? 0) - (baselineMetrics.pro_links ?? 0),
      contra_links: (treatmentMetrics.contra_links ?? 0) - (baselineMetrics.contra_links ?? 0),
    };
  };

  const describeCounterTaxonomySetup = (settings = {}) => {
    if (!settings?.counter_enable_causality) {
      return 'Counter tassonomia OFF';
    }
    const enabledParts = [];
    if (settings?.counter_pass_causal_identity) enabledParts.push('identity');
    if (settings?.counter_pass_taxonomy_attacks) enabledParts.push('attacks');
    if (settings?.counter_pass_norms) enabledParts.push('norms');
    if (enabledParts.length === 0) {
      return 'Counter tassonomia ON (senza pass-through)';
    }
    return `Counter tassonomia ON (${enabledParts.join(' + ')})`;
  };

  const createEmptyDoeView = () => ({
    counter_reasoner: {
      raw_response: '',
      statutes: [],
      precedents: [],
    },
    evaluation: {},
  });

  const createDoeComparisonSeed = ({
    setupASettings,
    setupBSettings,
    setupADescription,
    setupBDescription,
  }) => ({
    mode: 'automatic_ab',
    status: 'running',
    baseline: {
      label: 'A (Baseline)',
      description: setupADescription,
      settings: {
        reasoner_enable_causality: setupASettings.reasoner_enable_causality,
        counter_enable_causality: setupASettings.counter_enable_causality,
        counter_pass_causal_identity: setupASettings.counter_pass_causal_identity,
        counter_pass_taxonomy_attacks: setupASettings.counter_pass_taxonomy_attacks,
        counter_pass_norms: setupASettings.counter_pass_norms,
      },
      status: 'running',
      duration_ms: null,
      metrics: {},
      view: createEmptyDoeView(),
    },
    treatment: {
      label: 'B (Treatment)',
      description: setupBDescription,
      settings: {
        reasoner_enable_causality: setupBSettings.reasoner_enable_causality,
        counter_enable_causality: setupBSettings.counter_enable_causality,
        counter_pass_causal_identity: setupBSettings.counter_pass_causal_identity,
        counter_pass_taxonomy_attacks: setupBSettings.counter_pass_taxonomy_attacks,
        counter_pass_norms: setupBSettings.counter_pass_norms,
      },
      status: 'queued',
      duration_ms: null,
      metrics: {},
      view: createEmptyDoeView(),
    },
    delta: null,
    artifacts: {},
  });

  const persistDoeArtifacts = async (payload) => {
    const response = await fetch(`${API_BASE}/doe/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let errText = 'Errore nel salvataggio del log DoE';
      try {
        const err = await response.json();
        errText = err?.error || errText;
      } catch (_) {
        // ignore json parse failures
      }
      throw new Error(errText);
    }
    return response.json();
  };

  const handleDoeSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const claim = input.trim();
    const mergedDoeSettings = { ...pipelineSettings, ...doeSettings };
    const setupAPassCausalIdentity = !!mergedDoeSettings.counter_pass_causal_identity;
    const setupAPassTaxonomyAttacks = !!mergedDoeSettings.counter_pass_taxonomy_attacks;
    const setupAPassNorms = !!mergedDoeSettings.counter_pass_norms;
    const setupACounterEnabled = (
      setupAPassCausalIdentity
      || setupAPassTaxonomyAttacks
      || setupAPassNorms
    );
    const setupASettings = {
      ...mergedDoeSettings,
      counter_enable_causality: setupACounterEnabled,
      counter_pass_causal_identity: setupAPassCausalIdentity,
      counter_pass_taxonomy_attacks: setupAPassTaxonomyAttacks,
      counter_pass_norms: setupAPassNorms,
    };
    const setupBSettings = {
      ...mergedDoeSettings,
      counter_enable_causality: true,
      counter_pass_causal_identity: true,
      counter_pass_taxonomy_attacks: true,
      counter_pass_norms: true,
    };
    const setupADescription = describeCounterTaxonomySetup(setupASettings);
    const setupBDescription = describeCounterTaxonomySetup(setupBSettings);
    const doeComparisonSeed = createDoeComparisonSeed({
      setupASettings,
      setupBSettings,
      setupADescription,
      setupBDescription,
    });

    const payloadA = buildPipelinePayload(claim, setupASettings);

    setClaimMemoryMenuOpen(false);
    manualPipelineStopRef.current = false;
    activePipelineRunIdRef.current = null;
    pipelineAbortControllerRef.current = null;
    setIsStoppingPipeline(false);
    setDoeDisplaySetup('treatment');

    setInput('');
    shouldAutoScrollRef.current = true;
    setIsLoading(true);
    if (isCompletedPipelineRun(pipelineResult)) {
      setPipelineHistory((prev) => [...prev, pipelineResult].slice(-10));
    }
    setPipelineMessages((prev) => [...prev, { role: 'user', content: claim }]);
    const doeLiveSeed = createLivePipelineResult(claim);
    setPipelineResult({
      ...doeLiveSeed,
      doe_ab_comparison: doeComparisonSeed,
      _stream: {
        ...doeLiveSeed._stream,
        phases: {
          context_setup: 'done',
          support: 'active',
          counter: 'pending',
          final_evaluation: 'pending',
        },
        phase_progress: {
          context_setup: 100,
          support: 20,
          counter: 0,
          final_evaluation: 0,
        },
        phase_details: {
          context_setup: `DoE pronto: A=${setupADescription} | B=${setupBDescription}`,
          support: 'Reasoner condiviso A/B in esecuzione',
          counter: 'In attesa',
          final_evaluation: 'In attesa',
        },
      },
    });

    const updateDoeLivePipeline = (mutator) => {
      setPipelineResult((prev) => {
        if (!prev || prev.error) return prev;
        return mutator(prev);
      });
    };

    const updateDoeComparison = (mutator) => {
      setPipelineResult((prev) => {
        if (!prev) return prev;
        const currentComparison = prev.doe_ab_comparison || doeComparisonSeed;
        return {
          ...prev,
          doe_ab_comparison: mutator(currentComparison, prev),
        };
      });
    };

    const mergeDoeView = (setupKey, patch = {}) => {
      updateDoeComparison((comparison) => ({
        ...comparison,
        [setupKey]: {
          ...(comparison?.[setupKey] || {}),
          view: {
            ...createEmptyDoeView(),
            ...(comparison?.[setupKey]?.view || {}),
            ...patch,
          },
        },
      }));
    };

    const mergeDoeEvaluationPartial = (setupKey, partial = {}) => {
      updateDoeComparison((comparison) => ({
        ...comparison,
        [setupKey]: {
          ...(comparison?.[setupKey] || {}),
          view: {
            ...createEmptyDoeView(),
            ...(comparison?.[setupKey]?.view || {}),
            evaluation: mergeEvaluationPartial(
              comparison?.[setupKey]?.view?.evaluation,
              partial || {},
            ),
          },
        },
      }));
    };

    const setDoePhaseStatus = (phaseKey, phaseValue, detail = null) => {
      updateDoeLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phases: {
            ...(prev._stream?.phases || {}),
            [phaseKey]: phaseValue,
          },
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            [phaseKey]:
              detail !== null
                ? detail
                : prev._stream?.phase_details?.[phaseKey] || '',
          },
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            [phaseKey]:
              phaseValue === 'done'
                ? 100
                : phaseValue === 'active'
                  ? Math.max(8, prev._stream?.phase_progress?.[phaseKey] || 0)
                  : prev._stream?.phase_progress?.[phaseKey] || 0,
          },
        },
      }));
    };

    const setDoePhaseDetail = (phaseKey, detail) => {
      updateDoeLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            [phaseKey]: detail || '',
          },
        },
      }));
    };

    const setDoePhaseProgress = (phaseKey, value) => {
      updateDoeLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            [phaseKey]: Math.max(0, Math.min(100, value)),
          },
        },
      }));
    };

    const bumpDoePhaseProgress = (phaseKey, value) => {
      updateDoeLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            [phaseKey]: Math.max(
              prev._stream?.phase_progress?.[phaseKey] || 0,
              Math.max(0, Math.min(100, value)),
            ),
          },
        },
      }));
    };

    const appendDoePhaseToken = (phase, token, stepNumber = null) => {
      if (!token) return;
      updateDoeLivePipeline((prev) => {
        const next = {
          ...prev,
          _stream: {
            ...(prev._stream || {}),
            phase_progress: { ...(prev._stream?.phase_progress || {}) },
            support_steps: { ...(prev._stream?.support_steps || {}) },
            counter_steps: { ...(prev._stream?.counter_steps || {}) },
            support_max_step: prev._stream?.support_max_step || 0,
            counter_max_step: prev._stream?.counter_max_step || 0,
            support_conclusion_live: prev._stream?.support_conclusion_live || '',
          },
        };
        if (phase === 'support' || phase === 'support_conclusion') {
          next.reasoner = {
            ...(prev.reasoner || {}),
            raw_response: `${prev.reasoner?.raw_response || ''}${token}`,
          };
          if (stepNumber != null && phase === 'support') {
            const prevStepText = next._stream.support_steps[stepNumber] || '';
            next._stream.support_steps[stepNumber] = `${prevStepText}${token}`;
            next._stream.support_max_step = Math.max(
              next._stream.support_max_step,
              Number(stepNumber) || 0,
            );
            const approxProgress = Math.min(
              92,
              Math.max(
                12,
                (next._stream.support_max_step / Math.max(1, mergedDoeSettings.chain_max_steps)) * 92,
              ),
            );
            next._stream.phase_progress.support = approxProgress;
          } else if (phase === 'support_conclusion') {
            next._stream.support_conclusion_live = `${next._stream.support_conclusion_live}${token}`;
            next._stream.phase_progress.support = Math.max(
              next._stream.phase_progress.support || 0,
              96,
            );
          }
        } else if (phase === 'counter') {
          next.counter_reasoner = {
            ...(prev.counter_reasoner || {}),
            raw_response: `${prev.counter_reasoner?.raw_response || ''}${token}`,
          };
          if (stepNumber != null) {
            const prevStepText = next._stream.counter_steps[stepNumber] || '';
            next._stream.counter_steps[stepNumber] = `${prevStepText}${token}`;
            next._stream.counter_max_step = Math.max(
              next._stream.counter_max_step,
              Number(stepNumber) || 0,
            );
            const approxProgress = Math.min(
              92,
              Math.max(
                12,
                (next._stream.counter_max_step / Math.max(1, mergedDoeSettings.chain_max_steps)) * 92,
              ),
            );
            next._stream.phase_progress.counter = approxProgress;
          }
        }
        return next;
      });
    };

    const resetDoePhaseStepLive = (phase, stepNumber = null) => {
      if (stepNumber == null) return;
      updateDoeLivePipeline((prev) => {
        const next = {
          ...prev,
          _stream: {
            ...(prev._stream || {}),
            support_steps: { ...(prev._stream?.support_steps || {}) },
            counter_steps: { ...(prev._stream?.counter_steps || {}) },
          },
        };
        if (phase === 'support') {
          next._stream.support_steps[stepNumber] = '';
        } else if (phase === 'counter') {
          next._stream.counter_steps[stepNumber] = '';
        }
        return next;
      });
    };

    const runPipelineStream = async (payload, controller) => {
      const response = await fetch(`${API_BASE}/pipeline/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) {
        let errText = 'Errore nella pipeline';
        try {
          const err = await response.json();
          errText = err?.error || errText;
        } catch (_) {
          // ignore json parse failures
        }
        throw new Error(errText);
      }
      if (!response.body) {
        throw new Error('Streaming non disponibile: risposta senza body.');
      }

      const decoder = new TextDecoder('utf-8');
      const reader = response.body.getReader();
      let buffer = '';
      let finalPayload = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          if (!rawEvent.trim()) continue;
          const lines = rawEvent.split('\n');
          let eventName = 'message';
          let dataText = '';

          lines.forEach((line) => {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataText += line.slice(5).trim();
            }
          });

          if (!dataText) continue;

          let eventPayload;
          try {
            eventPayload = JSON.parse(dataText);
          } catch (_) {
            continue;
          }

          if (eventName === 'heartbeat' || eventName === 'status') continue;

          if (eventName === 'run_started') {
            activePipelineRunIdRef.current = eventPayload?.run_id || null;
            continue;
          }

          if (eventName === 'phase') {
            const phaseKey = eventPayload?.phase;
            const phaseStatus = eventPayload?.status;
            const phaseProgress = Number(eventPayload?.progress);
            const phaseDetail = eventPayload?.detail || '';
            const isKnownPhase = PIPELINE_PHASES.some((phase) => phase.key === phaseKey);
            if (!isKnownPhase) continue;
            if (phaseStatus) {
              setDoePhaseStatus(phaseKey, phaseStatus, phaseDetail || null);
            }
            if (Number.isFinite(phaseProgress)) {
              setDoePhaseProgress(phaseKey, phaseProgress);
            }
            if (phaseDetail) {
              setDoePhaseDetail(phaseKey, phaseDetail);
            }
            continue;
          }

          if (eventName === 'token') {
            const phase = eventPayload.phase || 'generic';
            if (eventPayload?.action === 'reset_step') {
              resetDoePhaseStepLive(phase, eventPayload?.step ?? null);
              continue;
            }
            if (phase === 'support' || phase === 'support_conclusion') {
              setDoePhaseStatus('context_setup', 'done');
              setDoePhaseStatus('support', 'active');
              if (phase === 'support' && eventPayload?.step != null) {
                setDoePhaseDetail(
                  'support',
                  `Step ${eventPayload.step}/${mergedDoeSettings.chain_max_steps}`,
                );
              } else if (phase === 'support_conclusion') {
                setDoePhaseDetail('support', 'Sintesi e formattazione conclusione');
              }
            }
            if (phase === 'counter') {
              setDoePhaseStatus('support', 'done');
              setDoePhaseStatus('counter', 'active');
              if (eventPayload?.step != null) {
                setDoePhaseDetail(
                  'counter',
                  `Step ${eventPayload.step}/${mergedDoeSettings.chain_max_steps}`,
                );
              }
            }
            appendDoePhaseToken(phase, eventPayload.token || '', eventPayload.step ?? null);
            continue;
          }

          if (eventName === 'retrieval_context') {
            setPipelineResult((prev) => ({
              ...(prev || {}),
              retrieval_context: eventPayload || {
                statutes: [],
                precedents: [],
                memory: {},
              },
            }));
            continue;
          }

          if (eventName === 'reasoner_result') {
            setDoePhaseStatus('support', 'done', 'Argomentazione e ASPIC+ completati');
            setPipelineResult((prev) => ({
              ...(prev || {}),
              reasoner: eventPayload || {},
            }));
            continue;
          }

          if (eventName === 'counter_result') {
            setDoePhaseStatus('counter', 'done', 'Contro-argomentazione e ASPIC+ completati');
            mergeDoeView('baseline', {
              counter_reasoner: eventPayload || {},
            });
            setPipelineResult((prev) => ({
              ...(prev || {}),
              counter_reasoner: eventPayload || {},
            }));
            continue;
          }

          if (eventName === 'evaluation_partial') {
            setDoePhaseStatus('final_evaluation', 'active');
            setDoePhaseDetail('final_evaluation', 'Aggiornamento valutazione in corso');
            mergeDoeEvaluationPartial('baseline', eventPayload || {});
            setPipelineResult((prev) => ({
              ...(prev || {}),
              evaluation: mergeEvaluationPartial(prev?.evaluation, eventPayload || {}),
            }));
            continue;
          }

          if (eventName === 'evaluation_status') {
            const relativeProgress = Number(eventPayload?.progress);
            if (Number.isFinite(relativeProgress)) {
              bumpDoePhaseProgress('final_evaluation', 78 + relativeProgress * 20);
            } else {
              bumpDoePhaseProgress('final_evaluation', 82);
            }
            continue;
          }

          if (eventName === 'evaluation_result') {
            setDoePhaseStatus('final_evaluation', 'done', 'Report finale consolidato');
            mergeDoeView('baseline', {
              evaluation: eventPayload || {},
            });
            setPipelineResult((prev) => ({
              ...(prev || {}),
              evaluation: eventPayload || {},
            }));
            continue;
          }

          if (eventName === 'error') {
            throw new Error(eventPayload.message || 'Errore nella pipeline');
          }

          if (eventName === 'cancelled') {
            throw new Error(eventPayload?.message || 'Esecuzione interrotta manualmente.');
          }

          if (eventName === 'final') {
            finalPayload = eventPayload;
            setPipelineResult((prev) => ({
              ...eventPayload,
              doe_ab_comparison: prev?.doe_ab_comparison || null,
              _stream: {
                ...(prev?._stream || {}),
                phases: {
                  context_setup: 'done',
                  support: 'done',
                  counter: 'done',
                  final_evaluation: 'done',
                },
                phase_progress: {
                  context_setup: 100,
                  support: 100,
                  counter: 100,
                  final_evaluation: 100,
                },
              },
            }));
            continue;
          }
        }
      }

      if (!finalPayload) {
        throw new Error('Streaming interrotto prima del risultato finale.');
      }
      return finalPayload;
    };

    const runCounterStream = async (payload, controller) => {
      const response = await fetch(`${API_BASE}/counter_reason/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) {
        let errText = 'Errore nel Counter-Reasoner (stream)';
        try {
          const err = await response.json();
          errText = err?.error || errText;
        } catch (_) {
          // ignore json parse failures
        }
        throw new Error(errText);
      }
      if (!response.body) {
        throw new Error('Streaming counter non disponibile: risposta senza body.');
      }

      const decoder = new TextDecoder('utf-8');
      const reader = response.body.getReader();
      let buffer = '';
      let finalPayload = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          if (!rawEvent.trim()) continue;
          const lines = rawEvent.split('\n');
          let eventName = 'message';
          let dataText = '';

          lines.forEach((line) => {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataText += line.slice(5).trim();
            }
          });

          if (!dataText) continue;

          let eventPayload;
          try {
            eventPayload = JSON.parse(dataText);
          } catch (_) {
            continue;
          }

          if (eventName === 'heartbeat') continue;
          if (eventName === 'run_started') {
            activePipelineRunIdRef.current = eventPayload?.run_id || null;
            continue;
          }
          if (eventName === 'phase') {
            const phaseKey = eventPayload?.phase;
            const phaseStatus = eventPayload?.status;
            const phaseProgress = Number(eventPayload?.progress);
            const phaseDetail = eventPayload?.detail || '';
            const isKnownPhase = PIPELINE_PHASES.some((phase) => phase.key === phaseKey);
            if (!isKnownPhase) continue;
            if (phaseStatus) {
              setDoePhaseStatus(phaseKey, phaseStatus, phaseDetail || null);
            }
            if (Number.isFinite(phaseProgress)) {
              setDoePhaseProgress(phaseKey, phaseProgress);
            }
            if (phaseDetail) {
              setDoePhaseDetail(phaseKey, phaseDetail);
            }
            continue;
          }
          if (eventName === 'status') {
            if (eventPayload?.message) {
              setDoePhaseStatus('counter', 'active');
              setDoePhaseDetail('counter', eventPayload.message);
            }
            continue;
          }
          if (eventName === 'retrieval_context') {
            setPipelineResult((prev) => ({
              ...(prev || {}),
              retrieval_context: eventPayload || {
                statutes: [],
                precedents: [],
                memory: {},
              },
            }));
            continue;
          }
          if (eventName === 'token') {
            const phase = eventPayload.phase || 'counter';
            if (eventPayload?.action === 'reset_step') {
              resetDoePhaseStepLive(phase, eventPayload?.step ?? null);
              continue;
            }
            setDoePhaseStatus('counter', 'active');
            if (eventPayload?.step != null) {
              setDoePhaseDetail(
                'counter',
                `Step ${eventPayload.step}/${mergedDoeSettings.chain_max_steps}`,
              );
            }
            appendDoePhaseToken(phase, eventPayload.token || '', eventPayload.step ?? null);
            continue;
          }
          if (eventName === 'counter_result') {
            setDoePhaseStatus('counter', 'done', 'Contro-argomentazione e ASPIC+ completati');
            mergeDoeView('treatment', {
              counter_reasoner: eventPayload || {},
            });
            setPipelineResult((prev) => ({
              ...(prev || {}),
              counter_reasoner: eventPayload || {},
            }));
            continue;
          }
          if (eventName === 'error') {
            throw new Error(eventPayload.message || 'Errore nel Counter-Reasoner');
          }
          if (eventName === 'cancelled') {
            throw new Error(eventPayload?.message || 'Counter-Reasoner interrotto manualmente.');
          }
          if (eventName === 'final') {
            finalPayload = eventPayload;
            continue;
          }
        }
      }

      if (!finalPayload) {
        throw new Error('Streaming Counter interrotto prima del risultato finale.');
      }
      return finalPayload;
    };

    const runEvaluateStream = async (payload, controller) => {
      const response = await fetch(`${API_BASE}/evaluate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) {
        let errText = 'Errore nella valutazione finale (stream)';
        try {
          const err = await response.json();
          errText = err?.error || errText;
        } catch (_) {
          // ignore json parse failures
        }
        throw new Error(errText);
      }
      if (!response.body) {
        throw new Error('Streaming evaluator non disponibile: risposta senza body.');
      }

      const decoder = new TextDecoder('utf-8');
      const reader = response.body.getReader();
      let buffer = '';
      let finalPayload = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          if (!rawEvent.trim()) continue;
          const lines = rawEvent.split('\n');
          let eventName = 'message';
          let dataText = '';

          lines.forEach((line) => {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataText += line.slice(5).trim();
            }
          });

          if (!dataText) continue;

          let eventPayload;
          try {
            eventPayload = JSON.parse(dataText);
          } catch (_) {
            continue;
          }

          if (eventName === 'heartbeat') continue;
          if (eventName === 'run_started') {
            activePipelineRunIdRef.current = eventPayload?.run_id || null;
            continue;
          }
          if (eventName === 'phase') {
            const phaseKey = eventPayload?.phase;
            const phaseStatus = eventPayload?.status;
            const phaseProgress = Number(eventPayload?.progress);
            const phaseDetail = eventPayload?.detail || '';
            const isKnownPhase = PIPELINE_PHASES.some((phase) => phase.key === phaseKey);
            if (!isKnownPhase) continue;
            if (phaseStatus) {
              setDoePhaseStatus(phaseKey, phaseStatus, phaseDetail || null);
            }
            if (Number.isFinite(phaseProgress)) {
              setDoePhaseProgress(phaseKey, phaseProgress);
            }
            if (phaseDetail) {
              setDoePhaseDetail(phaseKey, phaseDetail);
            }
            continue;
          }
          if (eventName === 'evaluation_partial') {
            setDoePhaseStatus('final_evaluation', 'active');
            setDoePhaseDetail('final_evaluation', 'Aggiornamento valutazione in corso');
            mergeDoeEvaluationPartial('treatment', eventPayload || {});
            setPipelineResult((prev) => ({
              ...(prev || {}),
              evaluation: mergeEvaluationPartial(prev?.evaluation, eventPayload || {}),
            }));
            continue;
          }
          if (eventName === 'evaluation_status') {
            const stage = eventPayload?.stage || '';
            const stageDetailMap = {
              start: 'Check KB catena principale',
              kb_reasoner_done: 'Check KB catena contraria',
              kb_counter_done: 'Verifica opposizione tra tesi',
              gate_done: 'Gate opposizione completato',
              repair_start: 'Riparazione citazioni/catene in corso',
              repair_done: 'Riparazione completata, avvio AQA',
              aqa_done: 'AQA completata, preparazione report',
              done: 'Valutazione completata',
            };
            if (stageDetailMap[stage]) {
              setDoePhaseDetail('final_evaluation', stageDetailMap[stage]);
            }
            if (stage === 'done') {
              setDoePhaseStatus('final_evaluation', 'done');
              setDoePhaseProgress('final_evaluation', 100);
              continue;
            }
            setDoePhaseStatus('final_evaluation', 'active');
            const stageProgressMap = {
              start: 12,
              kb_reasoner_done: 46,
              kb_counter_done: 72,
              gate_done: 80,
              repair_start: 84,
              repair_done: 92,
              aqa_done: 98,
            };
            if (stageProgressMap[stage] != null) {
              setDoePhaseProgress('final_evaluation', stageProgressMap[stage]);
            }
            continue;
          }
          if (eventName === 'evaluation_aqa_progress') {
            const message = eventPayload?.message || 'AQA in corso';
            const relativeProgress = Number(eventPayload?.progress);
            setDoePhaseStatus('final_evaluation', 'active');
            setDoePhaseDetail('final_evaluation', message);
            if (Number.isFinite(relativeProgress)) {
              bumpDoePhaseProgress('final_evaluation', 78 + relativeProgress * 20);
            } else {
              bumpDoePhaseProgress('final_evaluation', 82);
            }
            continue;
          }
          if (eventName === 'evaluation_result') {
            setDoePhaseStatus('final_evaluation', 'done', 'Report finale consolidato');
            mergeDoeView('treatment', {
              evaluation: eventPayload || {},
            });
            setPipelineResult((prev) => ({
              ...(prev || {}),
              evaluation: eventPayload || {},
            }));
            continue;
          }
          if (eventName === 'error') {
            throw new Error(eventPayload.message || 'Errore nella valutazione finale');
          }
          if (eventName === 'cancelled') {
            throw new Error(eventPayload?.message || 'Valutazione interrotta manualmente.');
          }
          if (eventName === 'final') {
            finalPayload = eventPayload;
            continue;
          }
        }
      }

      if (!finalPayload) {
        throw new Error('Streaming Evaluator interrotto prima del risultato finale.');
      }
      return finalPayload;
    };

    try {
      const runAController = new AbortController();
      pipelineAbortControllerRef.current = runAController;
      const runAStart = performance.now();
      const runAResult = await runPipelineStream(payloadA, runAController);
      const runADurationMs = Math.round(performance.now() - runAStart);
      const baselineMetrics = extractDoeMetrics(runAResult);

      updateDoeComparison((comparison) => ({
        ...comparison,
        baseline: {
          ...(comparison?.baseline || {}),
          status: 'done',
          duration_ms: runADurationMs,
          metrics: baselineMetrics,
          view: {
            ...createEmptyDoeView(),
            ...(comparison?.baseline?.view || {}),
            counter_reasoner: runAResult?.counter_reasoner || {},
            evaluation: runAResult?.evaluation || {},
          },
        },
        treatment: {
          ...(comparison?.treatment || {}),
          status: 'running',
        },
      }));

      const sharedReasonerConclusion = String(runAResult?.reasoner?.conclusion || '').trim();
      if (!sharedReasonerConclusion) {
        throw new Error('Conclusione del Reasoner non disponibile per il confronto DoE.');
      }

      setPipelineResult((prev) => ({
        ...(prev || createLivePipelineResult(claim)),
        _stream: {
          ...(prev?._stream || {}),
          phases: {
            context_setup: 'done',
            support: 'done',
            counter: 'active',
            final_evaluation: 'pending',
          },
          phase_progress: {
            context_setup: 100,
            support: 100,
            counter: 30,
            final_evaluation: 0,
          },
          phase_details: {
            context_setup: 'Run A baseline completato (Reasoner condiviso fissato)',
            support: `Reasoner condiviso completato (${formatDurationHuman(runADurationMs)})`,
            counter: `Run B treatment in corso (solo Counter + Evaluator): ${setupBDescription}`,
            final_evaluation: 'Confronto A/B in preparazione',
          },
        },
      }));

      const runBController = new AbortController();
      pipelineAbortControllerRef.current = runBController;
      const runBStart = performance.now();
      const runBRouting =
        runAResult?.final_routing
        || runAResult?.routing
        || {};
      const runBRetrievalStatutes = Array.isArray(runAResult?.retrieval_context?.statutes)
        ? runAResult.retrieval_context.statutes
        : null;
      const runBRetrievalPrecedents = Array.isArray(runAResult?.retrieval_context?.precedents)
        ? runAResult.retrieval_context.precedents
        : null;
      const runBCounterPayload = {
        claim,
        include_precedents: setupBSettings.include_precedents,
        max_statutes: setupBSettings.search_top_k_default,
        max_precedents: setupBSettings.precedents_limit_default,
        claim_context_memory_enabled: !!setupBSettings.claim_context_memory_enabled,
        claim_context_memory_overwrite:
          !!setupBSettings.claim_context_memory_enabled
          && !!setupBSettings.claim_context_memory_overwrite,
        reasoner_conclusion: sharedReasonerConclusion,
        routing: runBRouting,
        settings: {
          counter_model: setupBSettings.counter_model,
          counter_temperature: setupBSettings.counter_temperature,
          llm_max_tokens: setupBSettings.llm_max_tokens,
          counter_enable_causality: setupBSettings.counter_enable_causality,
          counter_pass_causal_identity: setupBSettings.counter_pass_causal_identity,
          counter_pass_taxonomy_attacks: setupBSettings.counter_pass_taxonomy_attacks,
          counter_pass_norms: setupBSettings.counter_pass_norms,
        },
      };
      if (runBRetrievalStatutes && runBRetrievalPrecedents) {
        runBCounterPayload.pre_retrieved_statutes = runBRetrievalStatutes;
        runBCounterPayload.pre_retrieved_precedents = runBRetrievalPrecedents;
      }
      const runBCounterResult = await runCounterStream(runBCounterPayload, runBController);
      const runBEvaluationPayload = {
        claim,
        domain:
          runAResult?.final_routing?.domain
          || runAResult?.routing?.domain
          || 'ENTRAMBI',
        reasoner_output: runAResult?.reasoner || {},
        counter_output: runBCounterResult || {},
        settings: {
          aqa_alpha: setupBSettings.aqa_alpha,
          aqa_beta: setupBSettings.aqa_beta,
          aqa_gamma: setupBSettings.aqa_gamma,
          aqa_min_semantic_overlap: setupBSettings.aqa_min_semantic_overlap,
          aqa_min_strength_ratio: setupBSettings.aqa_min_strength_ratio,
          aqa_damage_factor: setupBSettings.aqa_damage_factor,
          aqa_allow_factual_attacks: setupBSettings.aqa_allow_factual_attacks,
          aqa_allow_cross_codice: setupBSettings.aqa_allow_cross_codice,
        },
      };
      const runBEvaluationResult = await runEvaluateStream(
        runBEvaluationPayload,
        runBController,
      );
      const runBDurationMs = Math.round(performance.now() - runBStart);
      const runBResult = {
        ...runAResult,
        claim,
        counter_reasoner: runBCounterResult,
        evaluation: runBEvaluationResult,
      };

      const treatmentMetrics = extractDoeMetrics(runBResult);
      const deltaMetrics = computeDoeDelta(baselineMetrics, treatmentMetrics);

      let doeArtifacts = {};
      try {
        doeArtifacts = await persistDoeArtifacts({
          claim,
          mode: 'automatic_ab',
          reasoner_shared: {
            routing: runAResult?.final_routing || runAResult?.routing || {},
            retrieval_context: runAResult?.retrieval_context || {},
            reasoner: runAResult?.reasoner || {},
          },
          baseline: {
            label: 'A (Baseline)',
            description: setupADescription,
            settings: {
              reasoner_enable_causality: setupASettings.reasoner_enable_causality,
              counter_enable_causality: setupASettings.counter_enable_causality,
              counter_pass_causal_identity: setupASettings.counter_pass_causal_identity,
              counter_pass_taxonomy_attacks: setupASettings.counter_pass_taxonomy_attacks,
              counter_pass_norms: setupASettings.counter_pass_norms,
            },
            status: 'done',
            duration_ms: runADurationMs,
            metrics: baselineMetrics,
            view: {
              counter_reasoner: runAResult?.counter_reasoner || {},
              evaluation: runAResult?.evaluation || {},
            },
          },
          treatment: {
            label: 'B (Treatment)',
            description: setupBDescription,
            settings: {
              reasoner_enable_causality: setupBSettings.reasoner_enable_causality,
              counter_enable_causality: setupBSettings.counter_enable_causality,
              counter_pass_causal_identity: setupBSettings.counter_pass_causal_identity,
              counter_pass_taxonomy_attacks: setupBSettings.counter_pass_taxonomy_attacks,
              counter_pass_norms: setupBSettings.counter_pass_norms,
            },
            status: 'done',
            duration_ms: runBDurationMs,
            metrics: treatmentMetrics,
            view: {
              counter_reasoner: runBResult?.counter_reasoner || {},
              evaluation: runBResult?.evaluation || {},
            },
          },
          delta: {
            ...deltaMetrics,
            duration_ms: runBDurationMs - runADurationMs,
          },
        });
      } catch (persistError) {
        console.warn('Salvataggio log DoE non riuscito:', persistError);
      }

      const doeComparison = {
        mode: 'automatic_ab',
        status: 'completed',
        baseline: {
          label: 'A (Baseline)',
          description: setupADescription,
          settings: {
            reasoner_enable_causality: setupASettings.reasoner_enable_causality,
            counter_enable_causality: setupASettings.counter_enable_causality,
            counter_pass_causal_identity: setupASettings.counter_pass_causal_identity,
            counter_pass_taxonomy_attacks: setupASettings.counter_pass_taxonomy_attacks,
            counter_pass_norms: setupASettings.counter_pass_norms,
          },
          status: 'done',
          duration_ms: runADurationMs,
          metrics: baselineMetrics,
          view: {
            counter_reasoner: runAResult?.counter_reasoner || {},
            evaluation: runAResult?.evaluation || {},
          },
        },
        treatment: {
          label: 'B (Treatment)',
          description: setupBDescription,
          settings: {
            reasoner_enable_causality: setupBSettings.reasoner_enable_causality,
            counter_enable_causality: setupBSettings.counter_enable_causality,
            counter_pass_causal_identity: setupBSettings.counter_pass_causal_identity,
            counter_pass_taxonomy_attacks: setupBSettings.counter_pass_taxonomy_attacks,
            counter_pass_norms: setupBSettings.counter_pass_norms,
          },
          status: 'done',
          duration_ms: runBDurationMs,
          metrics: treatmentMetrics,
          view: {
            counter_reasoner: runBResult?.counter_reasoner || {},
            evaluation: runBResult?.evaluation || {},
          },
        },
        delta: {
          ...deltaMetrics,
          duration_ms: runBDurationMs - runADurationMs,
        },
        artifacts: doeArtifacts || {},
      };

      setPipelineResult({
        ...runBResult,
        claim,
        doe_ab_comparison: doeComparison,
        _stream: {
          ...(runBResult?._stream || {}),
          phases: {
            context_setup: 'done',
            support: 'done',
            counter: 'done',
            final_evaluation: 'done',
          },
          phase_details: {
            context_setup: 'Run A/B completati con Reasoner condiviso',
            support: `A baseline completo (${formatDurationHuman(runADurationMs)}, include Reasoner)`,
            counter: `B treatment completato (${formatDurationHuman(runBDurationMs)}, Counter+Evaluator)`,
            final_evaluation: 'Confronto A/B completato',
          },
          phase_progress: {
            context_setup: 100,
            support: 100,
            counter: 100,
            final_evaluation: 100,
          },
        },
      });
    } catch (error) {
      console.error('Errore DoE automatico:', error);
      const errorMessage = (error && error.message) ? error.message : 'Errore sconosciuto';
      const isManualStop = (
        manualPipelineStopRef.current
        || error?.name === 'AbortError'
        || String(errorMessage).toLowerCase().includes('interrotta manualmente')
      );
      if (isManualStop) {
        setPipelineResult((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            _stream: {
              ...(prev._stream || {}),
              phase_details: {
                ...(prev._stream?.phase_details || {}),
                final_evaluation: 'DoE interrotto manualmente',
              },
            },
          };
        });
      } else {
        setPipelineResult(null);
        setPipelineMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `DoE interrotto: ${errorMessage}.`,
          },
        ]);
      }
    } finally {
      pipelineAbortControllerRef.current = null;
      activePipelineRunIdRef.current = null;
      setIsStoppingPipeline(false);
      setIsLoading(false);
    }
  };

  const handlePipelineSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const claim = input.trim();
    const activePipelineSettings = activeTab === TABS.DOE
      ? { ...pipelineSettings, ...doeSettings }
      : pipelineSettings;
    setClaimMemoryMenuOpen(false);
    manualPipelineStopRef.current = false;
    activePipelineRunIdRef.current = null;
    pipelineAbortControllerRef.current = null;
    setIsStoppingPipeline(false);
    setInput('');
    shouldAutoScrollRef.current = true;
    setIsLoading(true);
    if (isCompletedPipelineRun(pipelineResult)) {
      setPipelineHistory((prev) => [...prev, pipelineResult].slice(-10));
    }
    setPipelineMessages((prev) => [...prev, { role: 'user', content: claim }]);
    setPipelineResult(createLivePipelineResult(claim));

    const updateLivePipeline = (mutator) => {
      setPipelineResult((prev) => {
        if (!prev || prev.error) return prev;
        return mutator(prev);
      });
    };

    const setPhaseStatus = (phaseKey, phaseValue, detail = null) => {
      updateLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phases: {
            ...(prev._stream?.phases || {}),
            [phaseKey]: phaseValue,
          },
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            [phaseKey]:
              detail !== null
                ? detail
                : prev._stream?.phase_details?.[phaseKey] || '',
          },
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            [phaseKey]:
              phaseValue === 'done'
                ? 100
                : phaseValue === 'active'
                  ? Math.max(8, prev._stream?.phase_progress?.[phaseKey] || 0)
                  : prev._stream?.phase_progress?.[phaseKey] || 0,
          },
        },
      }));
    };

    const setPhaseDetail = (phaseKey, detail) => {
      updateLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            [phaseKey]: detail || '',
          },
        },
      }));
    };

    const setPhaseProgress = (phaseKey, value) => {
      updateLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            [phaseKey]: Math.max(0, Math.min(100, value)),
          },
        },
      }));
    };

    const bumpPhaseProgress = (phaseKey, value) => {
      updateLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            [phaseKey]: Math.max(
              prev._stream?.phase_progress?.[phaseKey] || 0,
              Math.max(0, Math.min(100, value)),
            ),
          },
        },
      }));
    };

    const appendPhaseToken = (phase, token, stepNumber = null) => {
      if (!token) return;
      updateLivePipeline((prev) => {
        const next = {
          ...prev,
          _stream: {
            ...(prev._stream || {}),
            phase_progress: { ...(prev._stream?.phase_progress || {}) },
            support_steps: { ...(prev._stream?.support_steps || {}) },
            counter_steps: { ...(prev._stream?.counter_steps || {}) },
            support_max_step: prev._stream?.support_max_step || 0,
            counter_max_step: prev._stream?.counter_max_step || 0,
            support_conclusion_live: prev._stream?.support_conclusion_live || '',
          },
        };
        if (phase === 'support' || phase === 'support_conclusion') {
          next.reasoner = {
            ...(prev.reasoner || {}),
            raw_response: `${prev.reasoner?.raw_response || ''}${token}`,
          };
          if (stepNumber != null && phase === 'support') {
            const prevStepText = next._stream.support_steps[stepNumber] || '';
            next._stream.support_steps[stepNumber] = `${prevStepText}${token}`;
            next._stream.support_max_step = Math.max(
              next._stream.support_max_step,
              Number(stepNumber) || 0,
            );
            const approxProgress = Math.min(
              92,
              Math.max(
                12,
                (next._stream.support_max_step / Math.max(1, activePipelineSettings.chain_max_steps)) * 92,
              ),
            );
            next._stream.phase_progress.support = approxProgress;
          } else if (phase === 'support_conclusion') {
            next._stream.support_conclusion_live = `${next._stream.support_conclusion_live}${token}`;
            next._stream.phase_progress.support = Math.max(
              next._stream.phase_progress.support || 0,
              96,
            );
          }
        } else if (phase === 'counter') {
          next.counter_reasoner = {
            ...(prev.counter_reasoner || {}),
            raw_response: `${prev.counter_reasoner?.raw_response || ''}${token}`,
          };
          if (stepNumber != null) {
            const prevStepText = next._stream.counter_steps[stepNumber] || '';
            next._stream.counter_steps[stepNumber] = `${prevStepText}${token}`;
            next._stream.counter_max_step = Math.max(
              next._stream.counter_max_step,
              Number(stepNumber) || 0,
            );
            const approxProgress = Math.min(
              92,
              Math.max(
                12,
                (next._stream.counter_max_step / Math.max(1, activePipelineSettings.chain_max_steps)) * 92,
              ),
            );
            next._stream.phase_progress.counter = approxProgress;
          }
        }
        return next;
      });
    };

    const resetPhaseStepLive = (phase, stepNumber = null) => {
      if (stepNumber == null) return;
      updateLivePipeline((prev) => {
        const next = {
          ...prev,
          _stream: {
            ...(prev._stream || {}),
            support_steps: { ...(prev._stream?.support_steps || {}) },
            counter_steps: { ...(prev._stream?.counter_steps || {}) },
          },
        };
        if (phase === 'support') {
          next._stream.support_steps[stepNumber] = '';
        } else if (phase === 'counter') {
          next._stream.counter_steps[stepNumber] = '';
        }
        return next;
      });
    };

    const resetReasonerLiveForRefinement = (payload = {}) => {
      updateLivePipeline((prev) => ({
        ...prev,
        reasoner: {
          ...(prev.reasoner || {}),
          raw_response: '',
        },
        _stream: {
          ...(prev._stream || {}),
          support_steps: {},
          support_max_step: 0,
          support_conclusion_live: '',
          reasoner_refinement_active: true,
          reasoner_refinement_meta: payload || {},
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            support: 'Riclassificazione causale e rigenerazione con norme di tassonomia...',
          },
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            support: Math.max(55, Number(prev._stream?.phase_progress?.support || 0)),
          },
        },
      }));
    };

    const completeReasonerRefinementLive = (payload = {}) => {
      updateLivePipeline((prev) => ({
        ...prev,
        _stream: {
          ...(prev._stream || {}),
          reasoner_refinement_active: false,
          reasoner_refinement_meta: {
            ...(prev._stream?.reasoner_refinement_meta || {}),
            ...(payload || {}),
          },
          phase_details: {
            ...(prev._stream?.phase_details || {}),
            support: 'Rigenerazione del Reasoner completata, finalizzazione output...',
          },
          phase_progress: {
            ...(prev._stream?.phase_progress || {}),
            support: Math.max(90, Number(prev._stream?.phase_progress?.support || 0)),
          },
        },
      }));
    };

    const requestBody = JSON.stringify({
      claim,
      include_precedents: activePipelineSettings.include_precedents,
      max_statutes: activePipelineSettings.search_top_k_default,
      max_precedents: activePipelineSettings.precedents_limit_default,
      claim_context_memory_enabled: !!activePipelineSettings.claim_context_memory_enabled,
      claim_context_memory_overwrite:
        !!activePipelineSettings.claim_context_memory_enabled &&
        !!activePipelineSettings.claim_context_memory_overwrite,
      settings: {
        reasoner_model: activePipelineSettings.reasoner_model,
        counter_model: activePipelineSettings.counter_model,
        reasoner_temperature: activePipelineSettings.reasoner_temperature,
        counter_temperature: activePipelineSettings.counter_temperature,
        llm_max_tokens: activePipelineSettings.llm_max_tokens,
        search_min_kept_statutes: activePipelineSettings.search_min_kept_statutes,
        search_use_top_n_libri: activePipelineSettings.search_use_top_n_libri,
        chain_min_steps: activePipelineSettings.chain_min_steps,
        chain_max_steps: activePipelineSettings.chain_max_steps,
        aqa_alpha: activePipelineSettings.aqa_alpha,
        aqa_beta: activePipelineSettings.aqa_beta,
        aqa_gamma: activePipelineSettings.aqa_gamma,
        aqa_min_semantic_overlap: activePipelineSettings.aqa_min_semantic_overlap,
        aqa_min_strength_ratio: activePipelineSettings.aqa_min_strength_ratio,
        aqa_damage_factor: activePipelineSettings.aqa_damage_factor,
        aqa_allow_factual_attacks: activePipelineSettings.aqa_allow_factual_attacks,
        aqa_allow_cross_codice: activePipelineSettings.aqa_allow_cross_codice,
        reasoner_enable_causality: activePipelineSettings.reasoner_enable_causality,
        counter_enable_causality: activePipelineSettings.counter_enable_causality,
        counter_pass_causal_identity: activePipelineSettings.counter_pass_causal_identity,
        counter_pass_taxonomy_attacks: activePipelineSettings.counter_pass_taxonomy_attacks,
        counter_pass_norms: activePipelineSettings.counter_pass_norms,
      },
    });

    const runPipelineStreamAttempt = async () => {
      const controller = new AbortController();
      pipelineAbortControllerRef.current = controller;
      const response = await fetch(`${API_BASE}/pipeline/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody,
        signal: controller.signal,
      });

      if (!response.ok) {
        let errText = 'Errore nella pipeline';
        try {
          const err = await response.json();
          errText = err?.error || errText;
        } catch (_) {
          // ignore json parse failures
        }
        throw new Error(errText);
      }

      if (!response.body) {
        throw new Error('Streaming non disponibile: risposta senza body.');
      }

      const decoder = new TextDecoder('utf-8');
      const reader = response.body.getReader();
      let buffer = '';
      let finalPayload = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          if (!rawEvent.trim()) continue;
          const lines = rawEvent.split('\n');
          let eventName = 'message';
          let dataText = '';

          lines.forEach((line) => {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataText += line.slice(5).trim();
            }
          });

          if (!dataText) continue;

          let payload;
          try {
            payload = JSON.parse(dataText);
          } catch (_) {
            continue;
          }

          if (eventName === 'heartbeat') {
            continue;
          }

          if (eventName === 'run_started') {
            activePipelineRunIdRef.current = payload?.run_id || null;
            continue;
          }

          if (eventName === 'phase') {
            const phaseKey = payload?.phase;
            const phaseStatus = payload?.status;
            const phaseProgress = Number(payload?.progress);
            const phaseDetail = payload?.detail || '';
            const isKnownPhase = PIPELINE_PHASES.some((phase) => phase.key === phaseKey);
            if (!isKnownPhase) continue;
            if (phaseStatus) {
              setPhaseStatus(phaseKey, phaseStatus, phaseDetail || null);
            }
            if (Number.isFinite(phaseProgress)) {
              setPhaseProgress(phaseKey, phaseProgress);
            }
            if (phaseDetail) {
              setPhaseDetail(phaseKey, phaseDetail);
            }
            continue;
          }

          if (eventName === 'status') {
            continue;
          }

          if (eventName === 'reasoner_refinement_started') {
            setPhaseStatus('support', 'active');
            resetReasonerLiveForRefinement(payload || {});
            continue;
          }

          if (eventName === 'reasoner_refinement_completed') {
            setPhaseStatus('support', 'active');
            completeReasonerRefinementLive(payload || {});
            continue;
          }

          if (eventName === 'token') {
            const phase = payload.phase || 'generic';
            if (payload?.action === 'reset_step') {
              resetPhaseStepLive(phase, payload?.step ?? null);
              continue;
            }
            if (phase === 'support' || phase === 'support_conclusion') {
              setPhaseStatus('context_setup', 'done');
              setPhaseStatus('support', 'active');
              if (phase === 'support' && payload?.step != null) {
                setPhaseDetail(
                  'support',
                  `Step ${payload.step}/${pipelineSettings.chain_max_steps}`,
                );
              } else if (phase === 'support_conclusion') {
                setPhaseDetail('support', 'Sintesi e formattazione conclusione');
              }
            }
            if (phase === 'counter') {
              setPhaseStatus('support', 'done');
              setPhaseStatus('counter', 'active');
              if (payload?.step != null) {
                setPhaseDetail(
                  'counter',
                  `Step ${payload.step}/${pipelineSettings.chain_max_steps}`,
                );
              }
            }
            appendPhaseToken(phase, payload.token || '', payload.step ?? null);
            continue;
          }

          if (eventName === 'reasoner_result') {
            setPhaseStatus('support', 'done', 'Argomentazione e ASPIC+ completati');
            setPipelineResult((prev) => ({
              ...(prev || {}),
              reasoner: payload || {},
              _stream: {
                ...(prev?._stream || {}),
                reasoner_refinement_active: false,
              },
            }));
            continue;
          }

          if (eventName === 'retrieval_context') {
            setPipelineResult((prev) => ({
              ...(prev || {}),
              retrieval_context: payload || {
                statutes: [],
                precedents: [],
                memory: {},
              },
            }));
            continue;
          }

          if (eventName === 'counter_result') {
            setPhaseStatus('counter', 'done', 'Contro-argomentazione e ASPIC+ completati');
            setPipelineResult((prev) => ({
              ...(prev || {}),
              counter_reasoner: payload || {},
            }));
            continue;
          }

          if (eventName === 'evaluation_result') {
            setPhaseStatus('final_evaluation', 'done', 'Report finale consolidato');
            setPipelineResult((prev) => ({
              ...(prev || {}),
              evaluation: payload || {},
            }));
            continue;
          }

          if (eventName === 'evaluation_partial') {
            setPhaseStatus('final_evaluation', 'active');
            setPhaseDetail('final_evaluation', 'Aggiornamento valutazione in corso');
            setPipelineResult((prev) => ({
              ...(prev || {}),
              evaluation: mergeEvaluationPartial(prev?.evaluation, payload || {}),
            }));
            continue;
          }

          if (eventName === 'evaluation_status') {
            const stage = payload?.stage || '';
            const stageDetailMap = {
              start: 'Check KB catena principale',
              kb_reasoner_done: 'Check KB catena contraria',
              kb_counter_done: 'Verifica opposizione tra tesi',
              gate_done: 'Gate opposizione completato',
              repair_start: 'Riparazione citazioni/catene in corso',
              repair_done: 'Riparazione completata, avvio AQA',
              aqa_done: 'AQA completata, preparazione report',
              done: 'Valutazione completata',
            };
            if (stageDetailMap[stage]) {
              setPhaseDetail('final_evaluation', stageDetailMap[stage]);
            }
            if (stage === 'done') {
              setPhaseStatus('final_evaluation', 'done');
              setPhaseProgress('final_evaluation', 100);
              continue;
            }
            setPhaseStatus('final_evaluation', 'active');
            const stageProgressMap = {
              start: 12,
              kb_reasoner_done: 46,
              kb_counter_done: 72,
              gate_done: 80,
              repair_start: 84,
              repair_done: 92,
              aqa_done: 98,
            };
            if (stageProgressMap[stage] != null) {
              setPhaseProgress('final_evaluation', stageProgressMap[stage]);
            }
            continue;
          }

          if (eventName === 'evaluation_aqa_progress') {
            const message = payload?.message || 'AQA in corso';
            const relativeProgress = Number(payload?.progress);
            setPhaseStatus('final_evaluation', 'active');
            setPhaseDetail('final_evaluation', message);
            if (Number.isFinite(relativeProgress)) {
              bumpPhaseProgress('final_evaluation', 78 + relativeProgress * 20);
            } else {
              bumpPhaseProgress('final_evaluation', 82);
            }
            continue;
          }

          if (eventName === 'evaluation_citation_check') {
            setPhaseStatus('final_evaluation', 'active');
            const totals = payload?.totals || {};
            const agentLabel = payload?.agent === 'counter_reasoner'
              ? 'Counter'
              : 'Reasoner';
            const processed = Number(totals.processed ?? 0);
            const expected = Number(totals.expected_total ?? 0);
            const detail = expected > 0
              ? `Check KB ${agentLabel}: ${processed}/${expected}`
              : `Check KB ${agentLabel}: ${processed}`;
            setPhaseDetail('final_evaluation', detail);
            setPipelineResult((prev) => {
              if (!prev) return prev;
              const agentKey = payload?.agent;
              const check = payload?.check;
              if (!agentKey || !check) return prev;

              const prevEval = prev.evaluation || {};
              const prevReport = prevEval.consistency_report || {};
              const prevAgentReport = prevReport[agentKey] || {};
              const prevChecks = prevAgentReport.citation_checks || [];
              const checkKey = `${check.source_type || 'x'}::${check.citation || ''}`;
              const existingIdx = prevChecks.findIndex(
                (c) => `${c.source_type || 'x'}::${c.citation || ''}` === checkKey,
              );
              let nextChecks;
              if (existingIdx >= 0) {
                nextChecks = [...prevChecks];
                nextChecks[existingIdx] = check;
              } else {
                nextChecks = [...prevChecks, check];
              }
              const nextAgentReport = {
                ...prevAgentReport,
                citation_checks: nextChecks,
                total_citations: totals.processed ?? prevAgentReport.total_citations ?? 0,
                valid_citations: totals.valid ?? prevAgentReport.valid_citations ?? 0,
                invalid_citations: totals.invalid ?? prevAgentReport.invalid_citations ?? 0,
                text_matches: totals.text_matches ?? prevAgentReport.text_matches ?? 0,
                text_mismatches: totals.text_mismatches ?? prevAgentReport.text_mismatches ?? 0,
                repaired_citations: totals.repaired ?? prevAgentReport.repaired_citations ?? 0,
                dropped_citations: totals.dropped ?? prevAgentReport.dropped_citations ?? 0,
              };
              const nextReport = {
                ...prevReport,
                [agentKey]: nextAgentReport,
              };

              const streamState = prev._stream || {};
              const expectedByAgent = {
                ...(streamState.evaluation_expected_checks_by_agent || {}),
              };
              const expectedFromPayload = Number(totals.expected_total);
              if (Number.isFinite(expectedFromPayload) && expectedFromPayload > 0) {
                expectedByAgent[agentKey] = Math.max(
                  Number(expectedByAgent[agentKey] || 0),
                  expectedFromPayload,
                );
              }

              const reasonerReport = nextReport.reasoner || {};
              const counterReport = nextReport.counter_reasoner || {};
              const reasonerProcessed = Number(
                reasonerReport.total_citations
                ?? reasonerReport.citation_checks?.length
                ?? 0,
              );
              const counterProcessed = Number(
                counterReport.total_citations
                ?? counterReport.citation_checks?.length
                ?? 0,
              );
              const reasonerExpected = Math.max(
                Number(expectedByAgent.reasoner || 0),
                reasonerProcessed,
              );
              const counterExpected = Math.max(
                Number(expectedByAgent.counter_reasoner || 0),
                counterProcessed,
              );
              const currentEvalProgress = Number(streamState.phase_progress?.final_evaluation || 0);
              let preciseEvalProgress = currentEvalProgress;
              if (agentKey === 'reasoner') {
                const ratio = reasonerExpected > 0
                  ? Math.min(1, reasonerProcessed / reasonerExpected)
                  : 0;
                preciseEvalProgress = Math.max(preciseEvalProgress, 14 + ratio * 30);
              } else if (agentKey === 'counter_reasoner') {
                const ratio = counterExpected > 0
                  ? Math.min(1, counterProcessed / counterExpected)
                  : 0;
                preciseEvalProgress = Math.max(preciseEvalProgress, 46 + ratio * 24);
              }
              preciseEvalProgress = Math.min(90, preciseEvalProgress);
              const processedCount = reasonerProcessed + counterProcessed;

              return {
                ...prev,
                evaluation: {
                  ...prevEval,
                  consistency_report: nextReport,
                },
                _stream: {
                  ...streamState,
                  evaluation_checks_processed: processedCount,
                  evaluation_expected_checks_by_agent: expectedByAgent,
                  phase_progress: {
                    ...(streamState.phase_progress || {}),
                    final_evaluation: Math.max(
                      currentEvalProgress,
                      preciseEvalProgress,
                    ),
                  },
                },
              };
            });
            continue;
          }

          if (eventName === 'error') {
            throw new Error(payload.message || 'Errore nella pipeline');
          }

          if (eventName === 'cancelled') {
            throw new Error(payload?.message || 'Esecuzione interrotta manualmente.');
          }

          if (eventName === 'final') {
            finalPayload = payload;
            setPipelineResult((prev) => ({
              ...payload,
              _stream: {
                ...(prev?._stream || {}),
                phases: {
                  context_setup: 'done',
                  support: 'done',
                  counter: 'done',
                  final_evaluation: 'done',
                },
                phase_details: {
                  context_setup: 'Completata',
                  support: 'Completata',
                  counter: 'Completata',
                  final_evaluation: 'Completata',
                },
                phase_progress: {
                  context_setup: 100,
                  support: 100,
                  counter: 100,
                  final_evaluation: 100,
                },
                reasoner_refinement_active: false,
              },
            }));
            continue;
          }
        }
      }

      if (!finalPayload) {
        throw new Error('Streaming interrotto prima del risultato finale.');
      }
      return finalPayload;
    };

    const isRetriableStreamError = (message = '') => {
      const text = String(message || '').toLowerCase();
      return (
        text.includes('failed to fetch')
        || text.includes('networkerror')
        || text.includes('network error')
        || text.includes('load failed')
        || text.includes('already running')
      );
    };

    try {
      const maxRetries = 2;
      let finalPayload = null;

      for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
        if (attempt > 0) {
          setPipelineResult(createLivePipelineResult(claim));
          setPhaseStatus(
            'context_setup',
            'active',
            `Riconnessione stream (${attempt}/${maxRetries})...`,
          );
          setPhaseProgress('context_setup', 6);
        }
        try {
          finalPayload = await runPipelineStreamAttempt();
          break;
        } catch (attemptError) {
          const errorMessage = attemptError?.message || 'Errore sconosciuto';
          if (
            attempt < maxRetries
            && !manualPipelineStopRef.current
            && isRetriableStreamError(errorMessage)
          ) {
            const waitMs = 1200 * (attempt + 1);
            setPhaseStatus('final_evaluation', 'active');
            setPhaseDetail(
              'final_evaluation',
              `Connessione instabile, nuovo tentativo tra ${Math.ceil(waitMs / 1000)}s...`,
            );
            await new Promise((resolve) => setTimeout(resolve, waitMs));
            continue;
          }
          throw attemptError;
        }
      }

      if (!finalPayload) {
        throw new Error('Streaming interrotto prima del risultato finale.');
      }
    } catch (error) {
      console.error('Errore pipeline:', error);
      const errorMessage = (error && error.message) ? error.message : 'Errore sconosciuto';
      const isManualStop = (
        manualPipelineStopRef.current
        || error?.name === 'AbortError'
        || String(errorMessage).toLowerCase().includes('interrotta manualmente')
      );
      if (isManualStop) {
        setPipelineResult((prev) => {
          if (!prev) return prev;
          const previousPhases = prev._stream?.phases || {};
          const normalizedPhases = {};
          Object.entries(previousPhases).forEach(([phaseKey, status]) => {
            normalizedPhases[phaseKey] = status === 'done' ? 'done' : 'pending';
          });
          return {
            ...prev,
            _stream: {
              ...(prev._stream || {}),
              phases: normalizedPhases,
              phase_details: {
                ...(prev._stream?.phase_details || {}),
                final_evaluation: 'Esecuzione interrotta manualmente',
              },
            },
          };
        });
        setPipelineMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: 'Pipeline interrotta manualmente. Puoi correggere i parametri e rilanciare.',
          },
        ]);
      } else {
        setPipelineResult(null);
        setPipelineMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `Pipeline interrotta: ${errorMessage}. Ho eliminato i risultati parziali, puoi riprovare.`,
          },
        ]);
      }
    } finally {
      pipelineAbortControllerRef.current = null;
      activePipelineRunIdRef.current = null;
      setIsStoppingPipeline(false);
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
      case TABS.DOE:
        handleDoeSubmit();
        break;
      default:
        handleSearchSubmit();
    }
  };

  const getPlaceholder = () => {
    switch (activeTab) {
      case TABS.SEARCH:
        return 'Inserisci un claim per eseguire la ricerca: norme e precedenti...';
      case TABS.REASON:
        return 'Inserisci un claim per eseguire il ragionamento: tesi principale...';
      case TABS.PIPELINE:
        return 'Inserisci un claim per eseguire la pipeline completa: retrieval, tesi, controtesi e valutazione...';
      case TABS.DOE:
        return 'Inserisci un claim per eseguire il DoE automatico: run A baseline vs run B treatment...';
      default:
        return 'Scrivi un messaggio...';
    }
  };

  const sanitizeFilename = (value = '') => {
    const normalized = String(value || '')
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^a-zA-Z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 80);
    return normalized || 'claim';
  };

  const buildPipelinePdfFilename = (claim = '', prefix = 'pipeline') => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    return `lexcausa_${prefix}_${sanitizeFilename(claim)}_${timestamp}.pdf`;
  };

  const persistPdfExportArtifact = async ({
    blob,
    claim = '',
    prefix = 'pipeline',
    exportContext = 'pipeline',
    filename = 'lexcausa_export.pdf',
  }) => {
    if (!(blob instanceof Blob) || blob.size === 0) return null;

    const formData = new FormData();
    formData.append('pdf', blob, filename);
    formData.append('claim', claim || '');
    formData.append('prefix', prefix || 'pipeline');
    formData.append('export_context', exportContext || 'pipeline');
    formData.append('client_filename', filename);

    try {
      const response = await fetch(`${API_BASE}/pdf/export`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Persistenza PDF fallita (${response.status})`);
      }
      const data = await response.json();
      if (data?.pdf_file?.relative_path) {
        console.info(`PDF salvato automaticamente in ${data.pdf_file.relative_path}`);
      }
      return data;
    } catch (error) {
      console.error('Errore salvataggio automatico PDF:', error);
      return null;
    }
  };

  const downloadPipelineCardPdf = async ({
    reportData = pipelineResult,
    claim = '',
    key = 'pipeline',
    prefix = 'pipeline',
    exportContext = 'pipeline',
  }) => {
    if (!reportData || exportingPdfKey) return;
    setExportingPdfKey(key);
    let sandbox = null;
    let root = null;
    try {
      const sourceWidth = exportContext === 'doe' ? 1120 : 920;
      const pdfOrientation = exportContext === 'doe' ? 'landscape' : 'portrait';
      sandbox = document.createElement('div');
      sandbox.style.position = 'fixed';
      sandbox.style.left = '-100000px';
      sandbox.style.top = '0';
      sandbox.style.width = `${sourceWidth}px`;
      sandbox.style.background = '#ffffff';
      sandbox.style.zIndex = '-1';
      document.body.appendChild(sandbox);
      root = createRoot(sandbox);
      root.render(renderPdfExportContent({ reportData, exportContext }));

      if (document.fonts?.ready) {
        try {
          await document.fonts.ready;
        } catch (error) {
          console.warn('Font readiness failed during PDF export:', error);
        }
      }
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      await new Promise((resolve) => setTimeout(resolve, exportContext === 'doe' ? 220 : 140));

      const exportElement = sandbox.firstElementChild || sandbox;
      const exportWidth = Math.ceil(exportElement.getBoundingClientRect().width || exportElement.scrollWidth || sourceWidth);
      const exportHeight = Math.max(exportElement.scrollHeight, exportElement.clientHeight);
      const maxCanvasPixels = exportContext === 'doe' ? 30000 : 24000;
      const canvasScale = Math.max(
        exportContext === 'doe' ? 0.5 : 0.55,
        Math.min(exportContext === 'doe' ? 1.2 : 1.35, maxCanvasPixels / Math.max(exportHeight, exportWidth, 1)),
      );
      const pdfFilename = buildPipelinePdfFilename(claim, prefix);
      const html2pdfModule = await import('html2pdf.js/dist/html2pdf.bundle.min.js');
      const html2pdf = html2pdfModule.default || html2pdfModule;
      const worker = html2pdf()
        .set({
          margin: [8, 8, 8, 8],
          filename: pdfFilename,
          image: { type: 'jpeg', quality: 0.98 },
          html2canvas: {
            scale: canvasScale,
            useCORS: true,
            backgroundColor: '#ffffff',
            windowWidth: exportWidth,
            windowHeight: Math.min(exportHeight, 5000),
            scrollX: 0,
            scrollY: 0,
            ignoreElements: (element) =>
              element?.dataset?.pdfIgnore === 'true'
              || element?.classList?.contains('settings-panel')
              || element?.classList?.contains('input-area'),
          },
          jsPDF: { unit: 'mm', format: 'a4', orientation: pdfOrientation },
          pagebreak: {
            mode: ['css', 'legacy'],
            avoid: [
              '.pipeline-section',
              '.subsection',
              '.structured-block',
              '.causality-card',
              '.metagraph-wrapper',
              '.summary-card',
              '.aqa-full-section',
              '.pdf-export-kicker-block',
            ],
          },
        })
        .from(exportElement);

      const pdfBlob = await worker.outputPdf('blob');
      const persistPromise = persistPdfExportArtifact({
        blob: pdfBlob,
        claim,
        prefix,
        exportContext,
        filename: pdfFilename,
      });
      await worker.save();
      await persistPromise;
    } catch (error) {
      console.error('Errore durante export PDF:', error);
    } finally {
      if (root) {
        root.unmount();
      }
      if (sandbox && sandbox.parentNode) {
        sandbox.parentNode.removeChild(sandbox);
      }
      setExportingPdfKey(null);
    }
  };

  const isDoeTab = activeTab === TABS.DOE;
  const doeComparison = pipelineResult?.doe_ab_comparison;
  const doeHasComparison = Boolean(doeComparison);
  const doeHasCompletedComparison = Boolean(doeComparison?.delta);
  const doeBaselineView = doeComparison?.baseline?.view || null;
  const doeTreatmentView = doeComparison?.treatment?.view || null;
  const doeSelectedView = isDoeTab
    ? (doeDisplaySetup === 'baseline' ? doeBaselineView : doeTreatmentView)
    : null;
  const displayedCounterReasoner = isDoeTab
    ? (doeSelectedView?.counter_reasoner || {})
    : (pipelineResult?.counter_reasoner || {});
  const displayedEvaluation = isDoeTab
    ? (doeSelectedView?.evaluation || {})
    : (pipelineResult?.evaluation || {});
  const activeDoeRunKey = (
    doeComparison?.treatment?.status === 'running'
      ? 'treatment'
      : doeComparison?.baseline?.status === 'running'
        ? 'baseline'
        : null
  );
  const formatDoePercent = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    return `${(numeric * 100).toFixed(0)}%`;
  };
  const formatDoeSignedPp = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    const pp = numeric * 100;
    return `${pp >= 0 ? '+' : ''}${pp.toFixed(1)} pp`;
  };
  const formatDoeSignedInt = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    return `${numeric >= 0 ? '+' : ''}${Math.round(numeric)}`;
  };
  const formatDurationHuman = (valueMs, { signed = false } = {}) => {
    const numeric = Number(valueMs);
    if (!Number.isFinite(numeric)) return '—';

    const sign = signed
      ? (numeric > 0 ? '+' : numeric < 0 ? '-' : '')
      : '';
    const absMs = Math.abs(Math.round(numeric));

    if (absMs < 1000) {
      return signed ? '≈ 0 s' : 'meno di 1 s';
    }

    let totalSeconds = Math.round(absMs / 1000);
    const days = Math.floor(totalSeconds / 86400);
    totalSeconds -= days * 86400;
    const hours = Math.floor(totalSeconds / 3600);
    totalSeconds -= hours * 3600;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds - minutes * 60;

    const parts = [];
    if (days > 0) parts.push(`${days} g`);
    if (hours > 0) parts.push(`${hours} h`);
    if (minutes > 0) parts.push(`${minutes} min`);
    if (seconds > 0 || parts.length === 0) parts.push(`${seconds} s`);

    return `${sign}${parts.join(' ')}`;
  };
  const normalizeLiveStepText = (value = '') =>
    value
      .replace(/\s+/g, ' ')
      .replace(/^\s*(STEP|PASSO)\s*\d*\s*:\s*/i, '')
      .trim();

  const mergeEvaluationPartial = (prevEvaluation = {}, partial = {}) => {
    const merged = { ...(prevEvaluation || {}) };
    Object.entries(partial || {}).forEach(([key, value]) => {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        merged[key] = {
          ...(prevEvaluation?.[key] || {}),
          ...value,
        };
      } else {
        merged[key] = value;
      }
    });
    return merged;
  };

  const normalizeSectionText = (value = '') =>
    (value || '')
      .replace(/\n{2,}/g, '\n')
      .replace(/[ \t]{2,}/g, ' ')
      .trim();

  const extractNormCitations = (text = '') => {
    const citations = [];
    const seen = new Set();
    const pattern = /Art\.?\s*\d{1,4}(?:[-/][a-z0-9]+)?\s*(?:c\.\s*[cp]\.)?/gi;
    const matches = text.match(pattern) || [];
    matches.forEach((m) => {
      const normalized = m.replace(/\s+/g, ' ').trim();
      const key = normalized.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        citations.push(normalized);
      }
    });
    return citations;
  };

  const parseStructuredResponse = (rawText = '') => {
    const source = (rawText || '').replace(/\r/g, '').trim();
    if (!source) {
      return {
        premessa: '',
        nesso: '',
        conclusione: '',
        norms: [],
        chainSteps: [],
        unstructured: '',
      };
    }

    const markerRegex = /\*\*\s*(Premessa Alternativa|Premessa|Norma|Nesso Causale Alternativo|Nesso Causale|Conclusione Contraria|Conclusione|Catena di ragionamento)\s*\*\*\s*:?\s*/gi;
    const markers = [];
    let marker;
    while ((marker = markerRegex.exec(source)) !== null) {
      markers.push({
        label: (marker[1] || '').toLowerCase(),
        start: marker.index,
        contentStart: markerRegex.lastIndex,
      });
    }

    if (markers.length === 0) {
      return {
        premessa: '',
        nesso: '',
        conclusione: '',
        norms: extractNormCitations(source),
        chainSteps: [],
        unstructured: source,
      };
    }

    const sections = {
      premessa: '',
      norma: '',
      nesso: '',
      conclusione: '',
      chain: '',
    };
    markers.forEach((item, idx) => {
      const nextStart = idx + 1 < markers.length ? markers[idx + 1].start : source.length;
      const content = normalizeSectionText(source.slice(item.contentStart, nextStart));
      if (!content) return;
      if (item.label.includes('premessa')) {
        sections.premessa = sections.premessa ? `${sections.premessa}\n${content}` : content;
      } else if (item.label.includes('norma')) {
        sections.norma = sections.norma ? `${sections.norma}\n${content}` : content;
      } else if (item.label.includes('nesso causale')) {
        sections.nesso = sections.nesso ? `${sections.nesso}\n${content}` : content;
      } else if (item.label.includes('conclusione')) {
        sections.conclusione = sections.conclusione ? `${sections.conclusione}\n${content}` : content;
      } else if (item.label.includes('catena di ragionamento')) {
        sections.chain = sections.chain ? `${sections.chain}\n${content}` : content;
      }
    });

    let norms = [];
    if (sections.norma) {
      const normLines = sections.norma
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
      norms = normLines
        .filter((line) => line.startsWith('-') || line.startsWith('•'))
        .map((line) => line.replace(/^[-•]\s*/, '').trim())
        .filter(Boolean);
      if (norms.length === 0) {
        norms = extractNormCitations(sections.norma);
      }
    }

    const chainSteps = [];
    if (sections.chain) {
      const stepRegex = /(?:^|\n)\s*\d+\.\s+([\s\S]*?)(?=(?:\n\s*\d+\.|\s*$))/g;
      let sm;
      while ((sm = stepRegex.exec(sections.chain)) !== null) {
        const stepText = normalizeSectionText(sm[1] || '');
        if (stepText) {
          chainSteps.push(stepText);
        }
      }
      if (chainSteps.length === 0) {
        const fallbackLines = sections.chain
          .split('\n')
          .map((line) => line.replace(/^\s*\d+\.\s*/, '').trim())
          .filter(Boolean);
        fallbackLines.forEach((line) => chainSteps.push(line));
      }
    }

    return {
      premessa: sections.premessa,
      nesso: sections.nesso,
      conclusione: sections.conclusione,
      norms,
      chainSteps,
      unstructured: '',
    };
  };

  const parseConsistencySummary = (summaryText = '') => {
    const source = (summaryText || '').replace(/\r/g, '').trim();
    if (!source) {
      return { title: 'Riepilogo', sections: [], notes: [] };
    }
    const lines = source
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);

    const parsed = {
      title: 'Riepilogo',
      sections: [],
      notes: [],
    };

    let currentSection = null;
    lines.forEach((line) => {
      if (line.startsWith('## ')) {
        parsed.title = line.replace(/^##\s*/, '').trim() || parsed.title;
        return;
      }
      if (line.startsWith('### ')) {
        if (currentSection) parsed.sections.push(currentSection);
        currentSection = {
          name: line.replace(/^###\s*/, '').trim(),
          metrics: [],
          freeText: [],
        };
        return;
      }
      if (line.startsWith('- ')) {
        const entry = line.replace(/^-+\s*/, '').trim();
        const sepIndex = entry.indexOf(':');
        if (sepIndex > -1) {
          const label = entry.slice(0, sepIndex).trim();
          const value = entry.slice(sepIndex + 1).trim();
          if (currentSection) {
            currentSection.metrics.push({ label, value });
          } else {
            parsed.notes.push({ label, value });
          }
        } else if (currentSection) {
          currentSection.freeText.push(entry);
        }
        return;
      }
      if (currentSection) {
        currentSection.freeText.push(line);
      } else {
        parsed.notes.push({ label: 'Nota', value: line });
      }
    });

    if (currentSection) parsed.sections.push(currentSection);
    return parsed;
  };

  const getSummaryMetricClass = (label = '') => {
    const normalized = label.toLowerCase();
    if (normalized.includes('problemi')) return 'summary-metric-negative';
    if (normalized.includes('score')) return 'summary-metric-info';
    if (normalized.includes('valide')) return 'summary-metric-positive';
    if (normalized.includes('riparat')) return 'summary-metric-warning';
    return '';
  };

  const renderCausalityCard = (title, causality = {}) => {
    if (!causality || typeof causality !== 'object') return null;
    const rows = [
      { label: 'Tipo Causale', value: causality.causal_type_id || 'Non disponibile' },
      { label: 'Teoria', value: causality.theory_id || 'Non disponibile' },
      ...(causality.domain ? [{ label: 'Dominio', value: causality.domain }] : []),
      ...(causality.source ? [{ label: 'Fonte', value: causality.source }] : []),
    ];
    return (
      <div className="causality-card">
        <div className="causality-card-title">{title}</div>
        <div className="causality-grid">
          {rows.map((row) => (
            <div key={`causality-${title}-${row.label}`} className="causality-item">
              <span className="causality-label">{row.label}</span>
              <span className="causality-value">{row.value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const normalizeAttackLabel = (attackId = '') =>
    ATTACK_LABELS_IT[String(attackId || '').trim()]
    || String(attackId || '')
      .replace(/_/g, ' ')
      .trim();

  const renderCounterAttacksUsed = (counterData = {}, keyPrefix = 'counter-attacks') => {
    if (!counterData || typeof counterData !== 'object') return null;

    const selectedIds = Array.isArray(counterData.selected_attack_ids)
      ? counterData.selected_attack_ids.filter(Boolean)
      : [];
    const primaryId = counterData.selected_attack_id || '';
    const uniqueSelected = selectedIds.length > 0
      ? [...new Set(selectedIds)]
      : (primaryId ? [primaryId] : []);

    const byStepRaw = counterData?.aspic_ir?.metadata?.selected_attack_by_step;
    const byStep = Array.isArray(byStepRaw)
      ? byStepRaw
        .filter((item) => {
          if (!item || typeof item !== 'object') return false;
          if (item.attack_id) return true;
          return Array.isArray(item.attack_ids) && item.attack_ids.some(Boolean);
        })
        .map((item) => ({
          step: Number(item.step || 0),
          attack_ids: (
            Array.isArray(item.attack_ids)
              ? item.attack_ids
              : [item.attack_id]
          )
            .filter(Boolean)
            .map((id) => String(id)),
        }))
        .sort((a, b) => a.step - b.step)
      : [];

    if (uniqueSelected.length === 0 && byStep.length === 0) return null;

    return (
      <div className="subsection counter-attack-usage">
        <h4>Attacchi Utilizzati</h4>
        {uniqueSelected.length > 0 && (
          <div className="counter-attack-chip-row">
            {uniqueSelected.map((attackId, idx) => (
              <span
                key={`${keyPrefix}-selected-${attackId}-${idx}`}
                className="counter-attack-chip"
                title={attackId}
              >
                {normalizeAttackLabel(attackId)}
              </span>
            ))}
          </div>
        )}
        {byStep.length > 0 && (
          <ul className="counter-attack-step-list">
            {byStep.map((item, idx) => (
              <li key={`${keyPrefix}-step-${item.step}-${idx}`}>
                <strong>Step {item.step || idx + 1}</strong>
                <span>{item.attack_ids.map(normalizeAttackLabel).join(', ')}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  };

  const renderAspicCitationPills = (citations = {}, keyPrefix = 'aspic-cit') => {
    if (!citations || typeof citations !== 'object') return null;
    const statutes = Array.isArray(citations.statutes) ? citations.statutes : [];
    const precedents = Array.isArray(citations.precedents) ? citations.precedents : [];
    const unknown = Array.isArray(citations.unknown_statutes) ? citations.unknown_statutes : [];

    if (statutes.length === 0 && precedents.length === 0 && unknown.length === 0) return null;

    return (
      <div className="aspic-citations">
        {statutes.map((item, idx) => (
          <span
            key={`${keyPrefix}-statute-${item?.statute_id || item?.label || idx}`}
            className="aspic-citation-pill aspic-citation-pill-statute"
          >
            {item?.label || (item?.articolo ? `Art. ${item.articolo}` : item?.statute_id || `Norma ${idx + 1}`)}
          </span>
        ))}
        {precedents.map((item, idx) => (
          <span
            key={`${keyPrefix}-precedent-${item?.id || item?.title || idx}`}
            className="aspic-citation-pill aspic-citation-pill-precedent"
          >
            {item?.title || item?.id || `Precedente ${idx + 1}`}
          </span>
        ))}
        {unknown.map((item, idx) => (
          <span
            key={`${keyPrefix}-unknown-${item?.label || item?.article_num || idx}`}
            className="aspic-citation-pill aspic-citation-pill-unknown"
          >
            {item?.label || item?.article_num || `Sconosciuta ${idx + 1}`}
          </span>
        ))}
      </div>
    );
  };

  const renderAspicKeyValueFields = (obj = {}, keyPrefix = 'aspic-fields', excludedKeys = []) => {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
    const hidden = new Set(excludedKeys || []);
    const entries = Object.entries(obj)
      .filter(([key, value]) => !hidden.has(key) && value !== undefined && value !== null && value !== '');
    if (entries.length === 0) return null;

    return (
      <div className="aspic-full-fields">
        {entries.map(([key, value]) => (
          <div key={`${keyPrefix}-${key}`} className="aspic-full-field">
            <span>{key}</span>
            <strong>{typeof value === 'string' ? value : JSON.stringify(value)}</strong>
          </div>
        ))}
      </div>
    );
  };

  const renderAspicFullView = (aspicData = {}, keyPrefix = 'aspic') => {
    if (!aspicData || typeof aspicData !== 'object' || Object.keys(aspicData).length === 0) {
      return <p className="aspic-empty">Nessun contenuto ASPIC+ disponibile.</p>;
    }

    const chain = Array.isArray(aspicData.reasoning_chain) ? aspicData.reasoning_chain : [];
    const argumentsList = Array.isArray(aspicData.arguments) ? aspicData.arguments : [];
    const statutes = Array.isArray(aspicData.sources?.statutes) ? aspicData.sources.statutes : [];
    const precedents = Array.isArray(aspicData.sources?.precedents) ? aspicData.sources.precedents : [];
    const precedentNodes = Array.isArray(aspicData.precedent_nodes) ? aspicData.precedent_nodes : [];
    const precedentLinks = Array.isArray(aspicData.precedent_links) ? aspicData.precedent_links : [];
    const metadata = (aspicData.metadata && typeof aspicData.metadata === 'object') ? aspicData.metadata : {};
    const repairMeta = (aspicData._repair_metadata && typeof aspicData._repair_metadata === 'object')
      ? aspicData._repair_metadata
      : null;
    const coveredKeys = new Set([
      'schema',
      'role',
      'claim',
      'raw_response',
      'reasoning_chain',
      'arguments',
      'sources',
      'precedent_nodes',
      'precedent_links',
      'metadata',
      '_repair_metadata',
    ]);
    const extraEntries = Object.entries(aspicData).filter(([key]) => !coveredKeys.has(key));

    return (
      <div className="aspic-full">
        <div className="aspic-full-kpis">
          <div className="aspic-full-kpi">
            <span>Schema</span>
            <strong>{aspicData.schema || 'aspic_ir'}</strong>
          </div>
          <div className="aspic-full-kpi">
            <span>Ruolo</span>
            <strong>{aspicData.role || '-'}</strong>
          </div>
          <div className="aspic-full-kpi">
            <span>Step Catena</span>
            <strong>{chain.length}</strong>
          </div>
          <div className="aspic-full-kpi">
            <span>Argomenti</span>
            <strong>{argumentsList.length}</strong>
          </div>
          <div className="aspic-full-kpi">
            <span>Norme</span>
            <strong>{statutes.length}</strong>
          </div>
          <div className="aspic-full-kpi">
            <span>Precedenti</span>
            <strong>{precedents.length}</strong>
          </div>
        </div>

        {aspicData.claim && (
          <div className="aspic-full-section">
            <h6>Claim</h6>
            <p className="aspic-full-text">{aspicData.claim}</p>
          </div>
        )}

        {aspicData.raw_response && (
          <div className="aspic-full-section">
            <h6>Raw Response</h6>
            <pre className="aspic-full-pre">{aspicData.raw_response}</pre>
          </div>
        )}

        {chain.length > 0 && (
          <div className="aspic-full-section">
            <h6>Reasoning Chain</h6>
            <div className="aspic-full-stack">
              {chain.map((step, idx) => {
                const stepId = typeof step === 'object' ? (step?.id || `S${idx + 1}`) : `S${idx + 1}`;
                const stepText = typeof step === 'object' ? (step?.text || '') : String(step || '');
                const stepCitations = typeof step === 'object' ? step?.citations : null;
                return (
                  <div key={`${keyPrefix}-chain-${stepId}-${idx}`} className="aspic-full-card">
                    <div className="aspic-full-card-head">
                      <span className="aspic-full-card-id">{stepId}</span>
                    </div>
                    <p className="aspic-full-text">{stepText}</p>
                    {renderAspicCitationPills(stepCitations, `${keyPrefix}-chain-cit-${stepId}-${idx}`)}
                    {typeof step === 'object' && renderAspicKeyValueFields(
                      step,
                      `${keyPrefix}-chain-extra-${stepId}-${idx}`,
                      ['id', 'text', 'citations'],
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {argumentsList.length > 0 && (
          <div className="aspic-full-section">
            <h6>Arguments</h6>
            <div className="aspic-full-stack">
              {argumentsList.map((arg, argIdx) => {
                const premises = Array.isArray(arg?.premises) ? arg.premises : [];
                return (
                  <div key={`${keyPrefix}-arg-${arg?.id || argIdx}`} className="aspic-full-card">
                    <div className="aspic-full-card-head">
                      <span className="aspic-full-card-id">{arg?.id || `A${argIdx + 1}`}</span>
                      <span className="aspic-full-badge">{arg?.role || '-'}</span>
                    </div>

                    {premises.length > 0 && (
                      <div className="aspic-full-subsection">
                        <h6>Premesse</h6>
                        <div className="aspic-full-stack">
                          {premises.map((prem, premIdx) => (
                            <div key={`${keyPrefix}-arg-${argIdx}-prem-${prem?.id || premIdx}`} className="aspic-full-subcard">
                              <div className="aspic-full-card-head">
                                <span className="aspic-full-card-id">{prem?.id || `P${premIdx + 1}`}</span>
                                <span className="aspic-full-badge">{prem?.type || 'premise'}</span>
                              </div>
                              <p className="aspic-full-text">{prem?.text || ''}</p>
                              {renderAspicCitationPills(prem?.citations, `${keyPrefix}-arg-${argIdx}-prem-cit-${premIdx}`)}
                              {renderAspicKeyValueFields(
                                prem,
                                `${keyPrefix}-arg-${argIdx}-prem-extra-${premIdx}`,
                                ['id', 'type', 'text', 'citations'],
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {arg?.rule && (
                      <div className="aspic-full-subsection">
                        <h6>Rule</h6>
                        <div className="aspic-full-subcard">
                          <div className="aspic-full-card-head">
                            <span className="aspic-full-card-id">{arg.rule.id || 'R1'}</span>
                            <span className="aspic-full-badge">{arg.rule.type || 'defeasible'}</span>
                          </div>
                          <p className="aspic-full-text">{arg.rule.text || ''}</p>
                          {renderAspicKeyValueFields(
                            arg.rule,
                            `${keyPrefix}-arg-${argIdx}-rule-extra`,
                            ['id', 'type', 'text'],
                          )}
                        </div>
                      </div>
                    )}

                    {arg?.conclusion && (
                      <div className="aspic-full-subsection">
                        <h6>Conclusion</h6>
                        <div className="aspic-full-subcard">
                          <div className="aspic-full-card-head">
                            <span className="aspic-full-card-id">{arg.conclusion.id || 'C1'}</span>
                          </div>
                          <p className="aspic-full-text">{arg.conclusion.text || ''}</p>
                          {renderAspicCitationPills(arg.conclusion.citations, `${keyPrefix}-arg-${argIdx}-conc-cit`)}
                          {renderAspicKeyValueFields(
                            arg.conclusion,
                            `${keyPrefix}-arg-${argIdx}-conc-extra`,
                            ['id', 'text', 'citations'],
                          )}
                        </div>
                      </div>
                    )}

                    {renderAspicKeyValueFields(
                      arg,
                      `${keyPrefix}-arg-extra-${argIdx}`,
                      ['id', 'role', 'premises', 'rule', 'conclusion'],
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {(statutes.length > 0 || precedents.length > 0) && (
          <div className="aspic-full-section">
            <h6>Sources</h6>
            {statutes.length > 0 && (
              <div className="aspic-full-stack">
                {statutes.map((st, idx) => (
                  <div key={`${keyPrefix}-src-statute-${st?.statute_id || st?.label || idx}`} className="aspic-full-subcard">
                    <div className="aspic-full-card-head">
                      <span className="aspic-full-card-id">{st?.label || st?.statute_id || `Statuto ${idx + 1}`}</span>
                    </div>
                    <p className="aspic-full-text">{st?.title || '-'}</p>
                    {renderAspicKeyValueFields(st, `${keyPrefix}-src-stat-extra-${idx}`, ['label', 'title'])}
                  </div>
                ))}
              </div>
            )}
            {precedents.length > 0 && (
              <div className="aspic-full-stack">
                {precedents.map((pr, idx) => (
                  <div key={`${keyPrefix}-src-prec-${pr?.id || pr?.title || idx}`} className="aspic-full-subcard">
                    <div className="aspic-full-card-head">
                      <span className="aspic-full-card-id">{pr?.title || `Precedente ${idx + 1}`}</span>
                    </div>
                    <p className="aspic-full-text">{pr?.id || '-'}</p>
                    {renderAspicKeyValueFields(pr, `${keyPrefix}-src-prec-extra-${idx}`, ['id', 'title'])}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {(precedentNodes.length > 0 || precedentLinks.length > 0) && (
          <div className="aspic-full-section">
            <h6>Precedent Graph</h6>
            <div className="aspic-full-kpis">
              <div className="aspic-full-kpi">
                <span>Nodes</span>
                <strong>{precedentNodes.length}</strong>
              </div>
              <div className="aspic-full-kpi">
                <span>Links</span>
                <strong>{precedentLinks.length}</strong>
              </div>
            </div>
            {precedentNodes.length > 0 && (
              <div className="aspic-full-stack">
                {precedentNodes.map((node, idx) => (
                  <div key={`${keyPrefix}-prec-node-${node?.id || idx}`} className="aspic-full-subcard">
                    {renderAspicKeyValueFields(node, `${keyPrefix}-prec-node-fields-${idx}`)}
                  </div>
                ))}
              </div>
            )}
            {precedentLinks.length > 0 && (
              <div className="aspic-full-stack">
                {precedentLinks.map((link, idx) => (
                  <div key={`${keyPrefix}-prec-link-${idx}`} className="aspic-full-subcard">
                    {renderAspicKeyValueFields(link, `${keyPrefix}-prec-link-fields-${idx}`)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {(Object.keys(metadata).length > 0 || repairMeta) && (
          <div className="aspic-full-section">
            <h6>Metadata</h6>
            <div className="aspic-full-meta-grid">
              {Object.entries(metadata).map(([k, v]) => (
                <div key={`${keyPrefix}-meta-${k}`} className="aspic-full-meta-item">
                  <span>{k}</span>
                  <strong>{typeof v === 'string' ? v : JSON.stringify(v)}</strong>
                </div>
              ))}
              {repairMeta && Object.entries(repairMeta).map(([k, v]) => (
                <div key={`${keyPrefix}-repair-${k}`} className="aspic-full-meta-item aspic-full-meta-item-repair">
                  <span>{k}</span>
                  <strong>{typeof v === 'string' ? v : JSON.stringify(v)}</strong>
                </div>
              ))}
            </div>
          </div>
        )}

        {extraEntries.length > 0 && (
          <div className="aspic-full-section">
            <h6>Campi Extra</h6>
            <pre className="aspic-full-pre">{JSON.stringify(Object.fromEntries(extraEntries), null, 2)}</pre>
          </div>
        )}
      </div>
    );
  };

  const renderAspicOverview = (title, aspicData = {}, compact = false) => {
    if (!aspicData || typeof aspicData !== 'object' || Object.keys(aspicData).length === 0) {
      return null;
    }

    return (
      <div className={`aspic-overview ${compact ? 'aspic-overview-compact' : ''}`}>
        <details className="ir-toggle aspic-overview-toggle">
          <summary className="aspic-overview-summary">{title}</summary>
          {renderAspicFullView(aspicData, `${title}-full`)}
        </details>
      </div>
    );
  };

  const formatPrettyFieldLabel = (value = '') =>
    String(value || '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());

  const formatPrettyScalar = (value) => {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '-';
    return String(value);
  };

  const renderAqaMetricGrid = (metrics = {}, keyPrefix = 'aqa-metric') => {
    const entries = Object.entries(metrics || {}).filter(([, value]) => value !== undefined && value !== null);
    if (entries.length === 0) return null;
    return (
      <div className="aqa-full-metric-grid">
        {entries.map(([key, value]) => (
          <div key={`${keyPrefix}-${key}`} className="aqa-full-metric">
            <span>{formatPrettyFieldLabel(key)}</span>
            <strong>{typeof value === 'number' ? value.toFixed(4) : formatPrettyScalar(value)}</strong>
          </div>
        ))}
      </div>
    );
  };

  const renderAqaLinksCompact = (title, links = [], keyPrefix = 'aqa-links') => (
    <div className="aqa-full-section">
      <h6>{title}</h6>
      {links.length === 0 ? (
        <p className="aqa-tree-empty">Nessun link.</p>
      ) : (
        <div className="aqa-link-summary-list">
          {links.map((link, idx) => (
            <div key={`${keyPrefix}-${idx}`} className="aqa-link-summary-row">
              <div className="aqa-link-summary-main">
                <strong>{link.link_id || `Link ${idx + 1}`}</strong>
                <span className={`role-tag ${link.role === 'counter' ? 'role-contra' : 'role-pro'}`}>
                  {link.role === 'counter' ? 'C' : 'P'}
                </span>
              </div>
              <div className="aqa-link-summary-metrics">
                <span>Nesso {(link.nesso_plausibility ?? 0).toFixed(3)}</span>
                <span>Base {(link.base_score ?? 0).toFixed(3)}</span>
                <span>Norm {(link.norm_support ?? 0).toFixed(3)}</span>
                <span>Cog {(link.cogency ?? 0).toFixed(3)}</span>
                <span>Sem {(link.semantics ?? 0).toFixed(3)}</span>
                <span>Att {(link.attacks_sum ?? 0).toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderAqaFullView = (aqaData = {}) => {
    if (!aqaData || typeof aqaData !== 'object' || Object.keys(aqaData).length === 0) {
      return <p className="aqa-tree-empty">Nessun dettaglio AQA disponibile.</p>;
    }

    const proLinksCount = Array.isArray(aqaData.links?.pro) ? aqaData.links.pro.length : 0;
    const contraLinksCount = Array.isArray(aqaData.links?.contra) ? aqaData.links.contra.length : 0;
    const proLinks = Array.isArray(aqaData.links?.pro) ? aqaData.links.pro : [];
    const contraLinks = Array.isArray(aqaData.links?.contra) ? aqaData.links.contra : [];
    const weakestLinks = Array.isArray(aqaData.notes?.weakest_links) ? aqaData.notes.weakest_links : [];
    const dominantAttacks = Array.isArray(aqaData.notes?.dominant_attacks) ? aqaData.notes.dominant_attacks : [];
    const precedentSwings = Array.isArray(aqaData.notes?.precedent_swings) ? aqaData.notes.precedent_swings : [];
    const verdict = aqaData.verdict || '-';
    const finalScore = aqaData.net_plausibility?.final;

    return (
      <div className="aqa-full-view">
        <div className="aqa-full-kpis">
          <div className="aqa-full-kpi">
            <span>Verdetto</span>
            <strong>{verdict}</strong>
          </div>
          <div className="aqa-full-kpi">
            <span>Score Finale</span>
            <strong>{typeof finalScore === 'number' ? finalScore.toFixed(4) : '-'}</strong>
          </div>
          <div className="aqa-full-kpi">
            <span>Link Pro</span>
            <strong>{proLinksCount}</strong>
          </div>
          <div className="aqa-full-kpi">
            <span>Link Contro</span>
            <strong>{contraLinksCount}</strong>
          </div>
        </div>

        <div className="aqa-full-section">
          <h6>Pesi AQA</h6>
          {renderAqaMetricGrid(aqaData.weights || {}, 'aqa-weights')}
        </div>

        <div className="aqa-full-section">
          <h6>Plausibilità Netta</h6>
          {renderAqaMetricGrid(aqaData.net_plausibility || {}, 'aqa-net')}
        </div>

        {aqaData.chain_scores && (
          <div className="aqa-full-section">
            <h6>Score Catene</h6>
            <div className="aqa-chain-grid">
              <div className="aqa-chain-card">
                <h6>Pro</h6>
                {renderAqaMetricGrid(aqaData.chain_scores.pro || {}, 'aqa-chain-pro')}
              </div>
              <div className="aqa-chain-card">
                <h6>Contro</h6>
                {renderAqaMetricGrid(aqaData.chain_scores.contra || {}, 'aqa-chain-contra')}
              </div>
            </div>
          </div>
        )}

        {renderAqaLinksCompact('Link Pro (sintesi)', proLinks, 'aqa-pro')}
        {renderAqaLinksCompact('Link Contro (sintesi)', contraLinks, 'aqa-contra')}

        <div className="aqa-full-section">
          <h6>Note Tecniche</h6>
          <div className="aqa-full-metric-grid">
            <div className="aqa-full-metric">
              <span>Attacchi Abilitati</span>
              <strong>{formatPrettyScalar(aqaData.notes?.attacks_enabled)}</strong>
            </div>
            <div className="aqa-full-metric">
              <span>Weakest Links</span>
              <strong>{weakestLinks.length}</strong>
            </div>
            <div className="aqa-full-metric">
              <span>Dominant Attacks</span>
              <strong>{dominantAttacks.length}</strong>
            </div>
            <div className="aqa-full-metric">
              <span>Precedent Swings</span>
              <strong>{precedentSwings.length}</strong>
            </div>
          </div>
          {weakestLinks.length > 0 && (
            <ul className="aqa-compact-list">
              {weakestLinks.slice(0, 8).map((item, idx) => (
                <li key={`aqa-weak-${idx}`}>
                  {item.link_id || `Link ${idx + 1}`} - nesso {(item.nesso_plausibility ?? 0).toFixed(3)}
                </li>
              ))}
            </ul>
          )}
          {dominantAttacks.length > 0 && (
            <ul className="aqa-compact-list">
              {dominantAttacks.slice(0, 8).map((attack, idx) => (
                <li key={`aqa-attack-${idx}`}>
                  {(attack.attacker || '?')} → {(attack.target || '?')} ({(attack.value ?? 0).toFixed(3)})
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    );
  };

  const liveSupportSteps = Object.entries(pipelineResult?._stream?.support_steps || {})
    .map(([k, v]) => [Number(k), normalizeLiveStepText(v)])
    .filter(([_, text]) => Boolean(text))
    .sort((a, b) => a[0] - b[0]);

  const liveCounterSteps = Object.entries(pipelineResult?._stream?.counter_steps || {})
    .map(([k, v]) => [Number(k), normalizeLiveStepText(v)])
    .filter(([_, text]) => Boolean(text))
    .sort((a, b) => a[0] - b[0]);
  const liveSupportStepTexts = liveSupportSteps.map(([, stepText]) => stepText);
  const liveCounterStepTexts = liveCounterSteps.map(([, stepText]) => stepText);

  const liveCounterPhaseActive = pipelineResult?._stream?.phases?.counter === 'active';
  const selectedCounterLiveMode = isDoeTab
    ? (doeDisplaySetup === activeDoeRunKey && liveCounterPhaseActive)
    : liveCounterPhaseActive;
  const selectedCounterLiveSteps = selectedCounterLiveMode ? liveCounterStepTexts : [];
  const selectedCounterLiveBadgeLabel = isDoeTab
    ? (doeDisplaySetup === 'baseline' ? 'Setup A' : 'Setup B')
    : '';
  const selectedCounterLiveBadgeTone = doeDisplaySetup === 'baseline' ? 'baseline' : 'treatment';

  const aqaReport = displayedEvaluation?.aqa_report;
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

  const reasonerParsedResponse = parseStructuredResponse(pipelineResult?.reasoner?.raw_response || '');
  const counterParsedResponse = parseStructuredResponse(displayedCounterReasoner?.raw_response || '');
  const doeBaselineCounterParsed = parseStructuredResponse(
    doeBaselineView?.counter_reasoner?.raw_response || '',
  );
  const doeTreatmentCounterParsed = parseStructuredResponse(
    doeTreatmentView?.counter_reasoner?.raw_response || '',
  );
  const repairedReasonerParsedResponse = parseStructuredResponse(
    displayedEvaluation?.repaired_reasoner_chain || '',
  );
  const repairedCounterParsedResponse = parseStructuredResponse(
    displayedEvaluation?.repaired_counter_chain || '',
  );
  const parsedSummary = parseConsistencySummary(displayedEvaluation?.summary || '');
  const reasonerLiveConclusion = normalizeSectionText(pipelineResult?._stream?.support_conclusion_live || '');
  const reasonerRepairStats = displayedEvaluation?.consistency_report?.reasoner || {};
  const counterRepairStats = displayedEvaluation?.consistency_report?.counter_reasoner || {};
  const hasReasonerRepairs = Number(reasonerRepairStats.repaired_citations || 0) > 0
    || Number(reasonerRepairStats.dropped_citations || 0) > 0;
  const hasCounterRepairs = Number(counterRepairStats.repaired_citations || 0) > 0
    || Number(counterRepairStats.dropped_citations || 0) > 0;
  const hasAnyRepairs = hasReasonerRepairs || hasCounterRepairs;
  const evaluationConsistencyReport = displayedEvaluation?.consistency_report || {};
  const evaluationReasonerReport = evaluationConsistencyReport.reasoner;
  const evaluationCounterReport = evaluationConsistencyReport.counter_reasoner;
  const evaluationPhaseStatus = pipelineResult?._stream?.phases?.final_evaluation || 'pending';
  const evaluationPhaseActive = evaluationPhaseStatus === 'active';
  const evaluationPhaseProgress = Math.round(
    Number(pipelineResult?._stream?.phase_progress?.final_evaluation || 0),
  );
  const evaluationPhaseDetail = pipelineResult?._stream?.phase_details?.final_evaluation || '';
  const getUniqueNormalizedStrings = (items = []) => {
    const seen = new Set();
    const ordered = [];
    (items || []).forEach((item) => {
      const value = String(item || '').replace(/\s+/g, ' ').trim();
      if (!value) return;
      const key = value.toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      ordered.push(value);
    });
    return ordered;
  };
  const getCounterAttackLabels = (counterData = {}) => {
    const selectedIds = Array.isArray(counterData?.selected_attack_ids)
      ? counterData.selected_attack_ids
      : [];
    const fallbackId = counterData?.selected_attack_id ? [counterData.selected_attack_id] : [];
    return getUniqueNormalizedStrings(
      (selectedIds.length > 0 ? selectedIds : fallbackId).map(normalizeAttackLabel),
    );
  };
  const doeBaselineAttackLabels = getCounterAttackLabels(doeBaselineView?.counter_reasoner);
  const doeTreatmentAttackLabels = getCounterAttackLabels(doeTreatmentView?.counter_reasoner);
  const doeBaselineNorms = getUniqueNormalizedStrings(doeBaselineCounterParsed?.norms || []);
  const doeTreatmentNorms = getUniqueNormalizedStrings(doeTreatmentCounterParsed?.norms || []);
  const doeBaselineOnlyNorms = doeBaselineNorms.filter(
    (norm) => !doeTreatmentNorms.some((item) => item.toLowerCase() === norm.toLowerCase()),
  );
  const doeTreatmentOnlyNorms = doeTreatmentNorms.filter(
    (norm) => !doeBaselineNorms.some((item) => item.toLowerCase() === norm.toLowerCase()),
  );
  const doeHasRenderableComparison = Boolean(
    doeHasCompletedComparison
    || doeBaselineView?.counter_reasoner?.raw_response
    || doeTreatmentView?.counter_reasoner?.raw_response
    || doeBaselineView?.counter_reasoner?.aspic_ir
    || doeTreatmentView?.counter_reasoner?.aspic_ir
    || doeBaselineAttackLabels.length > 0
    || doeTreatmentAttackLabels.length > 0,
  );
  const buildDoeAnalysisSnapshot = (comparison = null, baselineView = null, treatmentView = null) => {
    if (!comparison?.delta) return null;
    const baselineParsed = parseStructuredResponse(
      baselineView?.counter_reasoner?.raw_response || '',
    );
    const treatmentParsed = parseStructuredResponse(
      treatmentView?.counter_reasoner?.raw_response || '',
    );
    const baselineAttackLabels = getCounterAttackLabels(baselineView?.counter_reasoner);
    const treatmentAttackLabels = getCounterAttackLabels(treatmentView?.counter_reasoner);
    const baselineNorms = getUniqueNormalizedStrings(baselineParsed?.norms || []);
    const treatmentNorms = getUniqueNormalizedStrings(treatmentParsed?.norms || []);
    const baselineOnlyNorms = baselineNorms.filter(
      (norm) => !treatmentNorms.some((item) => item.toLowerCase() === norm.toLowerCase()),
    );
    const treatmentOnlyNorms = treatmentNorms.filter(
      (norm) => !baselineNorms.some((item) => item.toLowerCase() === norm.toLowerCase()),
    );
    const deltaFinal = Number(comparison?.delta?.final_score);
    const deltaCounterConsistency = Number(comparison?.delta?.counter_consistency);
    const deltaContraLinks = Number(comparison?.delta?.contra_links);
    const deltaDuration = Number(comparison?.delta?.duration_ms);
    let headline = 'Impatto DoE neutro';
    let toneClass = 'summary-metric-info';
    if (Number.isFinite(deltaFinal) && deltaFinal > 0.01) {
      headline = 'Setup B migliora la forza dialettica del Counter';
      toneClass = 'summary-metric-positive';
    } else if (Number.isFinite(deltaFinal) && deltaFinal < -0.01) {
      headline = 'Setup A resta piu incisivo sul claim corrente';
      toneClass = 'summary-metric-warning';
    }
    const notes = [];
    if (Number.isFinite(deltaFinal)) {
      notes.push(
        deltaFinal > 0.01
          ? `Il treatment aumenta l'AQA finale di ${formatDoeSignedPp(deltaFinal)} rispetto al baseline.`
          : deltaFinal < -0.01
            ? `Il baseline mantiene un vantaggio di ${formatDoeSignedPp(deltaFinal)} sull'AQA finale.`
            : 'La differenza di AQA finale tra A e B e quasi nulla su questo claim.',
      );
    }
    if (treatmentAttackLabels.length > baselineAttackLabels.length) {
      notes.push(
        `Setup B usa attacchi tassonomici espliciti (${treatmentAttackLabels.join(', ')}) mentre Setup A resta piu attack-agnostic.`,
      );
    } else if (baselineAttackLabels.length > 0) {
      notes.push(`Setup A espone attacchi espliciti: ${baselineAttackLabels.join(', ')}.`);
    } else {
      notes.push('Setup A non espone attacchi tassonomici espliciti nella risposta finale.');
    }
    if (baselineOnlyNorms.length > 0 || treatmentOnlyNorms.length > 0) {
      notes.push(
        `Le norme distintive cambiano: A-only ${baselineOnlyNorms.join(', ') || 'nessuna'} | B-only ${treatmentOnlyNorms.join(', ') || 'nessuna'}.`,
      );
    }
    if (Number.isFinite(deltaContraLinks) && deltaContraLinks !== 0) {
      notes.push(
        `Il numero di link contro varia di ${formatDoeSignedInt(deltaContraLinks)} tra treatment e baseline.`,
      );
    }
    if (Number.isFinite(deltaCounterConsistency) && Math.abs(deltaCounterConsistency) >= 0.001) {
      notes.push(
        `La consistenza del Counter cambia di ${formatDoeSignedPp(deltaCounterConsistency)} tra i due setup.`,
      );
    }
    if (Number.isFinite(deltaDuration)) {
      notes.push(
        deltaDuration < 0
          ? `Setup B completa piu rapidamente di ${formatDurationHuman(Math.abs(deltaDuration))}.`
          : deltaDuration > 0
            ? `Setup B impiega ${formatDurationHuman(deltaDuration)} in piu rispetto ad A.`
            : 'I tempi di esecuzione A/B sono equivalenti.',
      );
    }
    return {
      headline,
      toneClass,
      notes,
    };
  };
  const doeAnalysis = doeHasCompletedComparison
    ? buildDoeAnalysisSnapshot(doeComparison, doeBaselineView, doeTreatmentView)
    : null;

  const renderDoeComparisonSection = () => {
    if (!isDoeTab || !doeHasRenderableComparison) return null;
    return (
      <div className="result-section pipeline-section" data-pdf-section="doe-comparison">
        <h3 className="section-header">
          <GitBranch size={20} style={{ color: '#6366f1' }} />
          Confronto DoE Automatico (A/B)
        </h3>
        <div className="summary-cards-grid">
          <div className="summary-card">
            <div className="summary-card-title">
              {doeComparison.baseline?.label || 'Setup A'} - {doeComparison.baseline?.description || 'Baseline'}
            </div>
            <div className="summary-metrics">
              <div className="summary-metric summary-metric-info">
                <span className="summary-metric-label">AQA Verdetto</span>
                <span className="summary-metric-value">{doeComparison.baseline?.metrics?.verdict || 'n/a'}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoePercent(doeComparison.baseline?.metrics?.final_score)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoePercent(doeComparison.baseline?.metrics?.counter_consistency)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(doeComparison.baseline?.duration_ms))
                    ? formatDurationHuman(doeComparison.baseline.duration_ms)
                    : '—'}
                </span>
              </div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-title">
              {doeComparison.treatment?.label || 'Setup B'} - {doeComparison.treatment?.description || 'Treatment'}
            </div>
            <div className="summary-metrics">
              <div className="summary-metric summary-metric-info">
                <span className="summary-metric-label">AQA Verdetto</span>
                <span className="summary-metric-value">{doeComparison.treatment?.metrics?.verdict || 'n/a'}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoePercent(doeComparison.treatment?.metrics?.final_score)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoePercent(doeComparison.treatment?.metrics?.counter_consistency)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(doeComparison.treatment?.duration_ms))
                    ? formatDurationHuman(doeComparison.treatment.duration_ms)
                    : '—'}
                </span>
              </div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-title">Delta (B - A)</div>
            <div className="summary-metrics">
              <div className="summary-metric summary-metric-positive">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoeSignedPp(doeComparison.delta?.final_score)}</span>
              </div>
              <div className="summary-metric summary-metric-info">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoeSignedPp(doeComparison.delta?.counter_consistency)}</span>
              </div>
              <div className="summary-metric summary-metric-warning">
                <span className="summary-metric-label">Link Contro</span>
                <span className="summary-metric-value">{formatDoeSignedInt(doeComparison.delta?.contra_links)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(doeComparison.delta?.duration_ms))
                    ? formatDurationHuman(doeComparison.delta.duration_ms, { signed: true })
                    : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="summary-cards-grid" style={{ marginTop: 12 }}>
          <div className="summary-card">
            <div className="summary-card-title">Counter A (baseline) - vista parallela</div>
            {(doeBaselineView?.counter_reasoner?.raw_response || doeComparison?.baseline?.status === 'done')
              ? renderStructuredResponse({ parsed: doeBaselineCounterParsed })
              : <p className="aqa-tree-empty">In attesa dell&apos;output baseline.</p>}
          </div>
          <div className="summary-card">
            <div className="summary-card-title">Counter B (treatment) - vista parallela</div>
            {(doeTreatmentView?.counter_reasoner?.raw_response || doeComparison?.treatment?.status === 'done')
              ? renderStructuredResponse({ parsed: doeTreatmentCounterParsed })
              : <p className="aqa-tree-empty">Treatment non ancora disponibile o in generazione.</p>}
          </div>
        </div>
      </div>
    );
  };

  const renderDoeAnalysisSection = () => {
    if (!isDoeTab || !doeAnalysis) return null;
    return (
      <div className="result-section pipeline-section" data-pdf-section="doe-analysis">
        <h3 className="section-header">
          <GitBranch size={20} style={{ color: '#2563eb' }} />
          Analisi DoE - Differenze A/B
        </h3>
        <div className="summary-cards-grid">
          <div className="summary-card">
            <div className="summary-card-title">Lettura tecnica</div>
            <div className={`summary-metric ${doeAnalysis.toneClass}`} style={{ marginBottom: 12 }}>
              <span className="summary-metric-label">Sintesi</span>
              <span className="summary-metric-value">{doeAnalysis.headline}</span>
            </div>
            <ul className="summary-notes-list">
              {doeAnalysis.notes.map((note, idx) => (
                <li key={`doe-analysis-note-${idx}`}>{note}</li>
              ))}
            </ul>
          </div>
          <div className="summary-card">
            <div className="summary-card-title">Attacchi e Norme Distintive</div>
            <div className="structured-block" style={{ marginBottom: 10 }}>
              <h5>Attacchi Setup A</h5>
              {doeBaselineAttackLabels.length > 0 ? (
                <ul className="structured-list">
                  {doeBaselineAttackLabels.map((label, idx) => (
                    <li key={`doe-a-attack-${idx}`}>{label}</li>
                  ))}
                </ul>
              ) : (
                <p>Nessun attacco tassonomico esplicito.</p>
              )}
            </div>
            <div className="structured-block" style={{ marginBottom: 10 }}>
              <h5>Attacchi Setup B</h5>
              {doeTreatmentAttackLabels.length > 0 ? (
                <ul className="structured-list">
                  {doeTreatmentAttackLabels.map((label, idx) => (
                    <li key={`doe-b-attack-${idx}`}>{label}</li>
                  ))}
                </ul>
              ) : (
                <p>Nessun attacco tassonomico esplicito.</p>
              )}
            </div>
            <div className="structured-block">
              <h5>Norme solo A / solo B</h5>
              <p>
                <strong>A:</strong> {doeBaselineOnlyNorms.join(', ') || 'nessuna'}
              </p>
              <p>
                <strong>B:</strong> {doeTreatmentOnlyNorms.join(', ') || 'nessuna'}
              </p>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-title">Conclusioni Counter</div>
            <div className="structured-block" style={{ marginBottom: 10 }}>
              <h5>Setup A</h5>
              <p>{doeBaselineCounterParsed?.conclusione || 'Conclusione non disponibile.'}</p>
            </div>
            <div className="structured-block">
              <h5>Setup B</h5>
              <p>{doeTreatmentCounterParsed?.conclusione || 'Conclusione non disponibile.'}</p>
            </div>
            {doeComparison?.artifacts?.doe_log_file?.relative_path && (
              <div className="structured-block doe-log-artifact">
                <h5>Artefatti DoE</h5>
                <p>Log: {doeComparison.artifacts.doe_log_file.relative_path}</p>
                {doeComparison?.artifacts?.doe_report_file?.relative_path && (
                  <p>Report: {doeComparison.artifacts.doe_report_file.relative_path}</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderStructuredResponse = ({
    parsed,
    liveSteps = [],
    liveConclusion = '',
    liveMode = false,
    variant = 'default',
  }) => {
    const hasLiveChain = liveMode && Array.isArray(liveSteps) && liveSteps.length > 0;
    const chainSteps = hasLiveChain
      ? liveSteps
      : (parsed.chainSteps?.length > 0 ? parsed.chainSteps : liveSteps);
    const norms = hasLiveChain
      ? extractNormCitations(`${chainSteps.join(' ')} ${liveConclusion || ''}`)
      : (
        parsed.norms?.length > 0
          ? parsed.norms
          : extractNormCitations(
            `${parsed.premessa || ''} ${parsed.nesso || ''} ${parsed.conclusione || ''} ${chainSteps.join(' ')}`,
          )
      );
    return (
      <div className={`structured-response ${variant === 'repaired' ? 'repaired-chain' : ''}`}>
        {parsed.premessa && (
          <div className="structured-block">
            <h5>Premessa</h5>
            <p>{parsed.premessa}</p>
          </div>
        )}
        {norms.length > 0 && (
          <div className="structured-block">
            <h5>Norme Richiamate</h5>
            <ul className="structured-list">
              {norms.map((norm, idx) => (
                <li key={`norm-${idx}`}>{norm}</li>
              ))}
            </ul>
          </div>
        )}
        {parsed.nesso && (
          <div className="structured-block">
            <h5>Nesso Causale</h5>
            <p>{parsed.nesso}</p>
          </div>
        )}
        {(parsed.conclusione || liveConclusion) && (
          <div className="structured-block">
            <h5>Conclusione</h5>
            <p>{parsed.conclusione || liveConclusion}</p>
          </div>
        )}
        {chainSteps.length > 0 && (
          <div className="structured-block">
            <h5>Catena di Ragionamento</h5>
            <ol className="live-steps-list">
              {chainSteps.map((stepText, idx) => (
                <li key={`chain-step-${idx}`}>
                  <span>{stepText}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
        {!parsed.premessa && !parsed.nesso && !parsed.conclusione && norms.length === 0 && chainSteps.length === 0 && parsed.unstructured && (
          <div className="raw-response">{parsed.unstructured}</div>
        )}
        {liveMode && (
          <div className="structured-live-tag">
            <Loader2 size={14} className="loading-spinner" />
            <span>Aggiornamento live in corso...</span>
          </div>
        )}
      </div>
    );
  };

  const renderPdfStatuteList = (title, statutes = [], keyPrefix = 'pdf-statutes') => {
    if (!Array.isArray(statutes) || statutes.length === 0) return null;
    return (
      <div className="subsection">
        <h4>{title} ({statutes.length})</h4>
        <ul className="articles-list pdf-export-list">
          {statutes.map((art, idx) => (
            <li key={`${keyPrefix}-${art?.articolo || art?.statute_id || idx}`}>
              <strong>{idx + 1}. Art. {art?.articolo || art?.statute_id || '-'}</strong>
              {art?.source && ` (${sourceShortLabel(art.source)})`}
              {art?.titolo && ` - ${art.titolo}`}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderPdfPrecedentList = (title, precedents = [], keyPrefix = 'pdf-precedents') => {
    if (!Array.isArray(precedents) || precedents.length === 0) return null;
    return (
      <div className="subsection">
        <h4>{title} ({precedents.length})</h4>
        <ul className="articles-list pdf-export-list">
          {precedents.map((prec, idx) => (
            <li key={`${keyPrefix}-${prec?.id || prec?.title || idx}`}>
              <strong>{idx + 1}. {prec?.title || `Precedente ${idx + 1}`}</strong>
              {prec?.id ? ` - ${prec.id}` : ''}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderPdfSummaryCards = (summaryText = '') => {
    if (!summaryText) return null;
    const summary = parseConsistencySummary(summaryText);
    if (summary.sections.length === 0) {
      return (
        <div className="summary-card">
          <div className="raw-response">{summaryText}</div>
        </div>
      );
    }
    return (
      <div className="summary-cards-grid">
        {summary.sections.map((section, idx) => (
          <div key={`pdf-summary-${section.name}-${idx}`} className="summary-card">
            <div className="summary-card-title">{section.name}</div>
            {section.metrics.length > 0 && (
              <div className="summary-metrics">
                {section.metrics.map((metric, metricIdx) => (
                  <div
                    key={`pdf-summary-metric-${section.name}-${metric.label}-${metricIdx}`}
                    className={`summary-metric ${getSummaryMetricClass(metric.label)}`}
                  >
                    <span className="summary-metric-label">{metric.label}</span>
                    <span className="summary-metric-value">{metric.value}</span>
                  </div>
                ))}
              </div>
            )}
            {section.freeText.length > 0 && (
              <ul className="summary-notes-list">
                {section.freeText.map((note, noteIdx) => (
                  <li key={`pdf-summary-note-${section.name}-${noteIdx}`}>{note}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderPdfConsistencyAgentCard = (title, report = {}, keyPrefix = 'pdf-consistency') => {
    if (!report || typeof report !== 'object' || Object.keys(report).length === 0) return null;
    const totalCitations = Number(report.total_citations || 0);
    const totalTextChecks = Number(report.text_matches || 0) + Number(report.text_mismatches || 0);
    return (
      <div className="summary-card">
        <div className="summary-card-title">{title}</div>
        <div className="summary-metrics">
          <div className="summary-metric summary-metric-positive">
            <span className="summary-metric-label">Score</span>
            <span className="summary-metric-value">
              {formatDoePercent(report.consistency_score)}
            </span>
          </div>
          <div className="summary-metric">
            <span className="summary-metric-label">Citazioni valide</span>
            <span className="summary-metric-value">
              {Number(report.valid_citations || 0)}/{totalCitations}
            </span>
          </div>
          <div className="summary-metric">
            <span className="summary-metric-label">Testo verificato</span>
            <span className="summary-metric-value">
              {Number(report.text_matches || 0)}/{totalTextChecks}
            </span>
          </div>
          {(Number(report.repaired_citations || 0) > 0 || Number(report.dropped_citations || 0) > 0) && (
            <div className="summary-metric summary-metric-warning">
              <span className="summary-metric-label">Riparazioni</span>
              <span className="summary-metric-value">
                {Number(report.repaired_citations || 0)} rip., {Number(report.dropped_citations || 0)} scart.
              </span>
            </div>
          )}
        </div>
        {Array.isArray(report.issues) && report.issues.length > 0 && (
          <ul className="summary-notes-list" key={`${keyPrefix}-issues`}>
            {report.issues.map((issue, idx) => (
              <li key={`${keyPrefix}-issue-${idx}`}>{issue}</li>
            ))}
          </ul>
        )}
      </div>
    );
  };

  const renderPdfEvaluationSection = (evaluation = {}, keyPrefix = 'pdf-evaluation') => {
    const consistencyReport = evaluation?.consistency_report || {};
    const summaryText = evaluation?.summary || '';
    const hasConsistency = Boolean(consistencyReport.reasoner || consistencyReport.counter_reasoner);
    if (!hasConsistency && !summaryText) return null;
    return (
      <div className="subsection">
        <h4>Evaluator - Verifica Consistenza</h4>
        {hasConsistency && (
          <div className="summary-cards-grid">
            {renderPdfConsistencyAgentCard(
              'Reasoner',
              consistencyReport.reasoner,
              `${keyPrefix}-reasoner`,
            )}
            {renderPdfConsistencyAgentCard(
              'Counter-Reasoner',
              consistencyReport.counter_reasoner,
              `${keyPrefix}-counter`,
            )}
          </div>
        )}
        {summaryText && renderPdfSummaryCards(summaryText)}
      </div>
    );
  };

  const renderPdfRepairSection = (evaluation = {}) => {
    const repairedReasonerChain = evaluation?.repaired_reasoner_chain || '';
    const repairedCounterChain = evaluation?.repaired_counter_chain || '';
    const consistencyReport = evaluation?.consistency_report || {};
    const hasRepairs = Boolean(repairedReasonerChain || repairedCounterChain);
    const hasRepairStats = Boolean(consistencyReport.reasoner || consistencyReport.counter_reasoner);
    if (!hasRepairs && !hasRepairStats) return null;
    return (
      <div className="subsection">
        <h4>Catene di Ragionamento Riparate</h4>
        {repairedReasonerChain && (
          <div className="structured-block">
            <h5>Reasoner - Catena Riparata</h5>
            {renderStructuredResponse({
              parsed: parseStructuredResponse(repairedReasonerChain),
              variant: 'repaired',
            })}
          </div>
        )}
        {repairedCounterChain && (
          <div className="structured-block">
            <h5>Counter-Reasoner - Catena Riparata</h5>
            {renderStructuredResponse({
              parsed: parseStructuredResponse(repairedCounterChain),
              variant: 'repaired',
            })}
          </div>
        )}
        {hasRepairStats && (
          <div className="consistency-stats">
            {consistencyReport.reasoner && (
              <span className="stat-item stat-repaired">
                🔧 Reasoner: {consistencyReport.reasoner.repaired_citations || 0} riparate, {consistencyReport.reasoner.dropped_citations || 0} scartate
              </span>
            )}
            {consistencyReport.counter_reasoner && (
              <span className="stat-item stat-repaired">
                🔧 Counter: {consistencyReport.counter_reasoner.repaired_citations || 0} riparate, {consistencyReport.counter_reasoner.dropped_citations || 0} scartate
              </span>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderPdfAqaSection = (evaluation = {}) => {
    const aqaData = evaluation?.aqa_report;
    const proLinks = Array.isArray(aqaData?.links?.pro) ? aqaData.links.pro : [];
    const contraLinks = Array.isArray(aqaData?.links?.contra) ? aqaData.links.contra : [];
    if (!aqaData || typeof aqaData !== 'object' || Object.keys(aqaData).length === 0) return null;
    return (
      <>
        <div className="subsection">
          <h4>AQA - Valutazione Argomentativa</h4>
          {renderAqaFullView(aqaData)}
        </div>
        {aqaData.enabled && (proLinks.length > 0 || contraLinks.length > 0) && (
          <div className="subsection pdf-export-metagraph-subsection">
            <h4>ASPIC+ Metagrafo</h4>
            <AspicMetagraph
              aqaReport={aqaData}
              reasonerIr={evaluation?.repaired_reasoner_aspic_ir}
              counterIr={evaluation?.repaired_counter_aspic_ir}
            />
          </div>
        )}
      </>
    );
  };

  const renderPdfSharedReasonerSection = (reportData = {}) => {
    const reasoner = reportData?.reasoner || {};
    const parsedReasoner = parseStructuredResponse(reasoner?.raw_response || '');
    const hasReasonerContent = Boolean(
      reasoner?.raw_response
      || reasoner?.causality
      || reasoner?.statutes?.length
      || reasoner?.precedents?.length,
    );
    if (!hasReasonerContent) return null;
    return (
      <div className="result-section pipeline-section">
        <h3 className="section-header">
          <CheckCircle2 size={20} style={{ color: '#10b981' }} />
          2. REASONER CONDIVISO
        </h3>
        <p className="pdf-export-context-note">
          La tesi principale viene generata una sola volta e riusata da Setup A e Setup B.
        </p>
        {reasoner?.causality && (
          <div className="subsection">
            <h4>Classificazione Causalità</h4>
            {renderCausalityCard('Mappatura Causale', reasoner.causality)}
          </div>
        )}
        {renderPdfStatuteList('Articoli Trovati (Reasoner)', reasoner?.statutes, 'pdf-reasoner-statutes')}
        {renderPdfPrecedentList('Precedenti Trovati (Reasoner)', reasoner?.precedents, 'pdf-reasoner-precedents')}
        {reasoner?.raw_response && (
          <div className="subsection">
            <h4>Risposta Completa</h4>
            {renderStructuredResponse({ parsed: parsedReasoner })}
          </div>
        )}
      </div>
    );
  };

  const renderPdfDoeSetupSection = ({
    stepNumber,
    setupKey,
    comparisonEntry = {},
    view = {},
  }) => {
    const counterData = view?.counter_reasoner || {};
    const evaluation = view?.evaluation || {};
    const parsedCounter = parseStructuredResponse(counterData?.raw_response || '');
    const title = `${stepNumber}. SETUP ${setupKey} - ${comparisonEntry?.label || setupKey}`;
    const description = comparisonEntry?.description || '';
    return (
      <div className="result-section pipeline-section pdf-export-setup-section">
        <div className={`pdf-export-kicker-block tone-${setupKey === 'A' ? 'baseline' : 'treatment'}`}>
          <span>Setup {setupKey}</span>
          <strong>{description || 'Counter-Reasoner'}</strong>
        </div>
        <h3 className="section-header">
          <XCircle size={20} style={{ color: setupKey === 'A' ? '#ef4444' : '#2563eb' }} />
          {title}
        </h3>
        <div className="summary-cards-grid">
          <div className="summary-card">
            <div className="summary-card-title">Metriche Setup {setupKey}</div>
            <div className="summary-metrics">
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Verdetto</span>
                <span className="summary-metric-value">{comparisonEntry?.metrics?.verdict || 'n/a'}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoePercent(comparisonEntry?.metrics?.final_score)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoePercent(comparisonEntry?.metrics?.counter_consistency)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(comparisonEntry?.duration_ms))
                    ? formatDurationHuman(comparisonEntry.duration_ms)
                    : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
        {counterData?.reasoner_causality && (
          <div className="subsection">
            <h4>Causalità del Reasoner (da Attaccare)</h4>
            {renderCausalityCard('Target Causale da Attaccare', counterData.reasoner_causality)}
          </div>
        )}
        {renderCounterAttacksUsed(counterData, `pdf-counter-attacks-${setupKey}`)}
        {renderPdfStatuteList(
          'Articoli Trovati (Counter-Reasoner)',
          counterData?.statutes,
          `pdf-counter-statutes-${setupKey}`,
        )}
        {renderPdfPrecedentList(
          'Precedenti Trovati (Counter-Reasoner)',
          counterData?.precedents,
          `pdf-counter-precedents-${setupKey}`,
        )}
        {counterData?.raw_response && (
          <div className="subsection">
            <h4>Controtesi</h4>
            {renderStructuredResponse({ parsed: parsedCounter })}
          </div>
        )}
        {renderPdfEvaluationSection(evaluation, `pdf-eval-${setupKey}`)}
        {renderPdfRepairSection(evaluation)}
        {renderPdfAqaSection(evaluation)}
      </div>
    );
  };

  const renderPdfDoeComparisonSection = (comparison = {}, baselineView = {}, treatmentView = {}) => {
    if (!comparison || typeof comparison !== 'object') return null;
    const baselineCounterParsed = parseStructuredResponse(
      baselineView?.counter_reasoner?.raw_response || '',
    );
    const treatmentCounterParsed = parseStructuredResponse(
      treatmentView?.counter_reasoner?.raw_response || '',
    );
    return (
      <div className="result-section pipeline-section">
        <h3 className="section-header">
          <GitBranch size={20} style={{ color: '#6366f1' }} />
          5. CONFRONTO DOE AUTOMATICO (A/B)
        </h3>
        <div className="summary-cards-grid">
          <div className="summary-card">
            <div className="summary-card-title">
              {comparison?.baseline?.label || 'Setup A'} - {comparison?.baseline?.description || 'Baseline'}
            </div>
            <div className="summary-metrics">
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoePercent(comparison?.baseline?.metrics?.final_score)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoePercent(comparison?.baseline?.metrics?.counter_consistency)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(comparison?.baseline?.duration_ms))
                    ? formatDurationHuman(comparison.baseline.duration_ms)
                    : '—'}
                </span>
              </div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-title">
              {comparison?.treatment?.label || 'Setup B'} - {comparison?.treatment?.description || 'Treatment'}
            </div>
            <div className="summary-metrics">
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoePercent(comparison?.treatment?.metrics?.final_score)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoePercent(comparison?.treatment?.metrics?.counter_consistency)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(comparison?.treatment?.duration_ms))
                    ? formatDurationHuman(comparison.treatment.duration_ms)
                    : '—'}
                </span>
              </div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-title">Delta (B - A)</div>
            <div className="summary-metrics">
              <div className="summary-metric">
                <span className="summary-metric-label">AQA Finale</span>
                <span className="summary-metric-value">{formatDoeSignedPp(comparison?.delta?.final_score)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Consistenza Counter</span>
                <span className="summary-metric-value">{formatDoeSignedPp(comparison?.delta?.counter_consistency)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Link Contro</span>
                <span className="summary-metric-value">{formatDoeSignedInt(comparison?.delta?.contra_links)}</span>
              </div>
              <div className="summary-metric">
                <span className="summary-metric-label">Durata</span>
                <span className="summary-metric-value">
                  {Number.isFinite(Number(comparison?.delta?.duration_ms))
                    ? formatDurationHuman(comparison.delta.duration_ms, { signed: true })
                    : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
        <div className="summary-cards-grid">
          <div className="summary-card">
            <div className="summary-card-title">Controtesi A - Sintesi</div>
            {baselineView?.counter_reasoner?.raw_response
              ? renderStructuredResponse({ parsed: baselineCounterParsed })
              : <p className="aqa-tree-empty">Output baseline non disponibile.</p>}
          </div>
          <div className="summary-card">
            <div className="summary-card-title">Controtesi B - Sintesi</div>
            {treatmentView?.counter_reasoner?.raw_response
              ? renderStructuredResponse({ parsed: treatmentCounterParsed })
              : <p className="aqa-tree-empty">Output treatment non disponibile.</p>}
          </div>
        </div>
      </div>
    );
  };

  const renderPdfDoeAnalysisSection = (comparison = {}, baselineView = {}, treatmentView = {}) => {
    const analysis = buildDoeAnalysisSnapshot(comparison, baselineView, treatmentView);
    if (!analysis) return null;
    return (
      <div className="result-section pipeline-section">
        <h3 className="section-header">
          <GitBranch size={20} style={{ color: '#2563eb' }} />
          6. ANALISI DOE - DIFFERENZE A/B
        </h3>
        <div className="summary-cards-grid">
          <div className="summary-card">
            <div className="summary-card-title">Lettura tecnica</div>
            <div className={`summary-metric ${analysis.toneClass}`} style={{ marginBottom: 12 }}>
              <span className="summary-metric-label">Sintesi</span>
              <span className="summary-metric-value">{analysis.headline}</span>
            </div>
            <ul className="summary-notes-list">
              {analysis.notes.map((note, idx) => (
                <li key={`pdf-doe-analysis-${idx}`}>{note}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  const renderPdfPipelineHeader = (exportContext = 'pipeline', reportData = {}) => {
    const isDoeExport = exportContext === 'doe' && reportData?.doe_ab_comparison;
    const exportLabel = isDoeExport ? 'Esportazione DoE' : 'Esportazione Pipeline';
    const exportTitle = isDoeExport
      ? 'Report Completo DoE + Pipeline'
      : 'Report Pipeline Completa';
    const exportSubtitle = isDoeExport
      ? 'Il PDF include pipeline condivisa, Setup A, Setup B, confronto finale DoE e metagrafi dei due Counter-Reasoner.'
      : 'Il PDF include retrieval, Reasoner, Counter-Reasoner, Evaluator e metagrafo finale.';
    return (
      <div className="pdf-export-header">
        <div className="pdf-export-kicker">{exportLabel}</div>
        <h2 className="pdf-export-heading">{exportTitle}</h2>
        <p className="pdf-export-subtitle">{exportSubtitle}</p>
      </div>
    );
  };

  const renderPdfPipelineExportContent = (reportData = {}) => {
    const counterData = reportData?.counter_reasoner || {};
    const evaluation = reportData?.evaluation || {};
    const parsedCounter = parseStructuredResponse(counterData?.raw_response || '');
    return (
      <div className="pdf-export-shell pdf-export-root pdf-export-context-pipeline">
        {renderPdfPipelineHeader('pipeline', reportData)}
        <div className="result-section pipeline-section">
          <h3 className="section-header">
            <FileText size={20} style={{ color: '#111827' }} />
            Claim Analizzato
          </h3>
          <p>{reportData?.claim || '—'}</p>
        </div>
        {(reportData?.retrieval_context?.statutes?.length > 0 || reportData?.retrieval_context?.precedents?.length > 0) && (
          <div className="result-section pipeline-section">
            <h3 className="section-header">
              <CheckCircle2 size={20} style={{ color: '#64748b' }} />
              1. CONTESTO PRE-RETRIEVAL
            </h3>
            <p className="pdf-export-context-note">
              Knowledge base condivisa
              {reportData?.retrieval_context?.memory?.enabled
                ? reportData?.retrieval_context?.memory?.hit
                  ? ' da memoria cache.'
                  : ' da retrieval live.'
                : '.'}
            </p>
            {renderPdfStatuteList('Articoli', reportData?.retrieval_context?.statutes, 'pdf-pipeline-ctx-statutes')}
            {renderPdfPrecedentList('Precedenti', reportData?.retrieval_context?.precedents, 'pdf-pipeline-ctx-precedents')}
          </div>
        )}
        {renderPdfSharedReasonerSection(reportData)}
        <div className="result-section pipeline-section">
          <h3 className="section-header">
            <XCircle size={20} style={{ color: '#ef4444' }} />
            3. COUNTER-REASONER
          </h3>
          {counterData?.reasoner_causality && (
            <div className="subsection">
              <h4>Causalità del Reasoner (da Attaccare)</h4>
              {renderCausalityCard('Target Causale da Attaccare', counterData.reasoner_causality)}
            </div>
          )}
          {renderCounterAttacksUsed(counterData, 'pdf-single-counter-attacks')}
          {renderPdfStatuteList('Articoli Trovati (Counter-Reasoner)', counterData?.statutes, 'pdf-single-counter-statutes')}
          {renderPdfPrecedentList('Precedenti Trovati (Counter-Reasoner)', counterData?.precedents, 'pdf-single-counter-precedents')}
          {counterData?.raw_response && (
            <div className="subsection">
              <h4>Risposta Completa</h4>
              {renderStructuredResponse({ parsed: parsedCounter })}
            </div>
          )}
          {renderPdfEvaluationSection(evaluation, 'pdf-single-eval')}
          {renderPdfRepairSection(evaluation)}
          {renderPdfAqaSection(evaluation)}
        </div>
      </div>
    );
  };

  const renderPdfDoeExportContent = (reportData = {}) => {
    const comparison = reportData?.doe_ab_comparison || {};
    const baselineView = comparison?.baseline?.view || {};
    const treatmentView = comparison?.treatment?.view || {};
    return (
      <div className="pdf-export-shell pdf-export-root pdf-export-context-doe">
        {renderPdfPipelineHeader('doe', reportData)}
        <div className="result-section pipeline-section">
          <h3 className="section-header">
            <GitBranch size={20} style={{ color: '#111827' }} />
            Claim Analizzato
          </h3>
          <p>{reportData?.claim || '—'}</p>
        </div>
        {(reportData?.retrieval_context?.statutes?.length > 0 || reportData?.retrieval_context?.precedents?.length > 0) && (
          <div className="result-section pipeline-section">
            <h3 className="section-header">
              <CheckCircle2 size={20} style={{ color: '#64748b' }} />
              1. CONTESTO PRE-RETRIEVAL
            </h3>
            <p className="pdf-export-context-note">
              Knowledge base condivisa tra Setup A e Setup B
              {reportData?.retrieval_context?.memory?.enabled
                ? reportData?.retrieval_context?.memory?.hit
                  ? ', recuperata da memoria cache.'
                  : ', recuperata live.'
                : '.'}
            </p>
            {renderPdfStatuteList('Articoli', reportData?.retrieval_context?.statutes, 'pdf-doe-ctx-statutes')}
            {renderPdfPrecedentList('Precedenti', reportData?.retrieval_context?.precedents, 'pdf-doe-ctx-precedents')}
          </div>
        )}
        {renderPdfSharedReasonerSection(reportData)}
        {renderPdfDoeSetupSection({
          stepNumber: 3,
          setupKey: 'A',
          comparisonEntry: comparison?.baseline,
          view: baselineView,
        })}
        {renderPdfDoeSetupSection({
          stepNumber: 4,
          setupKey: 'B',
          comparisonEntry: comparison?.treatment,
          view: treatmentView,
        })}
        {renderPdfDoeComparisonSection(comparison, baselineView, treatmentView)}
        {renderPdfDoeAnalysisSection(comparison, baselineView, treatmentView)}
      </div>
    );
  };

  const renderPdfExportContent = ({ reportData, exportContext }) => {
    if (exportContext === 'doe' && reportData?.doe_ab_comparison) {
      return renderPdfDoeExportContent(reportData);
    }
    return renderPdfPipelineExportContent(reportData);
  };

  const isPipelineLikeTab = activeTab === TABS.PIPELINE || activeTab === TABS.DOE;
  const isSearchTab = activeTab === TABS.SEARCH;
  const isReasonTab = activeTab === TABS.REASON;
  const isPipelineTab = activeTab === TABS.PIPELINE;
  const searchMessagesVisible = messages.filter(
    (msg, idx) => !(
      idx === 0
      && msg.role === 'assistant'
      && msg.content === TAB_WELCOME_MESSAGES[TABS.SEARCH]
    ),
  );
  const showModelsSection = isReasonTab || isPipelineLikeTab;
  const showRetrievalSection = isSearchTab || isPipelineLikeTab;
  const showCounterModelControls = isPipelineLikeTab;
  const showAdvancedPipelineSections = isPipelineLikeTab;
  const showTaxonomySection = isPipelineTab;
  const settingsPanelTitle = isSearchTab
    ? 'Impostazioni Ricerca'
    : isDoeTab
      ? 'Impostazioni DoE'
      : isReasonTab
        ? 'Impostazioni Ragionamento'
        : 'Impostazioni Pipeline';
  const inlineSettings = activeTab === TABS.DOE ? doeSettings : pipelineSettings;
  const updateInlineSetting = activeTab === TABS.DOE ? updateDoeSetting : updateSetting;

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <div className="header-content">
          <div className="avatar-icon assistant-avatar">
            <Scale size={24} />
          </div>
          <div>
            <h1 className="header-title">LexCausa</h1>
            <p className="header-subtitle">
              Framework per il ragionamento giuridico causale
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs-container">
        <button
          className={`tab-button ${activeTab === TABS.PIPELINE ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.PIPELINE)}
        >
          <FileText size={16} />
          <span>Pipeline Completa</span>
        </button>
        <button
          className={`tab-button ${activeTab === TABS.DOE ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.DOE)}
        >
          <GitBranch size={16} />
          <span>DoE</span>
        </button>
        <button
          className={`tab-button ${activeTab === TABS.REASON ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.REASON)}
        >
          <Brain size={16} />
          <span>Ragionamento</span>
        </button>
        <button
          className={`tab-button ${activeTab === TABS.SEARCH ? 'tab-active' : ''}`}
          onClick={() => setActiveTab(TABS.SEARCH)}
        >
          <Search size={16} />
          <span>Ricerca</span>
        </button>
      </div>

      {/* Settings Panel (collapsible) */}
      <div className="settings-panel">
        <button className="settings-toggle" onClick={() => setSettingsOpen(!settingsOpen)}>
          <Settings size={16} />
          <span>{settingsPanelTitle}</span>
          <span className={`settings-chevron ${settingsOpen ? 'open' : ''}`}>▸</span>
        </button>

        {settingsOpen && (
          <div className="settings-body">
            {showModelsSection && (
              <>
                {/* Models per step */}
                <fieldset className="settings-group">
                  <legend>{isReasonTab ? '🤖 Modello Reasoner' : '🤖 Modelli LLM per Step'}</legend>
                  <div className="settings-row">
                    <label>
                      <span>Reasoner</span>
                      <select
                        value={pipelineSettings.reasoner_model}
                        onChange={(e) => updateSetting('reasoner_model', e.target.value)}
                      >
                        {availableModels.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </label>
                    {showCounterModelControls && (
                      <label>
                        <span>Counter-Reasoner</span>
                        <select
                          value={pipelineSettings.counter_model}
                          onChange={(e) => updateSetting('counter_model', e.target.value)}
                        >
                          {availableModels.map((m) => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                      </label>
                    )}
                  </div>
                </fieldset>

                {/* Temperature & Max Tokens */}
                <fieldset className="settings-group">
                  <legend>🎛️ Parametri LLM</legend>
                  <div className="settings-row">
                    <label>
                      <span>Temperature Reasoner <strong>{pipelineSettings.reasoner_temperature.toFixed(2)}</strong></span>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={pipelineSettings.reasoner_temperature}
                        onChange={(e) => updateSetting('reasoner_temperature', parseFloat(e.target.value))}
                      />
                    </label>
                    {showCounterModelControls && (
                      <label>
                        <span>Temperature Counter <strong>{pipelineSettings.counter_temperature.toFixed(2)}</strong></span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={pipelineSettings.counter_temperature}
                          onChange={(e) => updateSetting('counter_temperature', parseFloat(e.target.value))}
                        />
                      </label>
                    )}
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
              </>
            )}

            {/* Search & Retrieval */}
            {showRetrievalSection && (
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
                    <span>Min Statuti Mantenuti</span>
                    <input
                      type="number"
                      min="1"
                      max="300"
                      step="1"
                      value={pipelineSettings.search_min_kept_statutes}
                      onChange={(e) => updateSetting('search_min_kept_statutes', parseInt(e.target.value, 10))}
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
                      max="50"
                      value={pipelineSettings.precedents_limit_default}
                      onChange={(e) => updateSetting('precedents_limit_default', parseInt(e.target.value, 10))}
                    />
                  </label>
                </div>
                {isPipelineLikeTab && (
                  <div className="settings-row">
                    <label className="settings-toggle-label">
                      <input
                        type="checkbox"
                        checked={pipelineSettings.include_precedents}
                        onChange={(e) => updateSetting('include_precedents', e.target.checked)}
                      />
                      <span>Includi Precedenti</span>
                    </label>
                  </div>
                )}
              </fieldset>
            )}

            {showAdvancedPipelineSections && (
              <fieldset className="settings-group">
                <legend>🔗 Catena di Ragionamento</legend>
                <div className="settings-row">
                  <label>
                    <span>Min Steps</span>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={pipelineSettings.chain_min_steps}
                      onChange={(e) => updateSetting('chain_min_steps', parseInt(e.target.value, 10))}
                    />
                  </label>
                  <label>
                    <span>Max Steps</span>
                    <input
                      type="number"
                      min="3"
                      max="20"
                      value={pipelineSettings.chain_max_steps}
                      onChange={(e) => updateSetting('chain_max_steps', parseInt(e.target.value, 10))}
                    />
                  </label>
                </div>
              </fieldset>
            )}

            {/* Causal Taxonomy */}
            {showTaxonomySection && (
              <fieldset className="settings-group">
                <legend>🧬 Tassonomia Causale</legend>
                <div className="settings-row">
                  <label className="settings-toggle-label">
                    <input
                      type="checkbox"
                      checked={pipelineSettings.reasoner_enable_causality}
                      onChange={(e) => updateSetting('reasoner_enable_causality', e.target.checked)}
                    />
                    <span>Abilita Tassonomia nel Reasoner</span>
                  </label>
                  <label className="settings-toggle-label">
                    <input
                      type="checkbox"
                      checked={pipelineSettings.counter_enable_causality}
                      onChange={(e) => updateSetting('counter_enable_causality', e.target.checked)}
                    />
                    <span>Abilita Tassonomia nel Counter</span>
                  </label>
                </div>
              </fieldset>
            )}

            {isDoeTab && (
              <div className="settings-note">
                I controlli DoE del Counter permettono di isolare l’effetto della tassonomia sul contro-ragionamento.
              </div>
            )}

            {/* AQA Weights */}
            {showAdvancedPipelineSections && (
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
            )}

            {/* AQA Cross-Attack Parameters */}
            {showAdvancedPipelineSections && (
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
            )}
          </div>
        )}
      </div>

      {/* Content Area */}
      <div
        className="messages-area"
        ref={messagesAreaRef}
        onScroll={handleMessagesScroll}
      >
        <div className="messages-container">
          <div className="message message-assistant">
            <div className="message-avatar assistant-avatar">
              <Bot size={20} />
            </div>
            <div className="message-bubble bubble-assistant">
              <p className="message-text">{TAB_WELCOME_MESSAGES[activeTab]}</p>
            </div>
          </div>

          {activeTab === TABS.SEARCH && (
            <>
              {searchMessagesVisible.map((msg, idx) => (
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

          {activeTab === TABS.SEARCH && searchMessagesVisible.length === 0 && !isLoading && (
            <div className="empty-state">
              <Search size={48} className="empty-icon" />
              <p>Esegui retrieval di norme e precedenti rilevanti a partire dal claim.</p>
            </div>
          )}

          {activeTab === TABS.REASON && (
            <>
              {reasonMessages.map((msg, idx) => (
                <div
                  key={`reason-msg-${idx}`}
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
                      {renderCausalityCard('Mappatura Causale', reasoningResult.causality)}
                    </div>
                  )}

                  {reasoningResult.aspic_ir && (
                    <div className="result-section">
                      {renderAspicOverview('ASPIC+ IR', reasoningResult.aspic_ir)}
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

          {isPipelineLikeTab && pipelineHistory.length > 0 && (
            <>
              {pipelineHistory.map((run, idx) => {
                const histReasonerParsed = parseStructuredResponse(run?.reasoner?.raw_response || '');
                const histCounterParsed = parseStructuredResponse(run?.counter_reasoner?.raw_response || '');
                const histSummary = parseConsistencySummary(run?.evaluation?.summary || '');
                const histRepairedReasonerParsed = parseStructuredResponse(
                  run?.evaluation?.repaired_reasoner_chain || '',
                );
                const histRepairedCounterParsed = parseStructuredResponse(
                  run?.evaluation?.repaired_counter_chain || '',
                );
                const histReasonerRepairStats = run?.evaluation?.consistency_report?.reasoner || {};
                const histCounterRepairStats = run?.evaluation?.consistency_report?.counter_reasoner || {};
                const histHasReasonerRepairs = Number(histReasonerRepairStats.repaired_citations || 0) > 0
                  || Number(histReasonerRepairStats.dropped_citations || 0) > 0
                  || Boolean(run?.evaluation?.repaired_reasoner_chain);
                const histHasCounterRepairs = Number(histCounterRepairStats.repaired_citations || 0) > 0
                  || Number(histCounterRepairStats.dropped_citations || 0) > 0
                  || Boolean(run?.evaluation?.repaired_counter_chain);
                const histHasAnyRepairs = histHasReasonerRepairs || histHasCounterRepairs;
                const histAqaReport = run?.evaluation?.aqa_report;
                const histAqaProLinks = histAqaReport?.links?.pro ?? [];
                const histAqaContraLinks = histAqaReport?.links?.contra ?? [];
                const histAqaVerdict = histAqaReport?.verdict ?? 'uncertain';
                const histAqaVerdictLabel = histAqaVerdict === 'plausible'
                  ? 'Plausibile'
                  : histAqaVerdict === 'implausible'
                    ? 'Implausibile'
                    : 'Incerto';
                const histAqaVerdictClass = histAqaVerdict === 'plausible'
                  ? 'aqa-verdict-positive'
                  : histAqaVerdict === 'implausible'
                    ? 'aqa-verdict-negative'
                    : 'aqa-verdict-uncertain';
                const histAqaProScore = histAqaReport?.net_plausibility?.pro ?? 0;
                const histAqaContraScore = histAqaReport?.net_plausibility?.contra ?? 0;
                const histAqaFinalScore = histAqaReport?.net_plausibility?.final ?? 0;
                return (
                  <React.Fragment key={`pipeline-history-${idx}`}>
                    <div className="message message-user">
                      <div className="message-avatar user-avatar">
                        <User size={20} />
                      </div>
                      <div className="message-bubble bubble-user">
                        <p className="message-text">{run?.claim || '—'}</p>
                      </div>
                    </div>
                    <div
                      className="result-card archived-pipeline-card"
                      ref={(el) => {
                        if (el) historyPipelinePdfRefs.current[idx] = el;
                        else delete historyPipelinePdfRefs.current[idx];
                      }}
                    >
                      <div className="result-card-toolbar" data-pdf-ignore="true">
                        <button
                          type="button"
                          className="pdf-download-btn"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            downloadPipelineCardPdf({
                              reportData: run,
                              claim: run?.claim || '',
                              key: `history-${idx}`,
                              prefix: `${run?.doe_ab_comparison ? 'doe_precedente' : 'pipeline_precedente'}_${idx + 1}`,
                              exportContext: run?.doe_ab_comparison ? 'doe' : 'pipeline',
                            });
                          }}
                          disabled={exportingPdfKey !== null}
                        >
                          {exportingPdfKey === `history-${idx}`
                            ? <Loader2 size={14} className="loading-spinner" />
                            : <Download size={14} />}
                          <span>{exportingPdfKey === `history-${idx}` ? 'Esporto PDF...' : 'Scarica PDF'}</span>
                        </button>
                      </div>
                      <details className="ir-toggle history-run-toggle">
                        <summary>
                          Risultato Pipeline Precedente #{idx + 1}
                        </summary>
                        {run?.reasoner?.raw_response && (
                          <div className="result-section" style={{ marginTop: '0.75rem' }}>
                            <h4>Argomentazione Principale</h4>
                            {renderStructuredResponse({ parsed: histReasonerParsed })}
                          </div>
                        )}
                        {run?.counter_reasoner?.raw_response && (
                          <div className="result-section">
                            <h4>Argomentazione Contraria</h4>
                            {renderCounterAttacksUsed(
                              run.counter_reasoner,
                              `history-counter-attacks-${idx}`,
                            )}
                            {renderStructuredResponse({ parsed: histCounterParsed })}
                          </div>
                        )}
                        {run?.evaluation?.summary && (
                          <div className="result-section">
                            <h4>{histSummary.title || 'Riepilogo'}</h4>
                            <div className="summary-cards-grid">
                              {histSummary.sections.map((section, sIdx) => (
                                <div key={`hist-summary-${idx}-${sIdx}`} className="summary-card">
                                  <div className="summary-card-title">{section.name}</div>
                                  {section.metrics.length > 0 && (
                                    <div className="summary-metrics">
                                      {section.metrics.map((metric, mIdx) => (
                                        <div
                                          key={`hist-summary-metric-${idx}-${sIdx}-${mIdx}`}
                                          className={`summary-metric ${getSummaryMetricClass(metric.label)}`}
                                        >
                                          <span className="summary-metric-label">{metric.label}</span>
                                          <span className="summary-metric-value">{metric.value}</span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  {section.freeText.length > 0 && (
                                    <ul className="summary-notes-list">
                                      {section.freeText.map((note, nIdx) => (
                                        <li key={`hist-summary-note-${idx}-${sIdx}-${nIdx}`}>{note}</li>
                                      ))}
                                    </ul>
                                  )}
                                </div>
                              ))}
                              {histSummary.sections.length === 0 && (
                                <div className="summary-card">
                                  <div className="raw-response">{run.evaluation.summary}</div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {run?.evaluation && histHasAnyRepairs && (
                          <div className="result-section pipeline-section">
                            <h4>Catene di Ragionamento Riparate</h4>
                            {run?.evaluation?.repaired_reasoner_chain && (
                              <div className="subsection">
                                <h5>Reasoner - Catena Riparata</h5>
                                {renderStructuredResponse({
                                  parsed: histRepairedReasonerParsed,
                                  variant: 'repaired',
                                })}
                              </div>
                            )}
                            {run?.evaluation?.repaired_counter_chain && (
                              <div className="subsection">
                                <h5>Counter-Reasoner - Catena Riparata</h5>
                                {renderStructuredResponse({
                                  parsed: histRepairedCounterParsed,
                                  variant: 'repaired',
                                })}
                              </div>
                            )}
                            <div className="consistency-stats">
                              <span className="stat-item stat-repaired">
                                🔧 Reasoner: {histReasonerRepairStats.repaired_citations || 0} riparate, {histReasonerRepairStats.dropped_citations || 0} scartate
                              </span>
                              <span className="stat-item stat-repaired">
                                🔧 Counter: {histCounterRepairStats.repaired_citations || 0} riparate, {histCounterRepairStats.dropped_citations || 0} scartate
                              </span>
                            </div>
                          </div>
                        )}

                        {histAqaReport && (
                          <div className="result-section pipeline-section">
                            <h4>AQA - Valutazione Argomentativa</h4>
                            <div className="aqa-stats">
                              <span className={`aqa-badge ${histAqaVerdictClass}`}>
                                Verdetto: {histAqaVerdictLabel}
                              </span>
                              <span className="stat-item stat-valid">Tesi primaria: {(histAqaProScore * 100).toFixed(0)}%</span>
                              <span className="stat-item stat-text">Controtesi: {(histAqaContraScore * 100).toFixed(0)}%</span>
                              <span className="stat-item stat-repaired">Finale: {(histAqaFinalScore * 100).toFixed(0)}%</span>
                            </div>
                            {(histAqaProLinks.length > 0 || histAqaContraLinks.length > 0) && (
                              <>
                                <AspicMetagraph
                                  aqaReport={histAqaReport}
                                  reasonerIr={run?.evaluation?.repaired_reasoner_aspic_ir}
                                  counterIr={run?.evaluation?.repaired_counter_aspic_ir}
                                />
                                <AttackTextDetails aqaReport={histAqaReport} />
                              </>
                            )}
                          </div>
                        )}
                      </details>
                    </div>
                  </React.Fragment>
                );
              })}
            </>
          )}

          {isPipelineLikeTab && pipelineResult && (
            <>
              <div className="message message-user">
                <div className="message-avatar user-avatar">
                  <User size={20} />
                </div>
                <div className="message-bubble bubble-user">
                  <p className="message-text">{pipelineResult.claim}</p>
                </div>
              </div>
              <div className="result-card" ref={currentPipelinePdfRef}>
              <div className="result-card-toolbar" data-pdf-ignore="true">
                <button
                  type="button"
                  className="pdf-download-btn"
                  onClick={() => {
                    downloadPipelineCardPdf({
                      reportData: pipelineResult,
                      claim: pipelineResult?.claim || '',
                      key: 'current',
                      prefix: activeTab === TABS.DOE ? 'doe_automatico' : 'pipeline_completa',
                      exportContext: activeTab === TABS.DOE ? 'doe' : 'pipeline',
                    });
                  }}
                  disabled={exportingPdfKey !== null}
                >
                  {exportingPdfKey === 'current'
                    ? <Loader2 size={14} className="loading-spinner" />
                    : <Download size={14} />}
                  <span>{exportingPdfKey === 'current' ? 'Esporto PDF...' : 'Scarica PDF'}</span>
                </button>
              </div>
              <h3 className="result-title">
                {activeTab === TABS.DOE ? <GitBranch size={20} /> : <FileText size={20} />}
                {activeTab === TABS.DOE ? 'Risultato DoE' : 'Risultato Pipeline Completa'}
              </h3>

              {pipelineResult.error ? (
                <p className="error-text">Errore: {pipelineResult.error}</p>
              ) : (
                <>
                  <div className="result-section" data-pdf-section="claim">
                    <h4>Claim Analizzato</h4>
                    <p>{pipelineResult.claim}</p>
                  </div>

                  {isDoeTab && doeHasComparison && (
                    <div className="doe-floating-switch-wrap" data-pdf-ignore="true">
                      <div className="doe-inline-switch doe-floating-switch">
                        <div className="ios-toggle-copy">
                          <div className="ios-toggle-title">Vista Counter</div>
                          <div className="ios-toggle-subtitle">
                            {doeDisplaySetup === 'baseline'
                              ? `Setup A baseline${doeComparison?.baseline?.status === 'running' ? ' in generazione' : ''}`
                              : `Setup B treatment${doeComparison?.treatment?.status === 'running' ? ' in generazione' : ''}`}
                          </div>
                        </div>
                        <div className="doe-inline-switch-controls">
                          <span className={`doe-inline-switch-label ${doeDisplaySetup === 'baseline' ? 'is-active' : ''}`}>A</span>
                          <button
                            type="button"
                            role="switch"
                            aria-label="Cambia vista Counter tra Setup A e Setup B"
                            aria-checked={doeDisplaySetup === 'treatment'}
                            aria-valuetext={doeDisplaySetup === 'baseline' ? 'Setup A baseline' : 'Setup B treatment'}
                            className={`ios-switch ${doeDisplaySetup === 'treatment' ? 'is-on' : ''}`}
                            onClick={() =>
                              applyDoeViewSelection(
                                doeDisplaySetup === 'baseline' ? 'treatment' : 'baseline',
                              )
                            }
                            title="Switch vista Counter tra Setup A e Setup B"
                          >
                            <span className="ios-switch-knob" />
                          </button>
                          <span className={`doe-inline-switch-label ${doeDisplaySetup === 'treatment' ? 'is-active' : ''}`}>B</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {pipelineResult._stream && (
                    <div className="result-section stream-progress-section" data-pdf-section="stream">
                      <h4>Avanzamento Live</h4>
                      <div className="stream-phase-grid">
                        {PIPELINE_PHASES.map((phase) => {
                          const phaseStatus = pipelineResult._stream?.phases?.[phase.key] || 'pending';
                          const phaseProgress = pipelineResult._stream?.phase_progress?.[phase.key] ?? 0;
                          const phaseDetail = pipelineResult._stream?.phase_details?.[phase.key] || '';
                          const phaseLabel = phaseStatus === 'active'
                            ? `In corso (${Math.round(phaseProgress)}%)`
                            : phaseStatus === 'done'
                              ? 'Completata'
                              : phaseStatus === 'error'
                                ? 'Errore'
                                : 'In attesa';
                          return (
                            <div key={phase.key} className={`stream-phase-item ${phaseStatus === 'active' ? 'active-phase' : ''}`}>
                              <div className="stream-phase-head">
                                <span>{phase.label}</span>
                                <span className={`stream-phase-badge status-${phaseStatus}`}>
                                  {phaseStatus === 'active' && <Loader2 size={12} className="loading-spinner" />}
                                  {phaseLabel}
                                </span>
                              </div>
                              {phaseDetail && (
                                <div className="stream-phase-detail">{phaseDetail}</div>
                              )}
                              <div className="stream-phase-track">
                                <div
                                  className={`stream-phase-fill status-${phaseStatus}`}
                                  style={{ width: `${Math.max(phaseStatus === 'pending' ? 4 : 0, phaseProgress)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {(pipelineResult.retrieval_context?.statutes?.length > 0
                    || pipelineResult.retrieval_context?.precedents?.length > 0) && (
                    <div className="result-section pipeline-section" data-pdf-section="retrieval">
                      <h3 className="section-header">
                        <CheckCircle2 size={20} style={{ color: '#64748b' }} />
                        Contesto Pre-retrieval
                      </h3>
                      <div className="subsection">
                        <h4>
                          Knowledge Base condivisa
                          {pipelineResult.retrieval_context?.memory?.enabled && (
                            <span style={{ marginLeft: 8, fontWeight: 500, opacity: 0.85 }}>
                              {pipelineResult.retrieval_context?.memory?.hit
                                ? '(da memoria cache)'
                                : '(da retrieval live)'}
                            </span>
                          )}
                        </h4>
                        {pipelineResult.retrieval_context?.statutes?.length > 0 && (
                          <>
                            <p style={{ marginTop: 0, marginBottom: 8, opacity: 0.85 }}>
                              Articoli ({pipelineResult.retrieval_context.statutes.length})
                            </p>
                            <CollapsibleList
                              items={pipelineResult.retrieval_context.statutes}
                              limit={5}
                              renderItem={(art, idx) => (
                                <li key={`ctx-stat-${idx}`}>
                                  <strong>{idx + 1}. Art. {art.articolo || art.statute_id}</strong>
                                  {art.source && ` (${sourceShortLabel(art.source)})`}
                                  {art.titolo && ` - ${art.titolo}`}
                                </li>
                              )}
                            />
                          </>
                        )}
                        {pipelineResult.retrieval_context?.precedents?.length > 0 && (
                          <>
                            <p style={{ marginTop: 12, marginBottom: 8, opacity: 0.85 }}>
                              Precedenti ({pipelineResult.retrieval_context.precedents.length})
                            </p>
                            <CollapsibleList
                              items={pipelineResult.retrieval_context.precedents}
                              limit={5}
                              renderItem={(prec, idx) => (
                                <li key={`ctx-pr-${idx}`}>
                                  <strong>{idx + 1}. {prec.title || `Precedente ${idx + 1}`}</strong>
                                </li>
                              )}
                            />
                          </>
                        )}
                      </div>
                    </div>
                  )}

                  {/* SEZIONE REASONER */}
                  <div className="result-section pipeline-section" data-pdf-section="reasoner">
                    <h3 className="section-header">
                      <CheckCircle2 size={20} style={{ color: '#10b981' }} />
                      1. REASONER - Tesi Principale
                    </h3>

                    {pipelineResult.reasoner?.causality && (
                      <div className="subsection">
                        <h4>Classificazione Causalità</h4>
                        {renderCausalityCard('Mappatura Causale', pipelineResult.reasoner.causality)}
                      </div>
                    )}

                    {pipelineResult.reasoner?.statutes && pipelineResult.reasoner.statutes.length > 0 && (
                      <div className="subsection">
                        <h4>Articoli Trovati (Reasoner) ({pipelineResult.reasoner.statutes.length})</h4>
                        <CollapsibleList items={pipelineResult.reasoner.statutes} limit={5} renderItem={(art, idx) => (
                          <li key={idx}>
                            <strong>{idx + 1}. Art. {art.articolo || art.statute_id}</strong>
                            {art.source && ` (${sourceShortLabel(art.source)})`}
                            {art.titolo && ` - ${art.titolo}`}
                          </li>
                        )} />
                      </div>
                    )}

                    {pipelineResult.reasoner?.precedents && pipelineResult.reasoner.precedents.length > 0 && (
                      <div className="subsection">
                        <h4>Precedenti Trovati (Reasoner) ({pipelineResult.reasoner.precedents.length})</h4>
                        <CollapsibleList items={pipelineResult.reasoner.precedents} limit={5} renderItem={(prec, idx) => (
                          <li key={idx}>
                            <strong>{idx + 1}. {prec.title || `Precedente ${idx + 1}`}</strong>
                          </li>
                        )} />
                      </div>
                    )}

                    {pipelineResult.reasoner?.aspic_ir && (
                      <div className="subsection">
                        {renderAspicOverview('ASPIC+ IR (Reasoner)', pipelineResult.reasoner.aspic_ir, true)}
                      </div>
                    )}
                    {pipelineResult._stream?.phases?.support === 'active' && !pipelineResult.reasoner?.aspic_ir && (
                      <div className="subsection">
                        <h4>ASPIC+ IR (Reasoner)</h4>
                        <div className="structured-live-tag">
                          <Loader2 size={14} className="loading-spinner" />
                          <span>Costruzione ASPIC+ in corso...</span>
                        </div>
                      </div>
                    )}

                    {pipelineResult._stream?.reasoner_refinement_active && (
                      <div className="subsection">
                        <h4>Refinement Reasoner</h4>
                        <div className="structured-live-tag">
                          <Loader2 size={14} className="loading-spinner" />
                          <span>
                            Riclassificazione causale e rigenerazione della catena con norme tassonomiche...
                          </span>
                        </div>
                      </div>
                    )}

                    {(pipelineResult.reasoner?.raw_response || liveSupportSteps.length > 0 || reasonerLiveConclusion) && (
                      <div className="subsection">
                        <h4>Risposta Completa</h4>
                        {renderStructuredResponse({
                          parsed: reasonerParsedResponse,
                          liveSteps: liveSupportStepTexts,
                          liveConclusion: reasonerLiveConclusion,
                          liveMode: pipelineResult._stream?.phases?.support === 'active',
                        })}
                      </div>
                    )}
                  </div>

                  {/* SEZIONE COUNTER-REASONER */}
                  <div className="result-section pipeline-section" data-pdf-section="counter">
                    <h3 className="section-header">
                      <XCircle size={20} style={{ color: '#ef4444' }} />
                      2. COUNTER-REASONER - Controtesi
                      {isDoeTab && (
                        <span className={`doe-setup-badge tone-${selectedCounterLiveBadgeTone}`}>
                          {selectedCounterLiveBadgeLabel}
                        </span>
                      )}
                    </h3>

                    {isDoeTab
                      && doeHasComparison
                      && !displayedCounterReasoner?.raw_response
                      && selectedCounterLiveSteps.length === 0
                      && !displayedCounterReasoner?.aspic_ir && (
                        <div className="subsection">
                          <div className="structured-live-tag">
                            <Loader2 size={14} className="loading-spinner" />
                            <span>
                              {doeDisplaySetup === 'baseline'
                                ? 'Setup A non ha ancora prodotto output del Counter.'
                                : 'Setup B non ha ancora prodotto output del Counter.'}
                            </span>
                          </div>
                        </div>
                      )}

                    {displayedCounterReasoner?.abstained && (
                      <div className="subsection" style={{ 
                        background: 'rgba(234, 179, 8, 0.1)', 
                        border: '1px solid rgba(234, 179, 8, 0.3)', 
                        borderRadius: '8px', 
                        padding: '16px',
                        marginBottom: '12px'
                      }}>
                        <h4 style={{ color: '#eab308', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <AlertTriangle size={18} />
                          Astensione del Counter-Reasoner
                        </h4>
                        {displayedEvaluation?.consistency_report?.counter_reasoner_gate?.abstain && (
                          <p style={{ margin: '8px 0 0', opacity: 0.95 }}>
                            Astensione applicata dal Polisher gate
                            {displayedEvaluation.consistency_report.counter_reasoner_gate.label
                              ? ` (label: ${displayedEvaluation.consistency_report.counter_reasoner_gate.label})`
                              : ''}.
                          </p>
                        )}
                        <p style={{ margin: '8px 0 0', opacity: 0.9 }}>
                          {displayedCounterReasoner.abstention_reason || 
                           'Il sistema non ha individuato contro-argomentazioni giuridicamente solide per questo caso.'}
                        </p>
                        <details className="ir-toggle" style={{ marginTop: '10px' }}>
                          <summary>Conclusione del Reasoner passata al Counter-Reasoner</summary>
                          <div className="raw-response" style={{ marginTop: '8px' }}>
                            {displayedCounterReasoner.reasoner_conclusion_context || 'Conclusione non disponibile.'}
                          </div>
                        </details>
                      </div>
                    )}

                    {displayedCounterReasoner?.reasoner_causality && (
                      <div className="subsection">
                        <h4>Causalità del Reasoner (da Attaccare)</h4>
                        {renderCausalityCard('Target Causale da Attaccare', displayedCounterReasoner.reasoner_causality)}
                      </div>
                    )}

                    {renderCounterAttacksUsed(
                      displayedCounterReasoner,
                      'pipeline-counter-attacks',
                    )}

                    {displayedCounterReasoner?.warrant_info && (
                      <div className="subsection">
                        <h4>Warrant e Causalità Attaccanti</h4>
                        <pre className="code-block">
                          {JSON.stringify(displayedCounterReasoner.warrant_info, null, 2)}
                        </pre>
                      </div>
                    )}

                    {displayedCounterReasoner?.statutes && displayedCounterReasoner.statutes.length > 0 && (
                      <div className="subsection">
                        <h4>Articoli Trovati (Counter-Reasoner) ({displayedCounterReasoner.statutes.length})</h4>
                        <CollapsibleList items={displayedCounterReasoner.statutes} limit={5} renderItem={(art, idx) => (
                          <li key={idx}>
                            <strong>{idx + 1}. Art. {art.articolo || art.statute_id}</strong>
                            {art.source && ` (${sourceShortLabel(art.source)})`}
                            {art.titolo && ` - ${art.titolo}`}
                          </li>
                        )} />
                      </div>
                    )}

                    {displayedCounterReasoner?.precedents && displayedCounterReasoner.precedents.length > 0 && (
                      <div className="subsection">
                        <h4>Precedenti Trovati (Counter-Reasoner) ({displayedCounterReasoner.precedents.length})</h4>
                        <CollapsibleList items={displayedCounterReasoner.precedents} limit={5} renderItem={(prec, idx) => (
                          <li key={idx}>
                            <strong>{idx + 1}. {prec.title || `Precedente ${idx + 1}`}</strong>
                          </li>
                        )} />
                      </div>
                    )}

                    {displayedCounterReasoner?.aspic_ir && (
                      <div className="subsection">
                        {renderAspicOverview('ASPIC+ IR (Counter-Reasoner)', displayedCounterReasoner.aspic_ir, true)}
                      </div>
                    )}
                    {selectedCounterLiveMode && !displayedCounterReasoner?.aspic_ir && (
                      <div className="subsection">
                        <h4>ASPIC+ IR (Counter-Reasoner)</h4>
                        <div className="structured-live-tag">
                          <Loader2 size={14} className="loading-spinner" />
                          <span>Costruzione ASPIC+ in corso...</span>
                        </div>
                      </div>
                    )}

                    {(displayedCounterReasoner?.raw_response || selectedCounterLiveSteps.length > 0) && (
                      <div className="subsection">
                        <h4>Risposta Completa</h4>
                        {renderStructuredResponse({
                          parsed: counterParsedResponse,
                          liveSteps: selectedCounterLiveSteps,
                          liveMode: selectedCounterLiveMode,
                        })}
                      </div>
                    )}
                  </div>

                  {/* SEZIONE EVALUATOR - Verifica Consistenza */}
                  {(evaluationPhaseActive || displayedEvaluation?.consistency_report) && (
                    <div className="result-section pipeline-section" data-pdf-section="evaluator">
                      <h3 className="section-header">
                        <ClipboardCheck size={20} style={{ color: '#8b5cf6' }} />
                        3. EVALUATOR - Verifica Consistenza
                      </h3>

                      {evaluationPhaseActive && (
                        <div className="subsection evaluation-live-status">
                          <div className="structured-live-tag">
                            <Loader2 size={14} className="loading-spinner" />
                            <span>Valutazione in corso ({evaluationPhaseProgress}%)</span>
                          </div>
                          {evaluationPhaseDetail && (
                            <div className="stream-phase-detail">{evaluationPhaseDetail}</div>
                          )}
                        </div>
                      )}

                      {/* Reasoner Consistency */}
                      {evaluationReasonerReport && (
                        <div className="subsection">
                          <h4>
                            Reasoner - Score: {(((evaluationReasonerReport.consistency_score ?? 0) * 100)).toFixed(0)}%
                          </h4>
                          <div className="consistency-stats">
                            <span className="stat-item stat-valid">
                              ✅ Valide: {(evaluationReasonerReport.valid_citations ?? 0)}/{(evaluationReasonerReport.total_citations ?? 0)}
                            </span>
                            <span className="stat-item stat-text">
                              📝 Testo match: {(evaluationReasonerReport.text_matches ?? 0)}/{((evaluationReasonerReport.text_matches ?? 0) + (evaluationReasonerReport.text_mismatches ?? 0))}
                            </span>
                          </div>

                          {evaluationReasonerReport.citation_checks?.length > 0 && (
                            <div className="citation-checks-list">
                              {evaluationReasonerReport.citation_checks.map((check, idx) => (
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

                          {evaluationReasonerReport.issues?.length > 0 && (
                            <div className="issues-list">
                              <h5><AlertTriangle size={14} /> Problemi Rilevati:</h5>
                              <ul>
                                {evaluationReasonerReport.issues.map((issue, idx) => (
                                  <li key={idx}>{issue}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Counter-Reasoner Consistency */}
                      {evaluationCounterReport && (
                        <div className="subsection">
                          <h4>
                            Counter-Reasoner - Score: {(((evaluationCounterReport.consistency_score ?? 0) * 100)).toFixed(0)}%
                          </h4>
                          <div className="consistency-stats">
                            <span className="stat-item stat-valid">
                              ✅ Valide: {(evaluationCounterReport.valid_citations ?? 0)}/{(evaluationCounterReport.total_citations ?? 0)}
                            </span>
                            <span className="stat-item stat-text">
                              📝 Testo match: {(evaluationCounterReport.text_matches ?? 0)}/{((evaluationCounterReport.text_matches ?? 0) + (evaluationCounterReport.text_mismatches ?? 0))}
                            </span>
                          </div>

                          {evaluationCounterReport.citation_checks?.length > 0 && (
                            <div className="citation-checks-list">
                              {evaluationCounterReport.citation_checks.map((check, idx) => (
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

                          {evaluationCounterReport.issues?.length > 0 && (
                            <div className="issues-list">
                              <h5><AlertTriangle size={14} /> Problemi Rilevati:</h5>
                              <ul>
                                {evaluationCounterReport.issues.map((issue, idx) => (
                                  <li key={idx}>{issue}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Summary */}
                      {displayedEvaluation.summary && (
                        <div className="subsection">
                          <h4>{parsedSummary.title || 'Riepilogo'}</h4>
                          <div className="summary-cards-grid">
                            {parsedSummary.sections.map((section, idx) => (
                              <div key={`summary-section-${idx}`} className="summary-card">
                                <div className="summary-card-title">{section.name}</div>
                                {section.metrics.length > 0 && (
                                  <div className="summary-metrics">
                                    {section.metrics.map((metric, mIdx) => (
                                      <div
                                        key={`summary-metric-${idx}-${mIdx}`}
                                        className={`summary-metric ${getSummaryMetricClass(metric.label)}`}
                                      >
                                        <span className="summary-metric-label">{metric.label}</span>
                                        <span className="summary-metric-value">{metric.value}</span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {section.freeText.length > 0 && (
                                  <ul className="summary-notes-list">
                                    {section.freeText.map((note, nIdx) => (
                                      <li key={`summary-note-${idx}-${nIdx}`}>{note}</li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            ))}
                            {parsedSummary.sections.length === 0 && (
                              <div className="summary-card">
                                <div className="raw-response">{displayedEvaluation.summary}</div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                    </div>
                  )}

                  {/* SEZIONE CATENE RIPARATE */}
                  {displayedEvaluation && hasAnyRepairs && (
                    <div className="result-section pipeline-section" data-pdf-section="repaired">
                      <h3 className="section-header">
                        <Wrench size={20} style={{ color: '#f59e0b' }} />
                        4. CATENE DI RAGIONAMENTO RIPARATE
                      </h3>

                      {/* Repaired Reasoner Chain */}
                      {hasReasonerRepairs && displayedEvaluation.repaired_reasoner_chain && (
                        <div className="subsection">
                          <h4 className="subsection-title-with-icon">
                            <CheckCircle2 size={20} style={{ color: '#10b981' }} />
                            Reasoner - Catena Riparata
                          </h4>
                          {renderStructuredResponse({
                            parsed: repairedReasonerParsedResponse,
                            variant: 'repaired',
                          })}

                          {/* Show repaired ASPIC IR if available */}
                          {displayedEvaluation.repaired_reasoner_aspic_ir && Object.keys(displayedEvaluation.repaired_reasoner_aspic_ir).length > 0 && (
                            <>
                              {renderAspicOverview(
                                'ASPIC+ IR Riparato (Reasoner)',
                                displayedEvaluation.repaired_reasoner_aspic_ir,
                                true,
                              )}
                            </>
                          )}
                        </div>
                      )}

                      {/* Repaired Counter-Reasoner Chain */}
                      {hasCounterRepairs && displayedEvaluation.repaired_counter_chain && (
                        <div className="subsection">
                          <h4 className="subsection-title-with-icon">
                            <XCircle size={20} style={{ color: '#ef4444' }} />
                            Counter-Reasoner - Catena Riparata
                            {isDoeTab && (
                              <span className={`doe-setup-badge tone-${selectedCounterLiveBadgeTone}`}>
                                {selectedCounterLiveBadgeLabel}
                              </span>
                            )}
                          </h4>
                          {renderStructuredResponse({
                            parsed: repairedCounterParsedResponse,
                            variant: 'repaired',
                          })}

                          {/* Show repaired ASPIC IR if available */}
                          {displayedEvaluation.repaired_counter_aspic_ir && Object.keys(displayedEvaluation.repaired_counter_aspic_ir).length > 0 && (
                            <>
                              {renderAspicOverview(
                                'ASPIC+ IR Riparato (Counter-Reasoner)',
                                displayedEvaluation.repaired_counter_aspic_ir,
                                true,
                              )}
                            </>
                          )}
                        </div>
                      )}

                      {/* Repair Statistics */}
                      {displayedEvaluation.consistency_report && (
                        <div className="subsection repair-stats">
                          <h4>Statistiche Riparazione</h4>
                          <div className="consistency-stats">
                            {displayedEvaluation.consistency_report.reasoner && (
                              <>
                                <span className="stat-item stat-repaired">
                                  🔧 Reasoner: {displayedEvaluation.consistency_report.reasoner.repaired_citations || 0} riparate, {displayedEvaluation.consistency_report.reasoner.dropped_citations || 0} scartate
                                </span>
                              </>
                            )}
                            {displayedEvaluation.consistency_report.counter_reasoner && (
                              <>
                                <span className="stat-item stat-repaired">
                                  🔧 Counter: {displayedEvaluation.consistency_report.counter_reasoner.repaired_citations || 0} riparate, {displayedEvaluation.consistency_report.counter_reasoner.dropped_citations || 0} scartate
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {evaluationPhaseActive && (
                    <div className="pipeline-working-indicator evaluation-working-footer">
                      <Loader2 size={14} className="loading-spinner" />
                      <span>{evaluationPhaseDetail || 'Valutazione AQA e attacchi in corso...'}</span>
                    </div>
                  )}

                  {/* SEZIONE AQA - Valutazione Argomentativa (sulle catene riparate) */}
                  {aqaReport && (
                    <div className="result-section pipeline-section" data-pdf-section="aqa">
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
                              Tesi primaria: {(aqaProScore * 100).toFixed(0)}%
                            </span>
                            <span className="stat-item stat-text">
                              Controtesi: {(aqaContraScore * 100).toFixed(0)}%
                            </span>
                            <span className="stat-item stat-repaired">
                              Finale: {(aqaFinalScore * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="aqa-meta">
                            <span>Link tesi primaria: {aqaProLinks.length}</span>
                            <span>Link controtesi: {aqaContraLinks.length}</span>
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
                                📚 Norm Tesi primaria: {((aqaReport.chain_scores.pro?.norm_support_avg ?? 0) * 100).toFixed(0)}%
                              </span>
                              <span className="stat-item stat-text">
                                📚 Norm Controtesi: {((aqaReport.chain_scores.contra?.norm_support_avg ?? 0) * 100).toFixed(0)}%
                              </span>
                              {aqaReport.chain_scores.pro?.attacks_avg > 0 && (
                                <span className="stat-item stat-repaired">
                                  ⚔️ Attacchi Tesi primaria: {((aqaReport.chain_scores.pro?.attacks_avg ?? 0)).toFixed(3)}
                                </span>
                              )}
                              {aqaReport.chain_scores.contra?.attacks_avg > 0 && (
                                <span className="stat-item stat-repaired">
                                  ⚔️ Attacchi Controtesi: {((aqaReport.chain_scores.contra?.attacks_avg ?? 0)).toFixed(3)}
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
                                  <h6>Reasoner (Tesi primaria)</h6>
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
                                  <h6>Counter-Reasoner (Controtesi)</h6>
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

                          <details className="ir-toggle aqa-full-toggle">
                            <summary>Dettagli AQA completi</summary>
                            {renderAqaFullView(aqaReport)}
                          </details>
                        </>
                      ) : (
                        <div className="aqa-disabled">AQA disabilitata</div>
                      )}
                    </div>
                  )}

                  {/* SEZIONE METAGRAFO ASPIC+ */}
                  {aqaReport && aqaReport.enabled && (aqaProLinks.length > 0 || aqaContraLinks.length > 0) && (
                    <div className="result-section pipeline-section" data-pdf-section="attacks-metagraph">
                      <h3 className="section-header">
                        <GitBranch size={20} style={{ color: '#7c3aed' }} />
                        6. ASPIC+ Metagrafo — Attacchi Incrociati
                      </h3>
                      <p style={{ fontSize: '0.82rem', color: '#6b7280', marginBottom: '0.75rem' }}>
                        Clicca su un nodo per vedere i dettagli. Scroll per zoomare, trascina per spostare il grafo.
                      </p>
                      <AspicMetagraph
                        aqaReport={aqaReport}
                        reasonerIr={displayedEvaluation?.repaired_reasoner_aspic_ir}
                        counterIr={displayedEvaluation?.repaired_counter_aspic_ir}
                      />
                    </div>
                  )}

                  {/* SEZIONE DETTAGLIO TESTUALE ATTACCHI */}
                  {aqaReport && aqaReport.enabled && (aqaProLinks.length > 0 || aqaContraLinks.length > 0) && (
                    <div className="result-section pipeline-section" data-pdf-section="attacks-detail">
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

                  {renderDoeComparisonSection()}
                  {renderDoeAnalysisSection()}
                </>
              )}
              </div>
            </>
          )}

          {activeTab === TABS.REASON && !reasoningResult && reasonMessages.length === 0 && !isLoading && (
            <div className="empty-state">
              <Brain size={48} className="empty-icon" />
              <p>Esegui il Reasoner per costruire la tesi principale con catena argomentativa.</p>
            </div>
          )}

          {isPipelineLikeTab && !pipelineResult && pipelineMessages.length === 0 && !isLoading && (
            <div className="empty-state">
              {activeTab === TABS.DOE
                ? <GitBranch size={48} className="empty-icon" />
                : <FileText size={48} className="empty-icon" />}
              <p>
                {activeTab === TABS.DOE
                  ? 'Esegui il DoE automatico: due run A/B sullo stesso claim con confronto diretto delle metriche.'
                  : 'Esegui la pipeline completa: retrieval, Reasoner, Counter-Reasoner e Polisher-Evaluator.'}
              </p>
            </div>
          )}

          {isLoading && activeTab !== TABS.PIPELINE && activeTab !== TABS.DOE && (
            <div className="message message-assistant">
              <div className="message-avatar assistant-avatar">
                <Loader2 size={20} className="loading-spinner" />
              </div>
              <div className="message-bubble bubble-assistant">
                <p>Elaborazione in corso...</p>
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
              className={`input-textarea ${isPipelineLikeTab ? 'has-inline-tool' : ''}`}
              disabled={isLoading}
            />
            {isPipelineLikeTab && (
              <div className="input-tools">
                <button
                  type="button"
                  className={`input-plus-button ${claimMemoryMenuOpen ? 'is-open' : ''}`}
                  title={activeTab === TABS.DOE ? 'Opzioni DoE' : 'Opzioni memoria pre-retrieval'}
                  aria-label={activeTab === TABS.DOE ? 'Apri opzioni DoE' : 'Apri opzioni memoria pre-retrieval'}
                  aria-expanded={claimMemoryMenuOpen}
                  onClick={() => setClaimMemoryMenuOpen((prev) => !prev)}
                  disabled={isLoading}
                >
                  <Plus size={18} />
                </button>
                {claimMemoryMenuOpen && (
                  <div className="input-tools-menu" role="dialog" aria-label="Opzioni aggiuntive">
                    <div className="input-tools-menu-header">
                      {activeTab === TABS.DOE ? 'Opzioni DoE' : 'Memoria pre-retrieval'}
                    </div>

                    <div className="ios-toggle-row">
                      <div className="ios-toggle-copy">
                        <div className="ios-toggle-title">Usa memoria</div>
                        <div className="ios-toggle-subtitle">
                          Riusa statuti e precedenti già filtrati per questo claim
                        </div>
                      </div>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={!!inlineSettings.claim_context_memory_enabled}
                        className={`ios-switch ${inlineSettings.claim_context_memory_enabled ? 'is-on' : ''}`}
                        onClick={() => {
                          const checked = !inlineSettings.claim_context_memory_enabled;
                          const updater = activeTab === TABS.DOE ? setDoeSettings : setPipelineSettings;
                          updater((prev) => ({
                            ...prev,
                            claim_context_memory_enabled: checked,
                            claim_context_memory_overwrite: checked
                              ? prev.claim_context_memory_overwrite
                              : false,
                          }));
                        }}
                      >
                        <span className="ios-switch-knob" />
                      </button>
                    </div>

                    <div className={`ios-toggle-row ${!inlineSettings.claim_context_memory_enabled ? 'is-disabled' : ''}`}>
                      <div className="ios-toggle-copy">
                        <div className="ios-toggle-title">Sovrascrivi</div>
                        <div className="ios-toggle-subtitle">
                          Rigenera e aggiorna la memoria per il claim corrente
                        </div>
                      </div>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={!!inlineSettings.claim_context_memory_overwrite}
                        className={`ios-switch ${inlineSettings.claim_context_memory_overwrite ? 'is-on' : ''}`}
                        onClick={() =>
                          updateInlineSetting(
                            'claim_context_memory_overwrite',
                            !inlineSettings.claim_context_memory_overwrite,
                          )
                        }
                        disabled={!inlineSettings.claim_context_memory_enabled}
                      >
                        <span className="ios-switch-knob" />
                      </button>
                    </div>

                    {activeTab === TABS.DOE && (
                      <>
                        <div className="input-tools-menu-header" style={{ marginTop: 10 }}>
                          Controlli Counter Setup A (DoE)
                        </div>

                        <div className="ios-toggle-row">
                          <div className="ios-toggle-copy">
                            <div className="ios-toggle-title">Reasoner: tassonomia</div>
                            <div className="ios-toggle-subtitle">
                              Abilita/disabilita classificazione causale nel Reasoner (uguale per run A e run B)
                            </div>
                          </div>
                          <button
                            type="button"
                            role="switch"
                            aria-checked={!!doeSettings.reasoner_enable_causality}
                            className={`ios-switch ${doeSettings.reasoner_enable_causality ? 'is-on' : ''}`}
                            onClick={() =>
                              updateDoeSetting(
                                'reasoner_enable_causality',
                                !doeSettings.reasoner_enable_causality,
                              )
                            }
                          >
                            <span className="ios-switch-knob" />
                          </button>
                        </div>

                        <div className="ios-toggle-row">
                          <div className="ios-toggle-copy">
                            <div className="ios-toggle-title">Passa causal_type/theory (Setup A)</div>
                            <div className="ios-toggle-subtitle">
                              Invia al Counter causal_type_id e theory_id del Reasoner
                            </div>
                          </div>
                          <button
                            type="button"
                            role="switch"
                            aria-checked={!!doeSettings.counter_pass_causal_identity}
                            className={`ios-switch ${doeSettings.counter_pass_causal_identity ? 'is-on' : ''}`}
                            onClick={() =>
                              updateDoeSetting(
                                'counter_pass_causal_identity',
                                !doeSettings.counter_pass_causal_identity,
                              )
                            }
                          >
                            <span className="ios-switch-knob" />
                          </button>
                        </div>

                        <div className="ios-toggle-row">
                          <div className="ios-toggle-copy">
                            <div className="ios-toggle-title">Passa attacchi tassonomici (Setup A)</div>
                            <div className="ios-toggle-subtitle">
                              Invia al Counter il pool di attack ids da tassonomia
                            </div>
                          </div>
                          <button
                            type="button"
                            role="switch"
                            aria-checked={!!doeSettings.counter_pass_taxonomy_attacks}
                            className={`ios-switch ${doeSettings.counter_pass_taxonomy_attacks ? 'is-on' : ''}`}
                            onClick={() =>
                              updateDoeSetting(
                                'counter_pass_taxonomy_attacks',
                                !doeSettings.counter_pass_taxonomy_attacks,
                              )
                            }
                          >
                            <span className="ios-switch-knob" />
                          </button>
                        </div>

                        <div className="ios-toggle-row">
                          <div className="ios-toggle-copy">
                            <div className="ios-toggle-title">Passa norme tassonomiche (Setup A)</div>
                            <div className="ios-toggle-subtitle">
                              Invia al Counter anchor norms e principle tests (non la KB pre-retrieval)
                            </div>
                          </div>
                          <button
                            type="button"
                            role="switch"
                            aria-checked={!!doeSettings.counter_pass_norms}
                            className={`ios-switch ${doeSettings.counter_pass_norms ? 'is-on' : ''}`}
                            onClick={() =>
                              updateDoeSetting(
                                'counter_pass_norms',
                                !doeSettings.counter_pass_norms,
                              )
                            }
                          >
                            <span className="ios-switch-knob" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="send-button"
          >
            <Send size={20} />
          </button>
          {isPipelineLikeTab && isLoading && (
            <button
              onClick={handleStopPipeline}
              disabled={isStoppingPipeline}
              className="stop-button"
              title="Interrompi esecuzione"
            >
              <Square size={18} />
            </button>
          )}
        </div>
        <p className="input-hint">
          Premi Invio per inviare, Shift+Invio per andare a capo
        </p>
      </div>
    </div>
  );
}
