import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';

export default function ChatbotInterface() {
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: 'Ciao! Sono LexCausa, il tuo assistente per ricerche legali nel Codice Civile e Penale italiano. Descrivimi il tuo caso e ti aiuterò a trovare gli articoli e i precedenti più rilevanti.' 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    
    // Aggiungi il messaggio utente
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // Chiamata al backend Flask (porta 8000 per non confliggere con Neo4j:7474)
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          top_k: 5  // Numero di risultati da restituire
        }),
      });

      if (!response.ok) {
        throw new Error('Errore nella risposta del server');
      }

      const data = await response.json();
      
      // La risposta contiene sia il testo formattato che i dati strutturati
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.response,
        // Salva anche i dati strutturati per eventuali usi futuri
        metadata: {
          classification: data.classification,
          articles: data.articles,
          precedents: data.precedents
        }
      }]);
    } catch (error) {
      console.error('Errore:', error);
      // Messaggio di errore all'utente
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Mi dispiace, si è verificato un errore. Riprova più tardi.' 
      }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900">LexCausa AI</h1>
            <p className="text-sm text-gray-500">Assistente Legale</p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-3 ${
                msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              {/* Avatar */}
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  msg.role === 'user'
                    ? 'bg-gray-700'
                    : 'bg-blue-600'
                }`}
              >
                {msg.role === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>

              {/* Message Bubble */}
              <div
                className={`px-4 py-3 rounded-2xl max-w-[70%] ${
                  msg.role === 'user'
                    ? 'bg-gray-700 text-white'
                    : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                }`}
              >
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {msg.content}
                </p>
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="px-4 py-3 rounded-2xl bg-white shadow-sm border border-gray-200">
                <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-3 items-end">
            <div className="flex-1 relative">
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
                placeholder="Scrivi un messaggio..."
                rows={1}
                className="w-full px-4 py-3 pr-12 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                style={{ minHeight: '48px', maxHeight: '120px' }}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              className="w-12 h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition-colors"
            >
              <Send className="w-5 h-5 text-white" />
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Premi Invio per inviare, Shift+Invio per andare a capo
          </p>
        </div>
      </div>
    </div>
  );
}