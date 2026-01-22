import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import './ChatBot.css';

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
    
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          top_k: 5
        }),
      });

      if (!response.ok) {
        throw new Error('Errore nella risposta del server');
      }

      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.response,
        metadata: {
          classification: data.classification,
          articles: data.articles,
          precedents: data.precedents
        }
      }]);
    } catch (error) {
      console.error('Errore:', error);
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
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <div className="header-content">
          <div className="avatar-icon assistant-avatar">
            <Bot size={24} />
          </div>
          <div>
            <h1 className="header-title">LexCausa AI</h1>
            <p className="header-subtitle">Assistente Legale</p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="messages-area">
        <div className="messages-container">
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

          {isLoading && (
            <div className="message message-assistant">
              <div className="message-avatar assistant-avatar">
                <Bot size={20} />
              </div>
              <div className="message-bubble bubble-assistant">
                <Loader2 size={20} className="loading-spinner" />
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
              placeholder="Scrivi un messaggio..."
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