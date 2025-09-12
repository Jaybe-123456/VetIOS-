import React, { useState, useRef, useEffect } from "react";

// Fallback to localhost for development if env var not set
const API_BASE = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("checking");
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check backend connection on component mount
  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus(data.services_initialized ? "connected" : "degraded");
      } else {
        setConnectionStatus("error");
      }
    } catch (error) {
      console.log("Backend connection check failed:", error);
      setConnectionStatus("offline");
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { sender: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input.trim();
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: currentInput }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      const botMessage = {
        sender: "bot",
        text: data.answer || "⚠️ Sorry, I couldn't generate an answer.",
        sources: data.sources || [],
        timestamp: data.timestamp || new Date().toISOString(),
        retrievalInfo: data.retrieval_info || {}
      };

      setMessages((prev) => [...prev, botMessage]);
      setConnectionStatus("connected");
      
    } catch (error) {
      console.error("Error sending message:", error);
      
      let errorMessage = "❌ Error connecting to backend.";
      if (error.message.includes("Failed to fetch")) {
        errorMessage = "🌐 Cannot reach the server. Please check your connection.";
        setConnectionStatus("offline");
      } else if (error.message.includes("500")) {
        errorMessage = "⚠️ Server error. The AI service might be initializing.";
        setConnectionStatus("error");
      }

      setMessages((prev) => [
        ...prev,
        { 
          sender: "bot", 
          text: errorMessage,
          sources: [],
          isError: true 
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (messageIndex, approved) => {
    try {
      const message = messages[messageIndex];
      if (message.sender !== "bot") return;

      const userQuestion = messageIndex > 0 ? messages[messageIndex - 1].text : "";
      
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          answer: message.text,
          approved: approved,
          sources: message.sources || []
        })
      });

      // Update message to show feedback was submitted
      setMessages(prev => prev.map((msg, idx) => 
        idx === messageIndex 
          ? { ...msg, feedbackSubmitted: true, userApproval: approved }
          : msg
      ));

    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  const getConnectionStatusIndicator = () => {
    switch (connectionStatus) {
      case "connected":
        return { color: "#10b981", text: "Connected", icon: "🟢" };
      case "checking":
        return { color: "#f59e0b", text: "Connecting...", icon: "🟡" };
      case "offline":
        return { color: "#ef4444", text: "Offline", icon: "🔴" };
      case "error":
        return { color: "#f97316", text: "Server Issues", icon: "🟠" };
      case "degraded":
        return { color: "#f59e0b", text: "Limited Service", icon: "🟡" };
      default:
        return { color: "#6b7280", text: "Unknown", icon: "⚪" };
    }
  };

  const status = getConnectionStatusIndicator();

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🐾 VetIOS Assistant</h1>
        <div className="connection-status" style={{ color: status.color }}>
          <span>{status.icon} {status.text}</span>
        </div>
      </div>
      
      <div className="chat-box">
        {messages.length === 0 && (
          <div className="welcome-message">
            <p>👋 Welcome to VetIOS! I'm your AI veterinary assistant.</p>
            <p>Ask me about animal health, treatments, diseases, or veterinary procedures.</p>
            <p>💡 Try asking: "What are the symptoms of kennel cough?" or "How do I treat a dog with parvo?"</p>
          </div>
        )}
        
        {messages.map((msg, i) => (
          <div key={i} className="message-container">
            <div
              className={`message ${msg.sender === "user" ? "user" : "bot"} ${msg.isError ? "error" : ""}`}
            >
              <p>{msg.text}</p>
              
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <p className="sources-title">📚 Sources:</p>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        <div className="source-item">
                          <strong>{source.title || "Veterinary Resource"}:</strong>
                          <span className="source-content"> {source.content || "No content available"}</span>
                          {source.category && (
                            <span className="source-category">Category: {source.category}</span>
                          )}
                          {source.species && source.species !== "multiple" && (
                            <span className="source-species">Species: {source.species}</span>
                          )}
                          {source.url && source.url !== "N/A" && (
                            <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-link">
                              View Source
                            </a>
                          )}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {msg.sender === "bot" && !msg.isError && !msg.feedbackSubmitted && (
                <div className="feedback-buttons">
                  <p className="feedback-title">Was this response helpful?</p>
                  <button 
                    onClick={() => submitFeedback(i, true)}
                    className="feedback-btn helpful"
                    title="Yes, this was helpful"
                  >
                    👍 Helpful
                  </button>
                  <button 
                    onClick={() => submitFeedback(i, false)}
                    className="feedback-btn not-helpful"
                    title="No, this wasn't helpful"
                  >
                    👎 Not Helpful
                  </button>
                </div>
              )}

              {msg.feedbackSubmitted && (
                <div className="feedback-thanks">
                  <p>✅ Thank you for your feedback!</p>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message bot loading-message">
            <p>⏳ VetIOS is analyzing your question...</p>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-form" onSubmit={sendMessage}>
        <input
          type="text"
          placeholder="Ask me about veterinary medicine..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          maxLength={500}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? "..." : "Send"}
        </button>
      </form>
      
      <div className="chat-footer">
        <small>
          VetIOS AI Veterinary Assistant | Status: {status.text}
          {messages.length > 0 && (
            <span> | {messages.filter(m => m.sender === 'bot' && !m.isError).length} responses generated</span>
          )}
        </small>
      </div>
    </div>
  );
}

export default ChatBox;
