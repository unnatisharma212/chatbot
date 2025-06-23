import React, { useState, useRef, useEffect } from "react";

const BACKEND_URL = "http://localhost:5000";

export default function App() {
  const [messages, setMessages] = useState([]); // current session only
  const [input, setInput] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch chat history from backend when panel opens
  useEffect(() => {
    if (showHistory) {
      setLoadingHistory(true);
      fetch(`${BACKEND_URL}/history`)
        .then((res) => res.json())
        .then((data) => {
          setHistory(data.history || []);
          setLoadingHistory(false);
        })
        .catch(() => setLoadingHistory(false));
    }
  }, [showHistory]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { sender: "user", text: input, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages((msgs) => [...msgs, userMsg]);
    const userInput = input;
    setInput("");
    // Send to backend
    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput }),
      });
      const data = await res.json();
      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: data.reply || "(No reply)", timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) },
      ]);
    } catch {
      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: "Error: Could not reach backend.", timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) },
      ]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-200/60 to-purple-200/60 flex flex-col items-center justify-center p-4 relative">
      {/* Floating Chat History Button */}
      <button
        className="fixed bottom-8 right-8 z-30 bg-white/70 backdrop-blur-lg shadow-xl rounded-full p-4 flex items-center gap-2 hover:bg-purple-500 hover:text-white transition-all border border-purple-200 text-purple-700 font-semibold text-lg"
        onClick={() => setShowHistory(true)}
        aria-label="Open Chat History"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M8 17l4 4 4-4m0-5V3m-8 4v10a4 4 0 004 4h4a4 4 0 004-4V7a4 4 0 00-4-4h-4a4 4 0 00-4 4z" /></svg>
        Chat History
      </button>

      {/* Chat History Side Panel */}
      <div className={`fixed top-0 right-0 h-full w-96 bg-white/90 shadow-2xl z-40 transform transition-transform duration-300 ${showHistory ? "translate-x-0" : "translate-x-full"} flex flex-col`}
        style={{backdropFilter: 'blur(16px)'}}>
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-purple-400/80 to-blue-400/80 text-white rounded-tr-3xl">
          <span className="font-bold text-xl tracking-wide">Chat History</span>
          <button className="text-2xl hover:text-red-400 transition" onClick={() => setShowHistory(false)}>&times;</button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {loadingHistory ? (
            <div className="text-gray-400 text-center mt-10">Loading...</div>
          ) : history.length === 0 ? (
            <div className="text-gray-400 text-center mt-10">No chat history yet.</div>
          ) : (
            history.map((msg, idx) => (
              <div key={idx} className="flex items-start gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${msg.sender === "user" ? "bg-blue-200" : "bg-purple-200"} text-xl font-bold`}>
                  {msg.sender === "user" ? "🧑" : "🤖"}
                </div>
                <div>
                  <div className="text-sm font-semibold text-gray-700">{msg.sender === "user" ? "You" : "Bot"}</div>
                  <div className="bg-white/80 border border-gray-100 rounded-xl px-4 py-2 shadow-sm text-gray-700 mt-1">
                    {msg.text}
                  </div>
                  <div className="text-[10px] text-gray-400 mt-1">{msg.timestamp}</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Card */}
      <div className="w-full max-w-2xl bg-white/70 backdrop-blur-2xl rounded-3xl shadow-2xl flex flex-col h-[38rem] border border-purple-100">
        {/* Header */}
        <div className="flex items-center gap-3 px-8 py-6 border-b bg-gradient-to-r from-purple-400/80 to-blue-400/80 text-white rounded-t-3xl">
          <span className="inline-block w-12 h-12 rounded-full bg-white/30 flex items-center justify-center text-3xl shadow">🤖</span>
          <span className="font-bold text-2xl tracking-wide drop-shadow">HCL Chatbot</span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"} items-end gap-2`}> 
              {msg.sender === "bot" && (
                <span className="w-10 h-10 rounded-full bg-purple-200 flex items-center justify-center text-2xl shadow">🤖</span>
              )}
              <div className={`max-w-xs px-5 py-3 rounded-2xl shadow-lg text-base font-medium flex flex-col gap-1 ${msg.sender === "user" ? "bg-blue-500 text-white rounded-br-none" : "bg-purple-100 text-gray-800 rounded-bl-none"}`}>
                {msg.text}
                <span className="text-[10px] text-gray-300 self-end">{msg.timestamp}</span>
              </div>
              {msg.sender === "user" && (
                <span className="w-10 h-10 rounded-full bg-blue-200 flex items-center justify-center text-2xl shadow">🧑</span>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <form onSubmit={sendMessage} className="flex items-center gap-3 px-8 py-6 border-t bg-white/60 rounded-b-3xl">
          <input
            className="flex-1 px-5 py-3 rounded-full border border-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-400 transition text-lg bg-white/80 shadow"
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            autoFocus
          />
          <button
            type="submit"
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white px-8 py-3 rounded-full font-semibold shadow-lg hover:from-purple-600 hover:to-blue-600 transition text-lg"
          >
            Send
          </button>
        </form>
      </div>
      <footer className="mt-8 text-gray-400 text-xs drop-shadow">Made with <span className="text-pink-400">♥</span> using React & Tailwind CSS</footer>
    </div>
  );
}
