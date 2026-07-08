import React, { useEffect, useRef, useState } from "react";

const API_BASE = "/api";

function useAutoScroll(dep) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [dep]);
  return ref;
}

function Message({ role, content, isError }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-[14.5px] leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-slate-700 text-slate-50"
            : isError
            ? "bg-red-950/60 text-red-300 border border-red-900"
            : "bg-slate-800 text-slate-100"
        }`}
      >
        {content}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi, I'm the SafeX Solutions assistant. Ask me about our services, mission, or how to get in touch.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const scrollRef = useAutoScroll(messages.length);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/suggested-questions`)
      .then((r) => r.json())
      .then((d) => setSuggestions(d.questions || []))
      .catch(() => {});
    inputRef.current?.focus();
  }, []);

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const history = messages
      .filter((m) => !m.isError)
      .map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content }));

    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, history }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Something went wrong: ${e.message}`, isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen flex flex-col bg-slate-900">
      <header className="px-5 py-4 border-b border-slate-800">
        <h1 className="text-slate-100 font-semibold text-base">SafeX Solutions</h1>
        <p className="text-slate-400 text-xs">FAQ Assistant</p>
      </header>

      <main
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3 max-w-2xl w-full mx-auto"
      >
        {messages.map((m, i) => (
          <Message key={i} {...m} />
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 rounded-2xl px-4 py-3 flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce" />
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce [animation-delay:0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce [animation-delay:0.3s]" />
            </div>
          </div>
        )}
      </main>

      {suggestions.length > 0 && messages.length < 2 && (
        <div className="max-w-2xl w-full mx-auto px-4 pb-2 flex flex-wrap gap-2">
          {suggestions.map((q) => (
            <button
              key={q}
              onClick={() => sendMessage(q)}
              className="text-xs text-slate-300 border border-slate-700 rounded-full px-3 py-1.5 hover:bg-slate-800 transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage(input);
        }}
        className="border-t border-slate-800 px-4 py-3"
      >
        <div className="max-w-2xl w-full mx-auto flex gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your question..."
            disabled={loading}
            className="flex-1 bg-slate-800 text-slate-100 placeholder:text-slate-500 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-slate-600 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-slate-100 text-slate-900 font-medium rounded-lg px-4 py-2.5 text-sm disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
