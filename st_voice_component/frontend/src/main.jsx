import React, { useEffect, useRef, useState } from "react";
import { Streamlit } from "streamlit-component-lib";

function App() {
  const [listening, setListening] = useState(false);
  const [preview, setPreview] = useState("");
  const recRef = useRef(null);

  useEffect(() => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      Streamlit.setComponentValue({ transcript: "", is_final: true, supported: false });
      return;
    }

    const rec = new SpeechRec();
    rec.lang = "en-IN";
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onstart = () => setListening(true);
    rec.onend = () => setListening(false);

    rec.onresult = (evt) => {
      let combined = "";
      for (let i = 0; i < evt.results.length; i++) combined += evt.results[i][0].transcript;
      const lastIsFinal = evt.results[evt.results.length - 1].isFinal;
      setPreview(combined);
      Streamlit.setComponentValue({ transcript: combined, is_final: lastIsFinal, supported: true });
    };

    rec.onerror = (e) => {
      Streamlit.setComponentValue({ transcript: "", is_final: true, supported: false, error: e.message || String(e) });
    };

    recRef.current = rec;
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(120);

    return () => {
      if (recRef.current) recRef.current.stop();
    };
  }, []);

  const toggle = () => {
    const r = recRef.current;
    if (!r) {
      alert("Speech API not supported in this browser. Use Chrome/Edge.");
      return;
    }
    try {
      if (listening) r.stop();
      else r.start();
    } catch (e) {}
  };

  return (
    <div style={{ color: "#fff", fontFamily: "Inter, Arial", padding: 8 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={toggle} style={{ background: listening ? "#ef4444" : "#2563eb", color: "#fff", border: "none", padding: "8px 12px", borderRadius: 8 }}>
          {listening ? "Stop" : "Start Listening"}
        </button>
        <div style={{ fontSize: 14 }}>{listening ? "Listening..." : "Stopped"}</div>
      </div>
      <div style={{ marginTop: 8, minHeight: 36, color: "#fff", fontSize: 16 }}>{preview}</div>
    </div>
  );
}

export default App;
