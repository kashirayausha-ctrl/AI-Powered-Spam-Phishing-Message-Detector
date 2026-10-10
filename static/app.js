const form = document.getElementById("analyze-form");
const messageInput = document.getElementById("message");
const senderInput = document.getElementById("sender");
const count = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const emptyState = document.getElementById("empty-state");
const results = document.getElementById("results");

const examples = {
  legitimate: {
    sender: "project.team@example.edu",
    message: "Hi team, our project review is scheduled for tomorrow at 10 AM in Lab 3. Please bring your laptop and the latest report. Reply here if you cannot attend."
  },
  spam: {
    sender: "offers@promo.example",
    message: "FLASH SALE! Congratulations, you have been selected for a free gift. Claim your reward today and enjoy 70% off our limited-time offers. Subscribe now!"
  },
  phishing: {
    sender: "security-alert@example-mail.test",
    message: "URGENT: Your account will be suspended today. Verify your password and OTP immediately at http://192.0.2.10/account-check to prevent account closure."
  }
};

messageInput.addEventListener("input", () => {
  count.textContent = `${messageInput.value.length} / 12000`;
});

document.querySelectorAll("[data-example]").forEach(button => {
  button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    senderInput.value = example.sender;
    messageInput.value = example.message;
    count.textContent = `${messageInput.value.length} / 12000`;
    messageInput.focus();
  });
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function labelClass(label) {
  return label.toLowerCase();
}

function renderResult(data, originalMessage) {
  const label = escapeHtml(data.label);
  const labelCls = labelClass(data.label);
  const scoreColor = data.risk_level === "High" ? "#ff7584" : data.risk_level === "Medium" ? "#ffc66d" : "#69e6ac";
  const probs = ["Legitimate", "Spam", "Phishing"].map(name => {
    const value = Number(data.class_probabilities[name] || 0);
    return `<div class="prob-row ${name.toLowerCase()}"><span>${name}</span><div class="prob-track"><div class="prob-fill" style="width:${value}%"></div></div><span class="prob-value">${value.toFixed(1)}%</span></div>`;
  }).join("");

  const indicators = data.indicators.map(item => `
    <div class="indicator">
      <span class="indicator-dot ${escapeHtml(item.severity)}"></span>
      <div><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(item.detail)}</p></div>
    </div>`).join("");

  results.innerHTML = `
    <div class="result-top">
      <div><div class="section-kicker">ANALYSIS COMPLETE</div><h2>${label} message</h2><p>${escapeHtml(data.explanation)}</p></div>
      <span class="status-pill ${labelCls}">${label.toUpperCase()}</span>
    </div>
    <div class="score-area">
      <div class="score-circle" style="--score:${data.risk_score};--score-color:${scoreColor}"><div class="score-value">${data.risk_score}<small>RISK / 100</small></div></div>
      <div class="score-copy"><strong>${escapeHtml(data.risk_level)} risk</strong><p>Combined screening score based on the model estimate and visible warning indicators. It is not a probability of compromise.</p><div class="confidence"><span>MODEL CONFIDENCE</span><span>${Number(data.confidence).toFixed(1)}%</span></div><div class="confidence-track"><div class="confidence-fill" style="width:${Math.max(0,Math.min(100,data.confidence))}%"></div></div></div>
    </div>
    <div class="block-heading">CLASS PROBABILITIES</div><div class="probability-list">${probs}</div>
    <div class="block-heading">SUSPICIOUS INDICATORS</div><div class="indicator-list">${indicators}</div>
    <div class="disclaimer">${escapeHtml(data.disclaimer)}</div>
    <div class="feedback-area"><p>Is this classification correct? Your feedback is saved locally for project evaluation.</p>
      <div class="feedback-actions">
        <button class="feedback-btn" data-feedback="legitimate">Legitimate</button>
        <button class="feedback-btn" data-feedback="spam">Spam</button>
        <button class="feedback-btn" data-feedback="phishing">Phishing</button>
      </div><div id="feedback-message" class="feedback-message" role="status"></div>
    </div>`;

  emptyState.classList.add("hidden");
  results.classList.remove("hidden");

  results.querySelectorAll("[data-feedback]").forEach(button => {
    button.addEventListener("click", async () => {
      const messageBox = document.getElementById("feedback-message");
      messageBox.textContent = "Saving feedback…";
      try {
        const response = await fetch("/api/feedback", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            message: originalMessage,
            predicted_label: data.label,
            corrected_label: button.dataset.feedback
          })
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Could not save feedback.");
        messageBox.textContent = body.message;
      } catch (error) {
        messageBox.textContent = error.message;
      }
    });
  });
  document.getElementById("result-panel").scrollIntoView({behavior:"smooth", block:"nearest"});
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    messageInput.focus();
    return;
  }
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector("span:first-child").textContent = "Analyzing…";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message, sender: senderInput.value.trim()})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    renderResult(data, message);
  } catch (error) {
    emptyState.classList.remove("hidden");
    results.classList.add("hidden");
    emptyState.querySelector("h2").textContent = "Unable to analyze";
    emptyState.querySelector("p").textContent = error.message;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.querySelector("span:first-child").textContent = "Analyze message";
  }
});
