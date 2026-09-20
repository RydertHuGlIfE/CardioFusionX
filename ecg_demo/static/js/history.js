(() => {
  const form = document.getElementById('history-form');
  if (!form) return;
  const loading = document.getElementById('history-loading');
  const result = document.getElementById('history-result');
  const showResult = (data) => {
    result.hidden = false;
    if (data.error) {
      result.className = 'history-result error-result';
      result.textContent = data.error;
      return;
    }
    const indicators = (data.feature_indicators || []).map((item) => `<li>${item.feature}: ${item.direction} (${item.magnitude})</li>`).join('');
    result.className = `history-result ${data.prediction ? 'elevated' : 'lower'}`;
    result.innerHTML = `<div><span class="result-label">${data.classification}</span><strong>${(data.probability * 100).toFixed(1)}%</strong></div><p>${data.probability_label}</p>${indicators ? `<h3>Model input indicators</h3><ul>${indicators}</ul>` : ''}<small>${data.disclaimer}</small>`;
  };
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    loading.hidden = false;
    result.hidden = true;
    try {
      const payload = Object.fromEntries(new FormData(form).entries());
      const response = await fetch('/predict-history', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      showResult(await response.json());
    } catch (error) {
      showResult({ error: 'The patient-history request failed. Check the server connection.' });
    } finally {
      loading.hidden = true;
    }
  });
  document.getElementById('history-reset')?.addEventListener('click', () => { result.hidden = true; result.textContent = ''; });
})();
