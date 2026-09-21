(() => {
  const signal = window.CFX_SIGNAL;
  const leads = window.CFX_LEADS;
  const rate = window.CFX_SAMPLE_RATE;
  const fileInput = document.getElementById('ecg_file');
  const fileName = document.getElementById('file-name');
  const time = Array.from({ length: signal[0].length }, (_, i) => i / rate);
  const traces = signal.map((lead, i) => ({ x: time, y: lead, name: leads[i], mode: 'lines', visible: true, line: { width: 1.1, color: i % 2 ? '#80e3ba' : '#6ad9df' } }));
  const layout = { template: 'plotly_dark', paper_bgcolor: 'transparent', plot_bgcolor: '#071317', hovermode: 'x unified', height: 520, margin: { t: 18, r: 18, b: 45, l: 55 }, xaxis: { title: 'Time (s)', gridcolor: '#193035' }, yaxis: { title: 'Amplitude', gridcolor: '#193035' }, legend: { orientation: 'h' } };
  Plotly.newPlot('ecg-plot', traces, layout, { responsive: true, displaylogo: false });
  const focusIndex = () => Number(document.getElementById('focus-lead').value);
  const drawFocus = () => {
    const index = focusIndex();
    Plotly.react('focus-plot', [{ x: time, y: signal[index], mode: 'lines', line: { color: '#f3bd69', width: 1.4 }, name: leads[index] }], { ...layout, height: 280, title: `Focused lead ${leads[index]}` }, { responsive: true, displaylogo: false });
  };
  drawFocus();
  document.querySelectorAll('.lead-toggle').forEach((box) => box.addEventListener('change', (event) => Plotly.restyle('ecg-plot', { visible: event.target.checked }, [Number(event.target.dataset.lead)])));
  document.getElementById('focus-lead').addEventListener('change', drawFocus);
  document.getElementById('reset-view').addEventListener('click', () => { Plotly.relayout('ecg-plot', { 'xaxis.autorange': true, 'yaxis.autorange': true }); Plotly.relayout('focus-plot', { 'xaxis.autorange': true, 'yaxis.autorange': true }); });
  fileInput.addEventListener('change', () => { fileName.textContent = fileInput.files[0]?.name || 'Choose ECG file'; });

  const postExport = (url, payload, filename, type) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then((response) => response.blob()).then((blob) => { const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = filename; link.click(); URL.revokeObjectURL(link.href); });
  const exportPayload = { result: window.CFX_RESULT, quality: window.CFX_QUALITY, model: window.CFX_RESULT.model };
  document.querySelector('.export-json')?.addEventListener('click', () => postExport('/api/export/json', exportPayload, 'cardiofusionx-result.json', 'application/json'));
  document.querySelector('.export-csv')?.addEventListener('click', () => postExport('/api/export/csv', exportPayload, 'cardiofusionx-predictions.csv', 'text/csv'));

  const intensity = document.getElementById('noise-intensity');
  const noiseValue = document.getElementById('noise-value');
  intensity?.addEventListener('input', () => { noiseValue.textContent = Number(intensity.value).toFixed(2); });
  const useCurrentSignal = () => Array.isArray(window.CFX_SIGNAL) && window.CFX_SIGNAL.length > 0;
  document.getElementById('stress-test')?.addEventListener('click', async () => {
    const output = document.getElementById('stress-output');
    if (!fileInput.files[0] && !useCurrentSignal()) { output.textContent = 'Keep the uploaded file selected or analyze a signal first to run a comparison.'; return; }
    output.textContent = 'Running controlled comparison...';
    if (fileInput.files[0]) {
      const body = new FormData(); body.append('ecg_file', fileInput.files[0]); body.append('noise_type', document.getElementById('noise-type').value); body.append('intensity', intensity.value); body.append('model_key', window.CFX_RESULT.model.key); body.append('threshold_strategy', window.CFX_RESULT.threshold_strategy);
      const response = await fetch('/api/stress-test', { method: 'POST', body }); const data = await response.json();
      output.textContent = data.error || `${data.changed_labels.length} labels changed by the controlled ${data.noise_type} perturbation. Quality: ${data.quality_before.status} -> ${data.quality_after.status}.`;
      return;
    }
    const payload = { signal: window.CFX_SIGNAL, filename: 'current-signal.mat', noise_type: document.getElementById('noise-type').value, intensity: intensity.value, model_key: window.CFX_RESULT.model.key, threshold_strategy: window.CFX_RESULT.threshold_strategy };
    const response = await fetch('/api/stress-test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const data = await response.json();
    output.textContent = data.error || `${data.changed_labels.length} labels changed by the controlled ${data.noise_type} perturbation. Quality: ${data.quality_before.status} -> ${data.quality_after.status}.`;
  });
  document.getElementById('counterfactual')?.addEventListener('click', async () => {
    const output = document.getElementById('counter-output');
    if (!fileInput.files[0] && !useCurrentSignal()) { output.textContent = 'Keep the uploaded file selected or analyze a signal first to run a comparison.'; return; }
    output.textContent = 'Comparing modified segment...';
    if (fileInput.files[0]) {
      const body = new FormData(); body.append('ecg_file', fileInput.files[0]); body.append('start', document.getElementById('counter-start').value); body.append('end', document.getElementById('counter-end').value); body.append('model_key', window.CFX_RESULT.model.key); body.append('threshold_strategy', window.CFX_RESULT.threshold_strategy);
      const response = await fetch('/api/counterfactual', { method: 'POST', body }); const data = await response.json();
      output.textContent = data.error || `Prediction changed after signal modification in samples ${data.start_sample}-${data.end_sample}. ${data.changed_labels.length} labels changed. This is not a causal explanation.`;
      return;
    }
    const payload = { signal: window.CFX_SIGNAL, filename: 'current-signal.mat', start: document.getElementById('counter-start').value, end: document.getElementById('counter-end').value, model_key: window.CFX_RESULT.model.key, threshold_strategy: window.CFX_RESULT.threshold_strategy };
    const response = await fetch('/api/counterfactual', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const data = await response.json();
    output.textContent = data.error || `Prediction changed after signal modification in samples ${data.start_sample}-${data.end_sample}. ${data.changed_labels.length} labels changed. This is not a causal explanation.`;
  });
})();
