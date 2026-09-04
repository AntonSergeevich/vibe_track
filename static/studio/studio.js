/* VibeTrack studio — загрузка трека, запуск рендера, партитура, запись голоса. */
(() => {
  const $ = (sel) => document.querySelector(sel);
  const api = (path) => `/api/${path}`;
  let state = { track: null, job: null, poll: null, score: null };

  async function request(path, { method = 'GET', body = null, form = null } = {}) {
    const opts = { method, headers: { 'X-CSRFToken': window.CSRF_TOKEN } };
    if (form) opts.body = form;
    else if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(api(path), opts);
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
    if (!res.ok) throw new Error(data.detail || `Ошибка ${res.status}`);
    return data;
  }

  /* ---------------------------------------------------------- возможности */
  request('capabilities/').then((caps) => {
    const b = caps.backends || {};
    const badge = (info, onText, offText) => {
      if (!info) return offText;
      const state = info.loaded ? ' (загружена)' : '';
      return info.available ? `${onText}: ${info.model}${state}` : offText;
    };
    $('#backends').innerHTML = [
      `Разделение дорожек: <b>${badge(b.demucs, 'Demucs', 'DSP (базовое)')}</b>`,
      `Текст песни: <b>${badge(b.whisper, 'Whisper', 'вручную')}</b>`,
      `Разбор описания: <b>${badge(b.llm, 'Claude', 'по ключевым словам')}</b>`,
    ].join(' · ');
    $('#refs').innerHTML = (caps.references || [])
      .map((r) => `<button type="button" class="chip">${r}</button>`).join('');
    $('#refs').addEventListener('click', (e) => {
      if (!e.target.classList.contains('chip')) return;
      const t = $('#prompt');
      t.value = t.value ? `${t.value.replace(/\s+$/, '')}, в духе ${e.target.textContent}`
                        : `ню-метал в духе ${e.target.textContent}`;
    });
  }).catch(() => {});

  /* ------------------------------------------------------------- загрузка */
  const dropzone = $('#dropzone');
  $('#pick').addEventListener('click', () => $('#file').click());
  dropzone.addEventListener('click', (e) => { if (e.target === dropzone) $('#file').click(); });
  ['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove('over');
  }));
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  });
  $('#file').addEventListener('change', (e) => e.target.files[0] && upload(e.target.files[0]));

  async function upload(file) {
    const info = $('#track-info');
    info.classList.remove('hidden');
    info.textContent = `Загружаю «${file.name}» (${(file.size / 1048576).toFixed(1)} МБ)…`;
    const form = new FormData();
    form.append('file', file);
    form.append('title', file.name.replace(/\.[^.]+$/, ''));
    try {
      state.track = await request('tracks/upload/', { method: 'POST', form });
      info.innerHTML = `Загружено: <b>${state.track.title}</b> (трек #${state.track.id})`;
      $('#run').disabled = false;
    } catch (err) {
      info.textContent = `Не получилось: ${err.message}`;
    }
  }

  /* --------------------------------------------------------------- рендер */
  $('#aggression').addEventListener('input', (e) => { $('#aggr-out').value = e.target.value; });

  function collectOverrides() {
    return {
      tuning: $('#tuning').value,
      bass_tuning: $('#bass_tuning').value,
      groove: $('#groove').value,
      aggression: parseFloat($('#aggression').value),
      instruments: [...document.querySelectorAll('input[name=instrument]:checked')].map((i) => i.value),
      vocals: {
        male: $('#v-male').checked,
        female: $('#v-female').checked,
        male_style: $('#v-male-style').value,
        female_style: $('#v-female-style').value,
        autotune: $('#v-autotune').checked,
      },
    };
  }

  $('#run').addEventListener('click', async () => {
    if (!state.track) return;
    $('#run').disabled = true;
    $('#progress').classList.remove('hidden');
    setProgress(1, 'ставлю в очередь');
    try {
      state.job = await request(`tracks/${state.track.id}/transform/`, {
        method: 'POST',
        body: {
          prompt: $('#prompt').value,
          overrides: collectOverrides(),
          options: { manual_lyrics: $('#manual-lyrics').value },
        },
      });
      pollJob();
    } catch (err) {
      setProgress(0, `ошибка: ${err.message}`);
      $('#run').disabled = false;
    }
  });

  function setProgress(pct, stage) {
    $('#progress .bar span').style.width = `${pct}%`;
    $('#progress .stage').textContent = `${stage} — ${pct}%`;
  }

  function pollJob() {
    clearInterval(state.poll);
    state.poll = setInterval(async () => {
      try {
        const st = await request(`renders/${state.job.id}/status/`);
        setProgress(st.progress, st.stage || st.status);
        if (st.status === 'done') { clearInterval(state.poll); showResult(); }
        if (st.status === 'error') {
          clearInterval(state.poll);
          setProgress(0, `ошибка: ${st.error?.split('\n')[0] || 'неизвестно'}`);
          $('#run').disabled = false;
        }
      } catch (err) { /* сеть моргнула — попробуем на следующем тике */ }
    }, 1500);
  }

  async function showResult() {
    const job = await request(`renders/${state.job.id}/`);
    state.job = job;
    $('#run').disabled = false;
    $('#step-result').classList.remove('hidden');
    $('#step-vocal').classList.remove('hidden');
    $('#warnings').innerHTML = (job.warnings || []).map((w) => `⚠ ${w}`).join('<br>');
    $('#master').src = job.master_url;
    $('#master-dl').href = job.master_url;

    const a = job.result?.analysis || {};
    const spec = job.spec || {};
    $('#analysis').innerHTML = [
      `Темп: <b>${Math.round(a.tempo || 0)} BPM</b>`,
      `Тональность: <b>${a.key_name || '—'}</b>`,
      `Строй: <b>${spec.tuning || '—'}</b>`,
      `Грув: <b>${spec.groove || '—'}</b>`,
      `Длительность: <b>${Math.round(a.duration || 0)} с</b>`,
    ].join('');

    $('#stems').innerHTML = (job.stems || []).map((s) => `
      <div class="stem">
        <div><div class="name">${s.label || s.name}</div>
             <div class="lvl">${s.rms_db ?? '—'} dB RMS</div></div>
        <audio controls preload="none" src="${s.file_url}"></audio>
        <a class="btn" href="${s.file_url}" download>↓</a>
      </div>`).join('');

    renderScore(job.score || {});
    $('#step-result').scrollIntoView({ behavior: 'smooth' });
  }

  function renderScore(score) {
    state.score = score;
    const views = {};
    if (score.lyric_sheet) views['Текст с аккордами'] = score.lyric_sheet;
    if (score.chord_chart) views['Аккорды'] = score.chord_chart;
    Object.entries(score.tabs || {}).forEach(([name, tab]) => { views[`Таб: ${name}`] = tab; });
    const names = Object.keys(views);
    if (!names.length) {
      $('#score-nav').innerHTML = '';
      $('#score-view').textContent = 'Партитура пуста: добавь текст песни или включи инструменты.';
      return;
    }
    $('#score-nav').innerHTML = names
      .map((n, i) => `<button data-name="${n}" class="${i ? '' : 'active'}">${n}</button>`).join('');
    $('#score-view').textContent = views[names[0]];
    $('#score-nav').onclick = (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      [...$('#score-nav').children].forEach((b) => b.classList.toggle('active', b === btn));
      $('#score-view').textContent = views[btn.dataset.name];
    };
  }

  /* -------------------------------------------------------- запись голоса */
  let recorder = null; let chunks = []; let timer = null; let started = 0;
  $('#take-autotune').addEventListener('input', (e) => {
    $('#take-at-out').value = e.target.value < 0 ? 'авто' : e.target.value;
  });

  $('#rec').addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        sendTake(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
      };
      recorder.start();
      started = Date.now();
      timer = setInterval(() => {
        const s = Math.floor((Date.now() - started) / 1000);
        $('#rec-time').textContent =
          `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
      }, 250);
      $('#rec').disabled = true; $('#stop').disabled = false;
      if ($('#play-along').checked && $('#master').src) { $('#master').currentTime = 0; $('#master').play(); }
    } catch (err) {
      $('#take-result').innerHTML = `<p class="warnings">Нет доступа к микрофону: ${err.message}</p>`;
    }
  });

  $('#stop').addEventListener('click', () => {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    clearInterval(timer);
    $('#rec').disabled = false; $('#stop').disabled = true;
    $('#master').pause();
  });

  async function sendTake(blob) {
    const out = $('#take-result');
    out.innerHTML = '<p>Обрабатываю дубль…</p>';
    const form = new FormData();
    form.append('raw_file', blob, 'take.webm');
    form.append('track', state.track.id);
    if (state.job) form.append('job', state.job.id);
    form.append('style', $('#take-style').value);
    form.append('gender', $('#take-gender').value);
    if ($('#take-target').value) form.append('target_gender', $('#take-target').value);
    const at = parseFloat($('#take-autotune').value);
    if (at >= 0) form.append('autotune', at);
    try {
      const take = await request('vocal-takes/', { method: 'POST', form });
      pollTake(take.id);
    } catch (err) {
      out.innerHTML = `<p class="warnings">Не вышло: ${err.message}</p>`;
    }
  }

  function pollTake(id) {
    const out = $('#take-result');
    const iv = setInterval(async () => {
      try {
        const take = await request(`vocal-takes/${id}/`);
        if (take.status === 'done') {
          clearInterval(iv);
          out.innerHTML = `
            <p class="note">${(take.notes || []).join(' · ') || 'Дубль обработан.'}</p>
            <label>Обработанный голос<audio controls src="${take.processed_url}"></audio></label>
            <label>Микс с минусовкой<audio controls src="${take.mixed_url}"></audio></label>
            <a class="btn" href="${take.mixed_url}" download>Скачать микс</a>`;
        } else if (take.status === 'error') {
          clearInterval(iv);
          out.innerHTML = `<p class="warnings">Ошибка: ${take.error}</p>`;
        }
      } catch (err) { /* повторим на следующем тике */ }
    }, 1500);
  }
})();
