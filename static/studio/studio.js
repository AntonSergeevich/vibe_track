/* VibeTrack studio: загрузка с прогрессом, выбор стиля, рендер, запись голоса. */
(() => {
  const $ = (sel) => document.querySelector(sel);
  const api = (path) => `/api/${path}`;
  const state = { track: null, job: null, poll: null, genre: 'nu_metal' };
  const STORE_KEY = 'vibetrack.session';

  /* Состояние переживает перезагрузку страницы: рендер идёт на сервере, и
     закрытая вкладка не должна означать потерянный трек. */
  function remember() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        track: state.track ? { id: state.track.id, title: state.track.title } : null,
        job: state.job ? state.job.id : null,
        genre: state.genre,
      }));
    } catch { /* приватный режим — просто не запоминаем */ }
  }

  function forget() {
    try { localStorage.removeItem(STORE_KEY); } catch { /* нечего чистить */ }
  }

  async function restore() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch { return; }
    if (!saved || !saved.track) return;

    state.track = saved.track;
    state.genre = saved.genre || 'nu_metal';
    document.querySelectorAll('.genre').forEach((g) =>
      g.classList.toggle('active', g.dataset.genre === state.genre));
    $('#track-name').textContent = saved.track.title;
    $('#track-meta').textContent = `трек #${saved.track.id} · восстановлен`;
    $('#track-chip').classList.remove('hidden');
    dropzone.classList.add('hidden');
    $('#step-style').classList.remove('hidden');

    if (!saved.job) return;
    try {
      const st = await request(`renders/${saved.job}/status/`);
      state.job = { id: saved.job };
      if (st.status === 'done') {
        showResult();
      } else if (st.status === 'error') {
        $('#progress').classList.remove('hidden');
        setProgress(0, 'error', `Ошибка: ${(st.error || '').split('\n')[0]}`);
      } else {
        $('#progress').classList.remove('hidden');
        $('#run').disabled = true;
        setProgress(st.progress, st.stage || st.status);
        pollJob();                      // обработка идёт дальше — просто снова следим
      }
    } catch {
      forget();                         // рендер удалён или чужой — начинаем заново
    }
  }

  const STAGE_NAMES = {
    load: 'Читаю файл', analyze: 'Определяю темп и тональность',
    separate: 'Разделяю на дорожки', transcribe: 'Распознаю текст',
    arrange: 'Собираю аранжировку', render: 'Играю инструменты',
    vocals: 'Обрабатываю вокал', mix: 'Свожу микс', score: 'Пишу табы',
    export: 'Сохраняю файлы', done: 'Готово', queued: 'В очереди',
  };

  async function request(path, { method = 'GET', body = null } = {}) {
    const opts = { method, headers: { 'X-CSRFToken': window.CSRF_TOKEN } };
    if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(api(path), opts);
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
    if (!res.ok) throw Object.assign(new Error(data.detail || `Ошибка ${res.status}`), { data, status: res.status });
    return data;
  }

  /* Названия моделей в шапке пользователю ничего не говорят: он пришёл за
     треком, а не за списком библиотек. Обещание результата полезнее. */

  /* ------------------------------------------------ загрузка с прогрессом */
  const dropzone = $('#dropzone');
  $('#pick').addEventListener('click', (e) => { e.stopPropagation(); $('#file').click(); });
  dropzone.addEventListener('click', () => $('#file').click());
  ['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove('over');
  }));
  dropzone.addEventListener('drop', (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
  $('#file').addEventListener('change', (e) => e.target.files[0] && upload(e.target.files[0]));
  $('#change-track').addEventListener('click', () => {
    $('#track-chip').classList.add('hidden');
    dropzone.classList.remove('hidden');
    $('#file').value = '';
    state.track = null;
    clearResult();
    forget();
  });

  /* Новый трек — чистый лист. Оставленные на странице дорожки и аккорды от
     прошлого трека путают сильнее, чем пустой экран: человек слушает старое
     и думает, что это результат нового. */
  function clearResult() {
    clearInterval(state.poll);
    state.poll = null;
    state.job = null;

    document.querySelectorAll('#step-result audio, #step-vocal audio')
      .forEach((el) => { el.pause(); el.removeAttribute('src'); el.load(); });

    $('#step-result').classList.add('hidden');
    $('#step-vocal').classList.add('hidden');
    $('#progress').classList.add('hidden');
    $('#stems').innerHTML = '';
    $('#warnings').innerHTML = '';
    $('#analysis').innerHTML = '';
    $('#score-nav').innerHTML = '';
    $('#score-view').textContent = '';
    $('#master-dl').removeAttribute('href');
    $('#cover-dl').removeAttribute('href');
    $('#cover-box').classList.add('hidden');
    $('#run').disabled = false;
    setProgress(0, 'queued');
  }

  function upload(file) {
    clearResult();
    const box = $('#upload-progress');
    const bar = box.querySelector('.bar span');
    box.classList.remove('hidden');
    dropzone.classList.add('hidden');
    $('#upload-name').textContent = file.name;
    bar.style.width = '0%';
    $('#upload-pct').textContent = '0%';

    const form = new FormData();
    form.append('file', file);
    form.append('title', file.name.replace(/\.[^.]+$/, ''));

    const xhr = new XMLHttpRequest();
    xhr.open('POST', api('tracks/upload/'));
    xhr.setRequestHeader('X-CSRFToken', window.CSRF_TOKEN);
    // прогресс отдаёт только XHR: fetch о ходе отправки ничего не сообщает
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      bar.style.width = `${pct}%`;
      $('#upload-pct').textContent = `${pct}%`;
    };
    xhr.upload.onload = () => { $('#upload-pct').textContent = 'обрабатываю…'; };
    xhr.onload = () => {
      box.classList.add('hidden');
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
      if (xhr.status !== 201) {
        dropzone.classList.remove('hidden');
        alert(data.detail || `Не удалось загрузить (${xhr.status})`);
        return;
      }
      state.track = data;
      state.job = null;
      remember();
      $('#track-name').textContent = data.title;
      $('#track-meta').textContent = `${(file.size / 1048576).toFixed(1)} МБ · трек #${data.id}`;
      $('#track-chip').classList.remove('hidden');
      $('#step-style').classList.remove('hidden');
      $('#step-style').scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    xhr.onerror = () => {
      box.classList.add('hidden'); dropzone.classList.remove('hidden');
      alert('Сеть оборвалась при загрузке');
    };
    xhr.send(form);
  }

  /* ------------------------------------------------------------- стили */
  $('#genres').addEventListener('click', (e) => {
    const card = e.target.closest('.genre');
    if (!card) return;
    document.querySelectorAll('.genre').forEach((g) => g.classList.toggle('active', g === card));
    state.genre = card.dataset.genre;
    remember();
  });

  $('#aggression').addEventListener('input', (e) => { $('#aggr-out').value = e.target.value; });
  $('#transpose').addEventListener('input', (e) => {
    const v = parseInt(e.target.value, 10);
    $('#transpose-out').value = v === 0 ? 'как в оригинале'
      : `${v > 0 ? '+' : ''}${v} полутон${Math.abs(v) === 1 ? '' : 'а'}`;
  });
  $('#take-autotune').addEventListener('input', (e) => {
    $('#take-at-out').value = e.target.value < 0 ? 'выключен' : e.target.value;
  });

  function collectOverrides() {
    const touched = $('#step-style').querySelector('details').open;
    const overrides = { genre: state.genre };
    if (!touched) return overrides;      // не трогал настройки — пусть решает жанр
    return {
      ...overrides,
      tuning: $('#tuning').value,
      bass_tuning: $('#bass_tuning').value,
      groove: $('#groove').value,
      aggression: parseFloat($('#aggression').value),
      transpose: parseInt($('#transpose').value, 10),
      instruments: [...document.querySelectorAll('input[name=instrument]:checked')].map((i) => i.value),
      vocals: {
        male: $('#v-male').checked,
        female: $('#v-female').checked,
        male_style: $('#v-male-style').value,
        female_style: $('#v-female-style').value,
        autotune: $('#v-autotune').checked,
        harmony: $('#v-harmony').checked,
      },
    };
  }

  /* ------------------------------------------------------------ рендер */
  $('#run').addEventListener('click', async () => {
    if (!state.track) return;
    clearResult();                       // повторный рендер того же трека — тоже с нуля
    $('#run').disabled = true;
    $('#progress').classList.remove('hidden');
    setProgress(1, 'queued');
    try {
      state.job = await request(`tracks/${state.track.id}/transform/`, {
        method: 'POST',
        body: {
          prompt: $('#prompt').value,
          overrides: collectOverrides(),
          options: {
            manual_lyrics: $('#manual-lyrics').value,
            generate_cover: $('#want-cover').checked,
          },
        },
      });
      remember();
      pollJob();
    } catch (err) {
      $('#run').disabled = false;
      if (err.status === 402) {
        $('#stage-name').textContent = err.data.detail;
        $('#stage-pct').textContent = '';
      } else {
        setProgress(0, 'error', err.message);
      }
    }
  });

  function setProgress(pct, stage, text) {
    $('#progress .bar span').style.width = `${pct}%`;
    $('#stage-name').textContent = text || STAGE_NAMES[stage] || stage;
    $('#stage-pct').textContent = `${pct}%`;
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
          setProgress(0, 'error', `Ошибка: ${(st.error || '').split('\n')[0]}`);
          $('#run').disabled = false;
        }
      } catch { /* сеть моргнула — попробуем на следующем тике */ }
    }, 1200);
  }

  async function showResult() {
    const job = await request(`renders/${state.job.id}/`);
    state.job = job;
    $('#run').disabled = false;
    $('#step-result').classList.remove('hidden');
    $('#step-vocal').classList.remove('hidden');
    $('#warnings').innerHTML = [
      ...(job.warnings || []).map((w) => `⚠ ${w}`),
      `<a class="chip" href="/cabinet/track/${job.id}/">Открыть страницу трека — она сохранится в кабинете</a>`,
    ].join('<br>');
    $('#master').src = job.master_url;
    $('#master-dl').href = job.master_url;

    const a = job.result?.analysis || {};
    const spec = job.spec || {};
    $('#analysis').innerHTML = [
      `Темп <b>${Math.round(a.tempo || 0)}</b>`,
      `Тональность <b>${a.key_name || '—'}</b>`,
      `Строй <b>${spec.tuning || '—'}</b>`,
      `Грув <b>${spec.groove || '—'}</b>`,
      `Длина <b>${Math.round(a.duration || 0)} с</b>`,
    ].map((t) => `<span>${t}</span>`).join('');

    // кавер — главный результат, ему отдельное место, а не строка среди дорожек
    const stems = job.stems || [];
    const cover = stems.find((s) => s.name === 'cover');
    $('#cover-box').classList.toggle('hidden', !cover);
    if (cover) {
      $('#cover').src = cover.file_url;
      $('#cover-dl').href = cover.file_url;
    }

    $('#stems').innerHTML = stems.filter((s) => s !== cover).map((s) => `
      <div class="stem">
        <div><div class="name">${s.label || s.name}</div>
             <div class="lvl">${s.rms_db ?? '—'} dB RMS</div></div>
        <audio controls preload="none" src="${s.file_url}"></audio>
        <div class="stem-actions"><a class="btn small" href="${s.file_url}" download>↓</a></div>
      </div>`).join('');

    renderScore(job.score || {});
    $('#step-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderScore(score) {
    const views = {};
    if (score.lyric_sheet) views['Текст с аккордами'] = score.lyric_sheet;
    if (score.chord_chart) views['Аккорды'] = score.chord_chart;
    Object.entries(score.tabs || {}).forEach(([name, tab]) => { views[`Таб: ${name}`] = tab; });
    const names = Object.keys(views);
    if (!names.length) {
      $('#score-nav').innerHTML = '';
      $('#score-view').textContent = 'Партитура пуста: включи инструменты или добавь текст песни.';
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

  restore();

  /* ----------------------------------------------------- запись голоса */
  let recorder = null; let chunks = []; let timer = null; let started = 0;

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

    const res = await fetch(api('vocal-takes/'), {
      method: 'POST', headers: { 'X-CSRFToken': window.CSRF_TOKEN }, body: form,
    });
    const data = await res.json();
    if (!res.ok) { out.innerHTML = `<p class="warnings">Не вышло: ${data.detail || res.status}</p>`; return; }
    pollTake(data.id);
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
            <label>Голос<audio controls src="${take.processed_url}"></audio></label>
            <label>Микс<audio controls src="${take.mixed_url}"></audio></label>
            <a class="btn small" href="${take.mixed_url}" download>Скачать микс</a>`;
        } else if (take.status === 'error') {
          clearInterval(iv);
          out.innerHTML = `<p class="warnings">Ошибка: ${take.error}</p>`;
        }
      } catch { /* повторим на следующем тике */ }
    }, 1200);
  }
})();
