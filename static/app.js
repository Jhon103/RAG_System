/**
 * app.js — Lógica frontend del sistema RAG.
 * Maneja interacciones con la API REST, renderizado de resultados
 * y notificaciones.
 */

const API_BASE = '';

// ─── State ──────────────────────────────────────────────
let isLoading = false;

// ─── DOM Ready ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    refreshStats();
    loadFiles();
    setupQueryInput();
});

// ─── Setup ──────────────────────────────────────────────
function setupQueryInput() {
    const input = document.getElementById('queryInput');

    // Auto-resize textarea
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });

    // Submit on Enter (Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });
}

// ─── API Calls ──────────────────────────────────────────

async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) options.body = JSON.stringify(body);

    const response = await fetch(`${API_BASE}${endpoint}`, options);
    const data = await response.json();

    if (data.error && !data.answer && !data.results) {
        throw new Error(data.error);
    }
    return data;
}

// ─── Refresh Stats ──────────────────────────────────────

async function refreshStats() {
    try {
        const stats = await apiCall('/api/stats');
        document.getElementById('statChunks').textContent = stats.total_chunks || 0;
        document.getElementById('statModel').textContent = shortenModel(stats.embedding_model || '—');
        if (stats.config) {
            document.getElementById('statStrategy').textContent =
                `${stats.config.strategy} (${stats.config.chunk_size}c)`;
        }
    } catch (e) {
        console.error('Stats error:', e);
    }

    try {
        const files = await apiCall('/api/files');
        document.getElementById('statFiles').textContent = files.total || 0;
    } catch (e) {
        console.error('Files count error:', e);
    }
}

function shortenModel(name) {
    if (name.length > 16) {
        return name.replace('all-', '').replace('-v2', '');
    }
    return name;
}

// ─── Index Documents ────────────────────────────────────

async function indexDocuments() {
    if (isLoading) return;

    const btn = document.getElementById('btnIndex');
    setButtonLoading(btn, true);
    showLoading('Indexando documentos... Generando embeddings...');

    try {
        const data = await apiCall('/api/index', 'POST');

        showToast(
            `✓ Indexados ${data.chunks_indexed} chunks de ${data.documents_loaded} documentos`,
            'success'
        );
        refreshStats();
        loadFiles();
    } catch (e) {
        showToast(`Error al indexar: ${e.message}`, 'error');
    } finally {
        setButtonLoading(btn, false);
        hideLoading();
    }
}

// ─── Submit Query ───────────────────────────────────────

async function submitQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();

    if (!query) {
        showToast('Escribe una pregunta primero.', 'warning');
        input.focus();
        return;
    }

    if (isLoading) return;

    const btn = document.getElementById('btnQuery');
    setButtonLoading(btn, true);
    showLoading('Buscando chunks relevantes...');

    try {
        const data = await apiCall('/api/query', 'POST', { query });

        if (data.error && !data.results?.length) {
            showToast(data.error, 'warning');
            hideResults();
        } else {
            renderResults(data);
        }
    } catch (e) {
        showToast(`Error en la consulta: ${e.message}`, 'error');
    } finally {
        setButtonLoading(btn, false);
        hideLoading();
    }
}

// ─── Render Results ─────────────────────────────────────

function renderResults(data) {
    const section = document.getElementById('resultsSection');
    section.classList.add('visible');

    // Meta badges
    const metaHtml = `
        <span class="meta-badge confidence">🎯 ${Math.round(data.confidence * 100)}%</span>
        <span class="meta-badge time">⏱ ${data.elapsed_ms}ms</span>
        <span class="meta-badge chunks">🧩 ${data.chunks_used} chunks</span>
    `;
    document.getElementById('answerMeta').innerHTML = metaHtml;

    // Query display
    document.getElementById('answerQuery').textContent = `"${data.query}"`;

    // Answer body — build a clean formatted version from chunks
    const answerBody = document.getElementById('answerBody');
    if (data.results && data.results.length > 0) {
        let formattedAnswer = '';
        const sourceGroups = {};

        data.results.forEach(r => {
            const src = r.source || 'unknown';
            if (!sourceGroups[src]) sourceGroups[src] = [];
            sourceGroups[src].push(r);
        });

        for (const [source, chunks] of Object.entries(sourceGroups)) {
            formattedAnswer += `📄 Fuente: ${source}\n${'─'.repeat(40)}\n\n`;
            chunks.forEach((chunk, i) => {
                const text = chunk.text.trim();
                const displayText = text.length > 600 ? text.substring(0, 600) + '...' : text;
                formattedAnswer += `[${i + 1}] ${displayText}\n\n`;
            });
        }

        formattedAnswer += `${'─'.repeat(50)}\n`;
        formattedAnswer += `📚 Fuentes: ${data.sources.join(', ')}\n`;
        formattedAnswer += `🎯 Confianza: ${Math.round(data.confidence * 100)}% | Similitud promedio: ${(data.avg_similarity * 100).toFixed(1)}%`;

        answerBody.textContent = formattedAnswer;
    } else {
        answerBody.textContent = 'No se encontraron resultados relevantes.';
    }

    // Sources tags
    const sourcesList = document.getElementById('sourcesList');
    if (data.results && data.results.length > 0) {
        const sourcesMap = {};
        data.results.forEach(r => {
            const src = r.source || 'unknown';
            if (!sourcesMap[src] || r.similarity > sourcesMap[src]) {
                sourcesMap[src] = r.similarity;
            }
        });

        sourcesList.innerHTML = Object.entries(sourcesMap).map(([src, sim]) => `
            <span class="source-tag">
                📄 ${src}
                <span class="sim-score">${(sim * 100).toFixed(1)}%</span>
            </span>
        `).join('');
    } else {
        sourcesList.innerHTML = '<span class="source-tag">Sin fuentes</span>';
    }

    // Chunks grid
    const chunksGrid = document.getElementById('chunksGrid');
    if (data.results && data.results.length > 0) {
        chunksGrid.innerHTML = data.results.map((r, i) => {
            const simClass = r.similarity >= 0.5 ? 'high' : r.similarity >= 0.3 ? 'medium' : 'low';
            const simPercent = (r.similarity * 100).toFixed(1);
            const text = r.text.trim();
            const id = `chunk-text-${i}`;

            return `
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span class="chunk-rank">${r.rank || i + 1}</span>
                        <span class="chunk-source">${r.source || 'unknown'}</span>
                        <span class="chunk-similarity ${simClass}">${simPercent}%</span>
                    </div>
                    <div class="chunk-text" id="${id}">${escapeHtml(text)}</div>
                    <div class="similarity-bar-wrapper">
                        <div class="similarity-bar-bg">
                            <div class="similarity-bar-fill ${simClass}" style="width: ${simPercent}%"></div>
                        </div>
                    </div>
                    ${text.length > 200 ? `<button class="chunk-expand" onclick="toggleChunk('${id}', this)">Mostrar más ▼</button>` : ''}
                </div>
            `;
        }).join('');
    } else {
        chunksGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-text">Sin resultados relevantes</div>
                <div class="empty-state-hint">Intenta con otra pregunta o indexa más documentos</div>
            </div>
        `;
    }

    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    document.getElementById('resultsSection').classList.remove('visible');
}

function toggleChunk(id, btn) {
    const el = document.getElementById(id);
    el.classList.toggle('expanded');
    btn.textContent = el.classList.contains('expanded') ? 'Mostrar menos ▲' : 'Mostrar más ▼';
}

// ─── Clear Database ─────────────────────────────────────

async function clearDatabase() {
    if (!confirm('¿Estás seguro? Se eliminarán todos los datos indexados.')) return;

    try {
        await apiCall('/api/clear', 'POST');
        showToast('Base de datos limpiada exitosamente.', 'success');
        refreshStats();
        hideResults();
    } catch (e) {
        showToast(`Error al limpiar: ${e.message}`, 'error');
    }
}

// ─── Load Files ─────────────────────────────────────────

async function loadFiles() {
    try {
        const data = await apiCall('/api/files');
        const list = document.getElementById('filesList');
        const badge = document.getElementById('fileCountBadge');

        badge.textContent = data.total || 0;

        if (!data.files || data.files.length === 0) {
            list.innerHTML = `
                <li class="empty-state">
                    <div class="empty-state-icon">📭</div>
                    <div class="empty-state-text">No hay documentos en ./docs/</div>
                    <div class="empty-state-hint">Añade archivos .txt o .md al directorio docs/</div>
                </li>
            `;
            return;
        }

        list.innerHTML = data.files.map(f => {
            const icon = getFileIcon(f.extension);
            const size = formatBytes(f.size_bytes);
            return `
                <li class="file-item">
                    <span class="file-icon">${icon}</span>
                    <span class="file-name">${escapeHtml(f.name)}</span>
                    <span class="file-ext">${f.extension}</span>
                    <span class="file-size">${size}</span>
                </li>
            `;
        }).join('');
    } catch (e) {
        console.error('Load files error:', e);
    }
}

// ─── UI Helpers ─────────────────────────────────────────

function setButtonLoading(btn, loading) {
    isLoading = loading;
    btn.disabled = loading;

    if (loading) {
        btn.dataset.originalHtml = btn.innerHTML;
        const text = btn.textContent.trim();
        btn.innerHTML = `<div class="spinner"></div> ${text}...`;
    } else {
        btn.innerHTML = btn.dataset.originalHtml || btn.innerHTML;
    }
}

function showLoading(text) {
    const overlay = document.getElementById('loadingOverlay');
    document.getElementById('loadingText').textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || ''}</span> ${escapeHtml(message)}`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function getFileIcon(ext) {
    const icons = {
        '.txt': '📝', '.md': '📋', '.py': '🐍', '.json': '📦',
        '.csv': '📊', '.log': '📜', '.rst': '📄',
    };
    return icons[ext] || '📄';
}
