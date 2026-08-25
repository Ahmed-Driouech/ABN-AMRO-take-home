<script setup>
import { ref, onMounted, computed, nextTick } from 'vue'

const tab = ref('chat')
const question = ref('')
const busy = ref(false)
const turns = ref([])
const datapoints = ref([])
const companies = ref([])
const filter = ref('')
const source = ref(null)
const thread = ref(null)

const history = computed(() =>
  turns.value.filter((t) => t.answer).slice(-3).map((t) => [t.question, t.answer]),
)
const shown = computed(() =>
  filter.value ? datapoints.value.filter((d) => d.company === filter.value) : datapoints.value,
)
const goals = computed(() => shown.value.filter((d) => d.kind !== 'fte'))
const fte = computed(() => shown.value.filter((d) => d.kind === 'fte'))

async function ask() {
  const q = question.value.trim()
  if (!q || busy.value) return
  question.value = ''
  busy.value = true
  const turn = { question: q, answer: '', citations: [], found: null }
  turns.value.push(turn)
  await nextTick()
  thread.value?.scrollTo({ top: thread.value.scrollHeight, behavior: 'smooth' })
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, history: history.value }),
    })
    if (!res.ok) throw new Error(await res.text())
    Object.assign(turn, await res.json())
    if (turn.citations?.length) source.value = turn.citations[0]
  } catch (err) {
    turn.answer = `Request failed: ${err.message}`
    turn.found = false
  } finally {
    busy.value = false
    await nextTick()
    thread.value?.scrollTo({ top: thread.value.scrollHeight, behavior: 'smooth' })
  }
}

const pages = (c) => (c.page === c.page_end ? `p. ${c.page}` : `pp. ${c.page}–${c.page_end}`)

onMounted(async () => {
  const [health, points] = await Promise.all([
    fetch('/api/health').then((r) => r.json()),
    fetch('/api/datapoints').then((r) => r.json()),
  ])
  companies.value = health.companies
  datapoints.value = points
})
</script>

<template>
  <div class="shell">
    <!-- Persistent left navigation for the application's pillars. -->
    <aside class="nav">
      <div class="brand">
        <span class="mark"></span>
        <div>
          <h1>Annual Report<br />Intelligence</h1>
          <p class="label-caps sub">Retrieval-Augmented Analysis</p>
        </div>
      </div>

      <nav>
        <button :class="{ on: tab === 'chat' }" @click="tab = 'chat'">Analysis</button>
        <button :class="{ on: tab === 'data' }" @click="tab = 'data'">Extracted Data</button>
      </nav>

      <div class="corpus">
        <p class="label-caps muted">Indexed Corpus</p>
        <ul>
          <li v-for="c in companies" :key="c">
            <span class="dot"></span>{{ c }} <span class="mono year">2024</span>
          </li>
        </ul>
      </div>
    </aside>

    <!-- Main content -->
    <main>
      <template v-if="tab === 'chat'">
        <header class="bar">
          <h2>Analysis</h2>
          <p class="muted">Answers are grounded in the indexed reports and cite their source page.</p>
        </header>

        <div class="thread" ref="thread">
          <div v-if="!turns.length" class="empty">
            <p class="label-caps muted">Try asking</p>
            <button
              v-for="s in [
                'How much did Shell spend on climate change adaptation in 2024?',
                'What were ASML\'s total personnel expenses in 2024?',
                'What sustainability targets has ABN AMRO set for 2030?',
              ]"
              :key="s"
              class="suggestion"
              @click="question = s; ask()"
            >
              {{ s }}
            </button>
          </div>

          <article v-for="(t, i) in turns" :key="i" class="turn">
            <p class="q">{{ t.question }}</p>

            <p v-if="!t.answer" class="thinking">Searching the reports…</p>

            <!-- AI response: sulfur-yellow accent bar marks generated content -->
            <div v-else class="response" :class="{ notfound: t.found === false }">
              <p class="a">{{ t.answer }}</p>
              <p v-if="t.found === false" class="notfound-badge label-caps">
                Not found in the indexed reports
              </p>
              <div v-if="t.citations?.length" class="cites">
                <span class="label-caps muted">Evidence</span>
                <button
                  v-for="(c, j) in t.citations"
                  :key="j"
                  class="chip"
                  :class="{ active: source === c }"
                  @click="source = c"
                >
                  {{ c.company }} <span class="mono">{{ pages(c) }}</span>
                </button>
              </div>
            </div>
          </article>
        </div>

        <form @submit.prevent="ask">
          <input
            v-model="question"
            :disabled="busy"
            placeholder="Ask about a company's annual report…"
          />
          <button :disabled="busy || !question.trim()">{{ busy ? 'Searching…' : 'Ask' }}</button>
        </form>
      </template>

      <template v-else>
        <header class="bar">
          <h2>Extracted Data</h2>
          <p class="muted">
            Extracted once per report at ingestion and stored, not looked up per question. Each
            row records the page it came from.
          </p>
        </header>

        <div class="scroll">
          <label class="filter label-caps">
            Company
            <select v-model="filter">
              <option value="">All</option>
              <option v-for="c in companies" :key="c" :value="c">{{ c }}</option>
            </select>
          </label>

          <section v-if="fte.length">
            <p class="label-caps muted section-label">Workforce (FTE)</p>
            <div class="metrics">
              <div v-for="(d, i) in fte" :key="i" class="metric">
                <p class="label-caps muted">{{ d.company }}</p>
                <p v-if="d.value" class="mono value">{{ d.value }} <em>{{ d.unit }}</em></p>
                <p v-else class="notfound-badge label-caps">not found</p>
                <p class="mono src">{{ d.page ? `p. ${d.page}` : '—' }}</p>
              </div>
            </div>
          </section>

          <section>
            <p class="label-caps muted section-label">Sustainability Goals</p>
            <table>
              <thead>
                <tr>
                  <th>Company</th><th>Goal</th><th>Target</th>
                  <th class="num">Year</th><th class="num">Source</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(d, i) in goals" :key="i">
                  <td>{{ d.company }}</td>
                  <td>{{ d.label }}</td>
                  <td class="muted">{{ d.value }}</td>
                  <td class="num mono">{{ d.target_year || '—' }}</td>
                  <td class="num mono">{{ d.page ? `p. ${d.page}` : '—' }}</td>
                </tr>
              </tbody>
            </table>
          </section>
        </div>
      </template>
    </main>

    <!-- Collapsible context panel: the source behind a citation. -->
    <aside class="context" :class="{ open: source }">
      <template v-if="source">
        <div class="ctx-head">
          <p class="label-caps muted">Source</p>
          <button class="close" @click="source = null" aria-label="Close">×</button>
        </div>
        <h3>{{ source.company }}</h3>
        <p class="mono ctx-page">{{ pages(source) }} · {{ source.content_type }}</p>
        <p class="ctx-section">{{ source.section || '—' }}</p>
        <p class="muted ctx-note">
          Cited by excerpt number rather than written by the model, so a page reference is always
          one retrieval actually returned.
        </p>
      </template>
      <p v-else class="muted ctx-empty">Select an evidence chip to inspect its source.</p>
    </aside>
  </div>
</template>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr) auto;
  height: 100vh;
  max-width: 1440px;
  margin: 0 auto;
}

/* ---------- left navigation ---------- */
.nav {
  background: var(--surface-container-low);
  border-right: 1px solid var(--outline-variant);
  padding: var(--gutter);
  display: flex;
  flex-direction: column;
  gap: var(--stack-lg);
}
.brand { display: flex; gap: 12px; align-items: flex-start; }
.mark {
  width: 28px; height: 28px; flex: none; border-radius: var(--radius);
  background: var(--primary); position: relative;
}
.mark::after {
  content: ''; position: absolute; inset: 9px 9px auto auto;
  width: 10px; height: 10px; background: var(--tertiary-fixed); border-radius: 1px;
}
.brand h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
.sub { color: var(--on-surface-variant); margin: 6px 0 0; }

nav { display: flex; flex-direction: column; gap: 2px; }
nav button {
  text-align: left; border: 0; background: transparent; color: var(--on-surface-variant);
  padding: 9px 12px; border-radius: var(--radius); font-weight: 500;
}
nav button:hover { background: var(--surface-container); }
nav button.on { background: var(--primary); color: var(--on-primary); }

.corpus { margin-top: auto; }
.corpus ul { list-style: none; padding: 0; margin: 12px 0 0; }
.corpus li {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 0; color: var(--on-surface-variant);
}
.dot { width: 6px; height: 6px; border-radius: 50%; background: var(--secondary); flex: none; }
.year { color: var(--outline); margin-left: auto; }

/* ---------- main ---------- */
main { display: flex; flex-direction: column; min-width: 0; }
.bar {
  padding: var(--gutter) var(--margin, 40px);
  border-bottom: 1px solid var(--outline-variant);
  background: var(--surface-container-lowest);
}
.bar h2 { font-size: 20px; font-weight: 500; margin: 0 0 4px; }
.bar p { margin: 0; font-size: 13px; }
.muted { color: var(--on-surface-variant); }

.thread, .scroll { flex: 1; overflow-y: auto; padding: var(--stack-lg) 40px; }

.empty { display: flex; flex-direction: column; align-items: flex-start; gap: var(--stack-sm); }
.suggestion {
  border: 1px solid var(--outline-variant); background: var(--surface-container-lowest);
  border-radius: var(--radius); padding: 10px 14px; text-align: left; color: var(--on-surface);
}
.suggestion:hover { background: var(--surface-container); border-color: var(--outline); }

.turn { margin-bottom: var(--stack-lg); }
.q { font-weight: 600; margin: 0 0 12px; font-size: 15px; }
.thinking { color: var(--on-surface-variant); font-style: italic; margin: 0; }

/* Sulfur-yellow accent bar distinguishes generated content from the question. */
.response {
  border: 1px solid var(--outline-variant);
  border-left: 3px solid var(--tertiary-fixed-dim);
  border-radius: var(--radius);
  background: var(--surface-container-lowest);
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgb(11 28 48 / 4%);
}
.response.notfound { border-left-color: var(--outline); }
.a { margin: 0; white-space: pre-wrap; line-height: 1.6; }
.notfound-badge {
  display: inline-block; margin: 10px 0 0; padding: 4px 8px; border-radius: var(--radius);
  background: var(--error-container); color: var(--on-error-container);
}
.cites {
  display: flex; flex-wrap: wrap; align-items: center; gap: var(--stack-sm);
  margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--outline-variant);
}
.chip {
  border: 1px solid var(--tertiary-fixed-dim); background: color-mix(in srgb, var(--tertiary-fixed) 22%, white);
  color: var(--on-tertiary-fixed-variant); border-radius: var(--radius);
  padding: 3px 8px; font-size: 12px; font-weight: 500;
}
.chip:hover, .chip.active { background: var(--tertiary-fixed); color: var(--on-tertiary-fixed); }

form { display: flex; gap: var(--stack-sm); padding: var(--stack-md) 40px var(--gutter); border-top: 1px solid var(--outline-variant); }
form input {
  flex: 1; padding: 12px 14px; border: 1px solid var(--outline);
  border-radius: var(--radius-lg); background: var(--surface-container-lowest);
}
form input:focus { outline: 2px solid var(--tertiary-fixed-dim); outline-offset: 1px; border-color: var(--outline); }
form button {
  padding: 12px 22px; border: 0; border-radius: var(--radius-lg);
  background: var(--secondary); color: var(--on-secondary); font-weight: 500;
}
form button:disabled { opacity: 0.45; }

/* ---------- extracted data ---------- */
.filter { display: inline-flex; align-items: center; gap: 8px; color: var(--on-surface-variant); margin-bottom: var(--stack-lg); }
.filter select {
  border: 1px solid var(--outline); border-radius: var(--radius); padding: 6px 8px;
  background: var(--surface-container-lowest); font-size: 13px; letter-spacing: 0; text-transform: none; font-weight: 400;
}
.section-label { margin: 0 0 12px; }
section { margin-bottom: var(--stack-lg); }

.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.metric {
  border: 1px solid var(--outline-variant); border-radius: var(--radius);
  background: var(--surface-container-lowest); padding: 12px;
}
.metric .value { font-size: 18px; margin: 8px 0 4px; }
.metric em { font-style: normal; font-size: 12px; color: var(--on-surface-variant); }
.metric .src { color: var(--outline); font-size: 11px; margin: 0; }

table { width: 100%; border-collapse: collapse; }
thead th {
  background: var(--surface-container); color: var(--on-surface-variant);
  font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--outline-variant);
}
tbody td { padding: 10px 12px; border-bottom: 1px solid var(--outline-variant); vertical-align: top; }
tbody tr:hover { background: var(--surface-container-low); }
.num { text-align: right; }

/* ---------- right context panel ---------- */
.context {
  width: 0; overflow: hidden; border-left: 1px solid transparent;
  transition: width 0.18s ease;
  background: var(--surface-container-low);
}
.context.open { width: 360px; border-left-color: var(--outline-variant); padding: var(--gutter); }
.ctx-head { display: flex; align-items: center; justify-content: space-between; }
.close { border: 0; background: transparent; font-size: 20px; line-height: 1; color: var(--on-surface-variant); }
.context h3 { font-size: 18px; font-weight: 600; margin: 16px 0 4px; }
.ctx-page { color: var(--on-surface-variant); margin: 0 0 16px; }
.ctx-section {
  border-left: 2px solid var(--tertiary-fixed-dim); padding-left: 10px;
  margin: 0 0 16px; color: var(--on-surface); font-size: 13px;
}
.ctx-note, .ctx-empty { font-size: 12px; line-height: 1.5; margin: 0; }
.ctx-empty { padding: var(--gutter); }

/* ---------- mobile ---------- */
@media (max-width: 900px) {
  .shell { grid-template-columns: 1fr; height: auto; }
  .nav { flex-direction: row; align-items: center; gap: var(--stack-md); border-right: 0; border-bottom: 1px solid var(--outline-variant); }
  .nav .corpus { display: none; }
  nav { flex-direction: row; margin-left: auto; }
  .thread, .scroll, .bar, form { padding-left: 16px; padding-right: 16px; }
  .context.open { width: 100%; }
  .scroll { overflow-x: auto; }
}
</style>
