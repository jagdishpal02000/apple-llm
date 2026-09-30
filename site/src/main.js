import './style.css'
import { highlight } from './highlight.js'

const $ = (sel, root = document) => root.querySelector(sel)
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)]
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key)
    localStorage.setItem(key, value)
  } catch {
    return null
  }
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    const area = Object.assign(document.createElement('textarea'), { value: text })
    document.body.append(area)
    area.select()
    document.execCommand('copy')
    area.remove()
  }
  button.dataset.done = ''
  button.textContent = 'Copied'
  setTimeout(() => {
    delete button.dataset.done
    button.textContent = 'Copy'
  }, 1400)
}

// ── theme: auto → light → dark ────────────────────────────
{
  const button = $('[data-theme-toggle]')
  const modes = ['auto', 'light', 'dark']
  const apply = (mode) => {
    if (mode === 'auto') delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = mode
    button.dataset.mode = mode
    button.setAttribute('aria-label', `Colour theme: ${mode}. Click to change.`)
  }
  apply(modes.includes(store('apple-llm-theme')) ? store('apple-llm-theme') : 'auto')
  button.addEventListener('click', () => {
    const next = modes[(modes.indexOf(button.dataset.mode) + 1) % modes.length]
    apply(next)
    store('apple-llm-theme', next)
  })
}

// ── code blocks: highlight + copy ─────────────────────────
for (const code of $$('pre > code[data-lang]')) {
  const source = code.textContent
  code.innerHTML = highlight(source, code.dataset.lang)
  const pre = code.parentElement
  const wrap = document.createElement('div')
  wrap.className = 'pre-wrap'
  pre.replaceWith(wrap)
  wrap.append(pre)
  const button = Object.assign(document.createElement('button'), {
    className: 'copy',
    type: 'button',
    textContent: 'Copy',
  })
  button.addEventListener('click', () => copyText(source, button))
  wrap.append(button)
}

// ── install switcher ──────────────────────────────────────
{
  const box = $('[data-install]')
  const cmd = $('[data-install-cmd]', box)
  const tabs = $$('[role="tab"]', box)
  for (const tab of tabs) {
    tab.addEventListener('click', () => {
      for (const t of tabs) t.setAttribute('aria-selected', String(t === tab))
      cmd.textContent = tab.dataset.cmd
    })
  }
  const button = $('[data-copy-install]', box)
  button.addEventListener('click', () => copyText(cmd.textContent, button))
}

// ── code tabs (arrow keys move between them) ──────────────
for (const box of $$('[data-tabs]')) {
  const tabs = $$('[role="tab"]', box)
  const select = (tab, focus) => {
    for (const t of tabs) {
      const on = t === tab
      t.setAttribute('aria-selected', String(on))
      t.tabIndex = on ? 0 : -1
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on
    }
    if (focus) tab.focus()
  }
  tabs.forEach((tab, i) => {
    tab.tabIndex = i === 0 ? 0 : -1
    tab.addEventListener('click', () => select(tab))
    tab.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key]
      if (!step) return
      e.preventDefault()
      select(tabs[(i + step + tabs.length) % tabs.length], true)
    })
  })
}

// ── terminal: replay the session as if typed ──────────────
{
  const pre = $('[data-term]')
  if (pre && !reduceMotion) {
    const lines = pre.innerHTML.split('\n')
    pre.style.minHeight = `${pre.offsetHeight}px`
    pre.textContent = ''
    pre.dataset.ready = ''
    const cursor = Object.assign(document.createElement('span'), { className: 't-cursor' })
    pre.append(cursor)
    const decode = (html) => Object.assign(document.createElement('textarea'), { innerHTML: html }).value
    const PROMPT = '<span class="t-p">$</span> '

    const play = async () => {
      for (const [i, line] of lines.entries()) {
        const out = document.createElement('span')
        pre.insertBefore(out, cursor)
        if (line.startsWith(PROMPT)) {
          await sleep(i === 0 ? 300 : 650)
          out.innerHTML = PROMPT
          for (const ch of decode(line.slice(PROMPT.length))) {
            out.append(ch)
            await sleep(ch === ' ' ? 45 : 16 + Math.random() * 26)
          }
          out.append('\n')
          await sleep(420)
        } else {
          out.innerHTML = `${line}\n`
          await sleep(55)
        }
      }
      const tail = document.createElement('span')
      tail.innerHTML = PROMPT
      pre.insertBefore(tail, cursor)
    }

    new IntersectionObserver((entries, observer) => {
      if (!entries[0].isIntersecting) return
      observer.disconnect()
      play()
    }, { threshold: 0.15 }).observe(pre)
  }
}

// ── live star count; everything reads fine without it ─────
fetch('https://api.github.com/repos/jagdishpal02000/apple-llm')
  .then((r) => (r.ok ? r.json() : null))
  .then((repo) => {
    if (typeof repo?.stargazers_count !== 'number') return
    for (const el of $$('[data-stars]')) el.textContent = `★ ${repo.stargazers_count}`
  })
  .catch(() => {})
