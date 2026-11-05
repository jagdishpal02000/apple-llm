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

// ── terminal: replay once, then interactive demo ────────────
{
  const pre = $('[data-term]')
  if (pre) {
    const figure = pre.closest('figure')
    const PROMPT = '<span class="t-p">$</span> '
    const PROMPT_TEXT = '$ '
    const originalHTML = pre.innerHTML
    const lines = originalHTML.split('\n')
    const decode = (html) => Object.assign(document.createElement('textarea'), { innerHTML: html }).value
    let skip = false
    let live = false
    if (figure) {
      figure.style.cursor = 'text'
      figure.addEventListener('click', (e) => {
        skip = true
        const ed = pre.querySelector('.t-row:last-child .t-edit, .t-row .t-edit:last-of-type')
        if (live && ed && e.target !== ed) ed.focus({ preventScroll: true })
      })
    }

    pre.style.minHeight = `${pre.offsetHeight}px`
    pre.textContent = ''
    const cursor = Object.assign(document.createElement('span'), { className: 't-cursor' })
    pre.append(cursor)
    pre.dataset.ready = ''

    const PROBE_OUT = `<span class="t-h">on-device</span>\n  variant     AFM 3 Core Advanced\n  context     8192 tokens\n  supports    guidedGeneration, toolCalling, vision\n  use cases   general, contentTagging\n\n<span class="t-h">cloud (private cloud compute)</span>\n  quota       belowLimit\n\n`
    const RUN_OUT = `<span class="t-o">{"tags": ["sourdough", "rye starter", "cast iron", "dutch oven", "recipe"]}</span>\n\n`
    const COUNT_OUT = `<span class="t-o">64 tokens of 8192 (1%)</span>\n`
    const SPEAK_OUT = `<span class="t-o">(said out loud, in your System voice. Your Mac would have; this page is silent.)</span>\n`
    const VOICES_OUT = `Your System voice is Samantha (en-US). To change it, search System Settings for "System voice".\n\n  Daniel    en-GB    male\n  Karen     en-AU    female\n  Rishi     en-IN    male\n  Samantha  en-US    female  your System voice\n`
    const TRANSCRIBE_OUT = `<span class="t-o">The harbor was quiet when the morning ferry arrived, and most of the shops along the water had not yet opened.</span>\n`
    const HELP_OUT = `Try one of these + Enter:\n  apple-llm probe\n  apple-llm run --use-case contentTagging --schema tags.json "A recipe for sourdough…"\n  apple-llm count --system "You are terse." "The quick brown fox…"\n  apple-llm speak "Hello there"\n  apple-llm voices\n  apple-llm transcribe meeting.mp4\n  apple-llm chat   (basic chat demo, 'exit' quits)\n  clear\n`
    const CHAT_PROMPT = '<span class="t-you">you:</span> '
    const CHAT_PROMPT_TEXT = 'you: '
    const CHAT_INTRO = `Interactive chat (demo, canned replies). 'exit' quits.\n`
    let inChat = false

    const chatReply = (raw) => {
      const msg = raw.trim().toLowerCase()
      if (!msg) return ''
      if (/^(exit|quit|\/exit|bye|goodbye)$/.test(msg)) return '__EXIT__'
      if (/^(clear|cls)$/.test(msg)) return '__CLEAR__'
      if (/\b(hi|hello|hey|yo|namaste)\b/.test(msg))
        return `Hello! I'm the on-device model, running free on this Mac - no API key, no cloud.\nAsk me to tag or rewrite something.\n`
      if (/(who|what) are you|your name|which model/.test(msg))
        return `I'm AFM 3 Core Advanced, Apple's ~20B sparse on-device model (1-4B active).\nFast and private; honest about limits - I don't write code.\n`
      if (/\b(tag|classify|categor)\b/.test(msg))
        return `Sure - on device I'd stream tags back as guaranteed JSON, e.g.\n{"tags": ["demo", "on-device", "free"]}\n`
      if (/\b(rewrite|proofread|summar)/.test(msg))
        return `Paste it after the real thing with: apple-llm rewrite "gonna grab a bite"\nOn device, nothing leaves your Mac.\n`
      if (/\b(cloud|bigger|reason|code)\b/.test(msg))
        return `For that I'd use the cloud tier (Siri AI's server model, 32k context):\napple-llm run --tier cloud "Solve this step by step: ..."\n`
      if (/\b(thank|thanks|great|cool|nice)\b/.test(msg))
        return `Anytime. Star the repo if the demo saved you an API bill.\n`
      const fallbacks = [
        `On your Mac I'd answer that in ~1s, privately on device.\nThis demo is canned - try 'exit', then apple-llm probe.\n`,
        `Good question for the device tier: short, concrete, no long reasoning needed.\nThe real chat streams it live. Try another, or 'exit'.\n`,
      ]
      chatReply.n = ((chatReply.n || 0) + 1) % fallbacks.length
      return fallbacks[chatReply.n]
    }

    const isChatCmd = (raw) => {
      const cmd = raw.trim().replace(/\s+/g, ' ')
      return cmd === 'chat' || cmd === 'apple-llm chat' || /^apple-llm chat(\s|$)/.test(cmd)
    }

    const demoOutput = (raw) => {
      const cmd = raw.trim().replace(/\s+/g, ' ')
      if (!cmd) return ''
      if (/^(help|apple-llm help|--help)$/.test(cmd)) return HELP_OUT
      if (/^(clear|apple-llm clear)$/.test(cmd)) return '__CLEAR__'
      if (cmd === 'apple-llm probe' || cmd === 'probe' || /(^|\s)probe(\s|$)/.test(` ${cmd} `)) return PROBE_OUT
      if (/^(apple-llm )?speak\b/.test(cmd)) return SPEAK_OUT
      if (/^(apple-llm )?voices\b/.test(cmd)) return VOICES_OUT
      if (/^(apple-llm )?transcribe\b/.test(cmd)) return TRANSCRIBE_OUT
      if (/\bcount\b/.test(cmd)) return COUNT_OUT
      if (/\brun\b/.test(cmd)) return RUN_OUT
      if (/^(npm install|pip install|npx apple-llm|apple-llm setup-cloud)/.test(cmd))
        return `This demo runs in your browser - nothing to install here.\nTry: apple-llm probe\n`
      return `zsh: command not found: ${cmd.split(' ')[0]}\nTry 'help' for demo commands.\n`
    }

    const history = []
    const printHTML = (html, before) => {
      const out = document.createElement('span')
      out.innerHTML = html
      const anchor = before || (cursor.isConnected ? cursor : null)
      pre.insertBefore(out, anchor)
    }
    const echoLine = (raw, row, chat) => {
      const echo = document.createElement('span')
      echo.textContent = `${chat ? CHAT_PROMPT_TEXT : PROMPT_TEXT}${raw}\n`
      echo.innerHTML = echo.innerHTML.replace(chat ? /^you: / : /^\$ /, chat ? CHAT_PROMPT : PROMPT)
      row.replaceWith(echo)
      return echo
    }
    const newPromptRow = (chat) => {
      const row = document.createElement('span')
      row.className = 't-row'
      row.innerHTML = chat ? CHAT_PROMPT : PROMPT
      const input = document.createElement('input')
      input.className = 't-edit'
      input.setAttribute('aria-label', chat ? 'Demo chat. Type a message, exit quits.' : 'Demo terminal. Type help and press Enter.')
      input.setAttribute('autocomplete', 'off')
      input.setAttribute('autocapitalize', 'off')
      input.setAttribute('spellcheck', 'false')
      input.setAttribute('enterkeyhint', chat ? 'send' : 'enter')
      input.dataset.h = ''
      input.dataset.chat = chat ? '1' : ''
      row.append(input)
      return { row, input }
    }

    // Single delegated handler for every prompt row, present and future
    pre.addEventListener('keydown', async (e) => {
      const active = e.target
      if (!active || !active.classList || !active.classList.contains('t-edit')) return
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        if (!history.length) return
        const idx = active.dataset.h === '' ? history.length - 1 : Math.max(0, Number(active.dataset.h) - 1)
        active.dataset.h = String(idx)
        active.value = history[idx]
        return
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        if (active.dataset.h === '') return
        const idx = Number(active.dataset.h) + 1
        if (idx >= history.length) { active.value = ''; active.dataset.h = '' }
        else { active.value = history[idx]; active.dataset.h = String(idx) }
        return
      }
      if (e.key !== 'Enter') return
      e.preventDefault()
      const raw = active.value
      const cur = active.closest('.t-row')
      const chat = inChat
      const echo = echoLine(raw, cur, chat)
      active.disabled = true
      if (raw.trim()) history.push(raw)
      if (chat) {
        const reply = chatReply(raw)
        if (reply === '__EXIT__') {
          inChat = false
          printHTML(`Back to zsh - try 'help'.\n`)
        } else if (reply === '__CLEAR__') {
          pre.textContent = ''
          pre.append(echo)
        } else if (reply) {
          if (!reduceMotion) await sleep(280 + Math.random() * 320)
          printHTML(`<span class="t-o">apple:</span> ${reply}`)
        }
      } else if (isChatCmd(raw)) {
        inChat = true
        if (!reduceMotion) await sleep(250)
        printHTML(CHAT_INTRO)
      } else {
        const out = demoOutput(raw)
        if (out === '__CLEAR__') {
          pre.textContent = ''
          pre.append(echo)
        } else if (out) {
          if (!reduceMotion) await sleep(220 + Math.random() * 280)
          printHTML(out)
        }
      }
      const { row, input } = newPromptRow(inChat)
      if (cursor.isConnected) cursor.remove()
      pre.append(row)
      input.focus({ preventScroll: true })
    })

    const goLive = () => {
      if (live) return
      live = true
      if (cursor.isConnected) cursor.remove()
      const { row } = newPromptRow()
      pre.append(row)
    }

    const play = async () => {
      if (reduceMotion) {
        pre.innerHTML = originalHTML
        goLive()
        return
      }
      for (const [i, line] of lines.entries()) {
        if (skip && i > 0) break
        const out = document.createElement('span')
        pre.insertBefore(out, cursor)
        if (line.startsWith(PROMPT)) {
          if (!skip) await sleep(i === 0 ? 300 : 650)
          out.innerHTML = PROMPT
          for (const ch of decode(line.slice(PROMPT.length))) {
            if (skip) break
            out.append(ch)
            await sleep(ch === ' ' ? 45 : 16 + Math.random() * 26)
          }
          out.append('\n')
          if (!skip) await sleep(420)
        } else {
          out.innerHTML = `${line}\n`
          if (!skip) await sleep(55)
        }
      }
      if (skip) {
        // user clicked mid-replay: show the full transcript instantly
        cursor.remove()
        pre.innerHTML = originalHTML
        pre.append(cursor)
      }
      goLive()
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
