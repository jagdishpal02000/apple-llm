// A small highlighter for the three languages on the page. Enough to make
// the samples readable without shipping a full grammar library.

const KEYWORDS = {
  ts: 'import|from|export|const|let|await|async|new|for|of|return|function|if|else|true|false|null|undefined',
  py: 'from|import|with|as|class|def|return|async|await|for|in|if|else|None|True|False',
}

const escape = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

function pattern(lang) {
  const string = String.raw`(?<s>'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|\x60[^\x60]*\x60)`
  const number = String.raw`(?<n>\b\d[\d_.]*\b)`
  if (lang === 'sh') {
    return [
      String.raw`(?<c>(?<=^|\s)#.*)`,
      string,
      String.raw`(?<k>(?<=\s)--?[a-zA-Z][\w-]*)`,
      String.raw`(?<f>(?<=^|&&\s|\|\s)[\w.-]+)`,
      number,
    ].join('|')
  }
  return [
    lang === 'py' ? String.raw`(?<c>#.*)` : String.raw`(?<c>\/\/.*)`,
    string,
    String.raw`(?<k>\b(?:${KEYWORDS[lang]})\b)`,
    String.raw`(?<f>\b[A-Za-z_]\w*(?=\())`,
    number,
  ].join('|')
}

export function highlight(source, lang) {
  if (!KEYWORDS[lang] && lang !== 'sh') return escape(source)
  const re = new RegExp(pattern(lang), 'gm')
  let html = ''
  let last = 0
  for (const match of source.matchAll(re)) {
    const [kind, text] = Object.entries(match.groups).find(([, v]) => v !== undefined)
    html += escape(source.slice(last, match.index))
    html += `<span class="hl-${kind}">${escape(text)}</span>`
    last = match.index + text.length
  }
  return html + escape(source.slice(last))
}
