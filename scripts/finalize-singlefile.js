import { readFile, writeFile } from 'node:fs/promises'

const outputPath = 'dist/index.html'
const html = await readFile(outputPath, 'utf8')
let finalizedHtml = html
  .replace(/<script type="module" crossorigin>/g, '<script>')
  .replace(/<script type="module">/g, '<script>')
  .replace(/ crossorigin(?=[ >])/g, '')

// vite-plugin-singlefile places the application script in <head>. The
// source app mounts to #app in <body>, so running it from <head> races the
// parser and leaves an empty page. Move the inlined app script after #app.
const appScriptMatch = finalizedHtml.match(/<script>([\s\S]*?)<\/script>/i)
if (!appScriptMatch) {
  throw new Error('Single-file build does not contain an inline application script')
}

finalizedHtml = finalizedHtml.replace(appScriptMatch[0], '')
// Use a replacement callback so `$&` and similar sequences inside minified
// JavaScript are kept verbatim instead of being interpreted by String.replace.
finalizedHtml = finalizedHtml.replace('</body>', () => `${appScriptMatch[0]}\n  </body>`)

if (/<script[^>]+src=/.test(finalizedHtml)) {
  throw new Error('Single-file build still contains an external script reference')
}

await writeFile(outputPath, finalizedHtml)
console.log(`Finalized ${outputPath} for static anonymous hosting`)
