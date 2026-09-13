import { readFile, writeFile } from 'node:fs/promises'

const outputPath = 'dist/index.html'
const html = await readFile(outputPath, 'utf8')
const finalizedHtml = html
  .replace(/<script type="module" crossorigin>/g, '<script>')
  .replace(/<script type="module">/g, '<script>')
  .replace(/ crossorigin(?=[ >])/g, '')

if (/<script[^>]+src=/.test(finalizedHtml)) {
  throw new Error('Single-file build still contains an external script reference')
}

await writeFile(outputPath, finalizedHtml)
console.log(`Finalized ${outputPath} for static anonymous hosting`)
