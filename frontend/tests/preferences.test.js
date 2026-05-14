import test from 'node:test'
import assert from 'node:assert/strict'

import {
  createStorageAdapter,
  readPreferences,
  toggleLanguage,
  toggleTheme
} from '../src/lib/preferences.js'

test('readPreferences hydrates persisted language, theme, and font size', () => {
  const storage = createStorageAdapter({
    robotagent_lang: 'en',
    robotagent_theme: 'light',
    robotagent_font_size: '18'
  })

  assert.deepEqual(readPreferences(storage), {
    lang: 'en',
    theme: 'light',
    fontSize: 18
  })
})

test('readPreferences defaults to light theme', () => {
  assert.deepEqual(readPreferences(createStorageAdapter()), {
    lang: 'zh',
    theme: 'light',
    fontSize: 15
  })
})

test('toggle helpers flip current theme and language', () => {
  assert.equal(toggleTheme('dark'), 'light')
  assert.equal(toggleTheme('light'), 'dark')
  assert.equal(toggleLanguage('zh'), 'en')
  assert.equal(toggleLanguage('en'), 'zh')
})
