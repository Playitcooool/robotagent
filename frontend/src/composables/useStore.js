/**
 * Vue 3 Store with selector-based subscriptions.
 *
 * Inspired by Claude Code's state/store.ts
 *
 * Features:
 * - Immutable state updates (Object.is comparison)
 * - Selector-based subscriptions (only re-renders when selected value changes)
 * - Batch-friendly updates via functional setState
 */

import { shallowRef, computed } from 'vue'

/**
 * Create a store instance with selector-based subscriptions.
 *
 * @param {Object} initialState - Initial state object
 * @returns {Object} Store instance with getState, setState, subscribe methods
 *
 * @example
 * const store = createStore({
 *   settings: { theme: 'dark', lang: 'zh' },
 *   auth: { user: null, token: null }
 * })
 *
 * // Direct access
 * store.getState().settings.theme
 *
 * // Immutable update
 * store.setState(prev => ({
 *   ...prev,
 *   settings: { ...prev.settings, theme: 'light' }
 * }))
 *
 * // Subscribe to all changes
 * const unsubscribe = store.subscribe(() => {
 *   console.log('State changed:', store.getState())
 * })
 *
 * // Selector-based subscription
 * const theme = useSelector(store, s => s.settings.theme)
 */
export function createStore(initialState) {
  const state = shallowRef(initialState)
  const listeners = new Set()

  function getState() {
    return state.value
  }

  function setState(updater) {
    const next = typeof updater === 'function' ? updater(state.value) : updater

    // Object.is comparison for early exit (like Claude Code's store.ts)
    if (!Object.is(next, state.value)) {
      state.value = next
      // Notify all listeners
      listeners.forEach(listener => listener())
    }
  }

  function subscribe(listener) {
    listeners.add(listener)
    // Return unsubscribe function
    return () => listeners.delete(listener)
  }

  return {
    getState,
    setState,
    subscribe
  }
}

/**
 * Vue composable for selector-based state subscription.
 *
 * Uses computed internally to only trigger re-renders when
 * the selected value actually changes (Object.is comparison).
 *
 * @param {Object} store - Store instance from createStore
 * @param {Function} selector - Function to extract value from state
 * @returns {ComputedRef} Computed value that updates reactively
 *
 * @example
 * const store = createStore({ count: 0 })
 * const count = useSelector(store, s => s.count)
 *
 * // In template: {{ count.value }}
 * // Or unwrapped: {{ count }}
 */
export function useSelector(store, selector) {
  return computed(() => selector(store.getState()))
}

/**
 * Vue composable for accessing store state directly.
 * Reacts to any state change (not selector-specific).
 *
 * @param {Object} store - Store instance from createStore
 * @returns {Ref} Shallow ref to the entire state
 *
 * @example
 * const state = useStoreState(store)
 * // Access: state.value.settings.theme
 */
export function useStoreState(store) {
  return shallowRef(store.getState())
}

/**
 * Middleware pattern for state change side effects.
 * Inspired by Claude Code's onChangeAppState.ts
 *
 * @param {Object} store - Store instance
 * @param {Object} handlers - Object mapping state paths to handler functions
 * @returns {Function} Unsubscribe function
 *
 * @example
 * const unsubscribe = createStateMiddleware(store, {
 *   'settings.theme': (prev, next) => {
 *     console.log('Theme changed:', prev, '->', next)
 *     localStorage.setItem('theme', next)
 *   },
 *   'auth.token': (prev, next) => {
 *     if (next) sessionStorage.setItem('token', next)
 *     else sessionStorage.removeItem('token')
 *   }
 * })
 */
export function createStateMiddleware(store, handlers) {
  let prevState = store.getState()

  return store.subscribe(() => {
    const nextState = store.getState()

    for (const [path, handler] of Object.entries(handlers)) {
      const prevValue = getNestedValue(prevState, path)
      const nextValue = getNestedValue(nextState, path)

      if (!Object.is(prevValue, nextValue)) {
        handler(prevValue, nextValue)
      }
    }

    prevState = nextState
  })
}

/**
 * Get nested value from object by dot-notation path.
 * @param {Object} obj - Source object
 * @param {string} path - Dot-notation path (e.g., 'settings.theme')
 * @returns {*} Value at path, or undefined if not found
 */
function getNestedValue(obj, path) {
  return path.split('.').reduce((current, key) => {
    return current && current[key] !== undefined ? current[key] : undefined
  }, obj)
}

// Export a default store factory for convenience
export default { createStore, useSelector, useStoreState, createStateMiddleware }
