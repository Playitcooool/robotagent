import { ref } from 'vue'

const LANG_KEY = 'robotagent_lang'

const translations = {
  zh: {
    about: '项目介绍',
    chat: '对话',
    logout: '退出登录',
    currentUser: '当前用户',
    switchTheme: '切换主题',
    switchLang: '切换语言',
    welcome: '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。',
    // Sidebar
    history: '对话历史',
    sessions: '个会话',
    emptySession: '空会话',
    newChat: '新对话',
    rename: '重命名',
    export: '导出',
    share: '分享',
    delete: '删除会话',
    exportSuccess: '会话已导出',
    shareSuccess: '链接已复制到剪贴板',
    renameSuccess: '会话已重命名',
    confirmDelete: (title) => `确认删除会话「${title}」？`,
  },
  en: {
    about: 'About',
    chat: 'Chat',
    logout: 'Logout',
    currentUser: 'Current User',
    switchTheme: 'Switch Theme',
    switchLang: 'Switch Language',
    welcome: 'Welcome to RobotAgent! Select a session on the left or start a new conversation.',
    // Sidebar
    history: 'History',
    sessions: 'sessions',
    emptySession: 'Empty session',
    newChat: 'New Chat',
    rename: 'Rename',
    export: 'Export',
    share: 'Share',
    delete: 'Delete',
    exportSuccess: 'Session exported',
    shareSuccess: 'Link copied to clipboard',
    renameSuccess: 'Session renamed',
    confirmDelete: (title) => `Delete session "${title}"?`,
  }
}

export function useI18n () {
  const lang = ref(localStorage.getItem(LANG_KEY) || 'zh')

  function t (key, ...args) {
    const val = translations[lang.value]?.[key] || translations['zh'][key] || key
    if (typeof val === 'function') return val(...args)
    return val
  }

  function toggleLang () {
    lang.value = lang.value === 'zh' ? 'en' : 'zh'
    localStorage.setItem(LANG_KEY, lang.value)
  }

  return { lang, t, toggleLang, translations }
}
