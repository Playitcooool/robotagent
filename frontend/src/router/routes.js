export const routes = [
  { path: '/', redirect: '/chat' },
  {
    path: '/chat/:sessionId?',
    name: 'chat',
    component: () => import('../views/WorkbenchView.vue')
  },
  {
    path: '/about',
    name: 'about',
    component: () => import('../components/AboutView.vue')
  }
]
