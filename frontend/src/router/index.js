import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../components/ChatView.vue'
import AboutView from '../components/AboutView.vue'

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat/:sessionId?', name: 'chat', component: ChatView },
  { path: '/about', name: 'about', component: AboutView }
]

export const router = createRouter({
  history: createWebHistory(),
  routes
})
