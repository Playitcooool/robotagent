<template>
  <div class="tool-results">
    <div class="header">工具结果</div>
    <div class="body">
      <div v-if="!result" class="empty">无工具执行信息</div>

      <div v-else>
        <!-- if result contains an `image` or `image_url`, render image -->
        <div v-if="isImage(result)" class="image-wrap">
          <img :src="getImageUrl(result)" alt="tool image" />
        </div>

        <pre v-if="isText(result)">{{ pretty(result) }}</pre>

        <!-- support list of items or structured JSON -->
        <div v-if="isJson(result)">
          <h4>数据</h4>
          <pre>{{ pretty(result) }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ToolResults',
  props: { result: { type: [Object, String, null], default: null } },
  methods: {
    isImage (r) {
      if (!r) return false
      if (typeof r === 'string') return r.match(/https?:.*\.(png|jpg|jpeg|gif|svg)/i)
      if (typeof r === 'object') return r.image || r.image_url || r.url
      return false
    },
    getImageUrl (r) {
      if (typeof r === 'string') return r
      return r.image || r.image_url || r.url
    },
    isText (r) { return typeof r === 'string' },
    isJson (r) { return typeof r === 'object' },
    pretty (r) {
      try { return typeof r === 'string' ? r : JSON.stringify(r, null, 2) } catch (e) { return String(r) }
    }
  }
}
</script>
