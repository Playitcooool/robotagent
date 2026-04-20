<template>
  <section class="landing-hero">
    <div class="eyebrow">{{ lang === 'zh' ? 'Robot Operations Console' : 'Robot Operations Console' }}</div>
    <h1 class="headline">
      {{ lang === 'zh' ? '把机器人任务拆解、执行、验证放进同一个工作台。' : 'Run planning, simulation, and verification from one robot workbench.' }}
    </h1>
    <p class="lead">
      {{ lang === 'zh'
        ? '输入一个任务目标，RobotAgent 会在同一界面里组织规划、调用分析与仿真代理，并持续回传执行证据。'
        : 'Describe the mission goal and RobotAgent will coordinate planning, analysis, simulation, and evidence in one workspace.' }}
    </p>

    <div class="hero-grid">
      <button
        v-for="item in suggestions"
        :key="item.title"
        type="button"
        class="hero-card"
        @click="$emit('prompt', item.prompt)"
      >
        <span class="hero-card-tag">{{ item.tag }}</span>
        <strong>{{ item.title }}</strong>
        <p>{{ item.description }}</p>
      </button>
    </div>
  </section>
</template>

<script>
export default {
  name: 'LandingHero',
  props: {
    lang: { type: String, default: 'zh' }
  },
  emits: ['prompt'],
  computed: {
    suggestions () {
      if (this.lang === 'en') {
        return [
          {
            tag: 'Simulation',
            title: 'Validate a grasping plan',
            description: 'Generate a short manipulation plan and simulate whether the grasp succeeds.',
            prompt: 'Please design a simple grasping plan for a cube on the table and verify it in simulation.'
          },
          {
            tag: 'Analysis',
            title: 'Analyze trajectory data',
            description: 'Inspect trajectory stability, deviations, and likely failure causes.',
            prompt: 'Analyze the robot trajectory data and explain the main instability indicators.'
          },
          {
            tag: 'Mission',
            title: 'Break down a multi-step task',
            description: 'Turn a natural-language robot request into an executable mission checklist.',
            prompt: 'Break down a pick-and-place mission for a tabletop robot arm and list the execution steps.'
          }
        ]
      }

      return [
        {
          tag: '仿真',
          title: '验证抓取任务',
          description: '生成一个简短抓取计划，并在仿真里验证是否成功。',
          prompt: '请为桌面上的立方体设计一个简单抓取方案，并在仿真中验证结果。'
        },
        {
          tag: '分析',
          title: '分析轨迹稳定性',
          description: '检查轨迹波动、偏差来源和潜在失败原因。',
          prompt: '请分析这段机器人轨迹数据，并解释最主要的不稳定指标。'
        },
        {
          tag: '任务',
          title: '拆解复杂任务',
          description: '把自然语言需求拆成可执行的机器人任务清单。',
          prompt: '请把一个桌面机械臂的 pick-and-place 任务拆解成可执行步骤。'
        }
      ]
    }
  }
}
</script>

<style scoped>
.landing-hero {
  width: min(940px, 100%);
  margin: 0 auto 22px;
  text-align: left;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  color: #8fb7ff;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.16em;
}

.headline {
  max-width: 18ch;
  margin: 0;
  font-size: clamp(2.4rem, 4vw, 4.3rem);
  line-height: 1.02;
  letter-spacing: -0.05em;
  text-wrap: balance;
}

.lead {
  max-width: 72ch;
  margin: 18px 0 0;
  color: rgba(230, 237, 243, 0.78);
  font-size: 15px;
  line-height: 1.8;
}

.hero-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 28px;
}

.hero-card {
  display: grid;
  gap: 10px;
  text-align: left;
  border: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.01)),
    rgba(9, 14, 24, 0.74);
  padding: 18px;
  color: var(--text);
  cursor: pointer;
  transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}

.hero-card:hover {
  transform: translateY(-3px);
  border-color: rgba(95, 156, 255, 0.34);
  box-shadow: 0 18px 32px rgba(0, 0, 0, 0.24);
}

.hero-card-tag {
  display: inline-flex;
  width: fit-content;
  padding: 4px 9px;
  border-radius: 999px;
  background: rgba(67, 143, 255, 0.14);
  color: #bcd4ff;
  font-size: 11px;
  font-weight: 700;
}

.hero-card strong {
  font-size: 16px;
}

.hero-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
  font-size: 13px;
}

@media (max-width: 960px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
}
</style>
