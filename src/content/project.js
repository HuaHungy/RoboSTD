const simulationDemoGroups = [
  { prefix: 'adjust_bottle', count: 4, title: 'Adjust Bottle' },
  { prefix: 'click_alarmclock', count: 4, title: 'Click Alarmclock' },
  { prefix: 'move_pillbottle_pad', count: 2, title: 'Move Pillbottle Pad' },
  { prefix: 'open_laptop', count: 2, title: 'Open Laptop' },
  { prefix: 'place_can_basket', count: 4, title: 'Place Can Basket' },
  { prefix: 'place_container_plate', count: 2, title: 'Place Container Plate' },
  { prefix: 'place_phone_stand', count: 2, title: 'Place Phone Stand' },
  { prefix: 'rotate_qrcode', count: 2, title: 'Rotate QRCode' },
  { prefix: 'stamp_seal', count: 2, title: 'Stamp Seal' },
  { prefix: 'turn_switch', count: 2, title: 'Turn Switch' },
]

const isAnonymousBuild = import.meta.env.MODE === 'anonymous'
const publicAuthors = [
  [
    { name: 'Hongyu Wu', marker: '*1,2' },
    { name: 'Chong Liu', marker: '*1,2' },
    { name: 'Shaoxuan Xie', marker: '1' },
    { name: 'Shaqi Luo', marker: '1' },
    { name: 'Hanyu Feng', marker: '2' },
  ],
  [
    { name: 'Shihan Wu', marker: '†1' },
    { name: 'Guocai Yao', marker: '†1' },
  ],
]
const anonymousAuthors = [[{ name: 'Anonymous Authors', marker: '' }]]

const simulationDemos = simulationDemoGroups.flatMap(({ prefix, count, title }) =>
  Array.from({ length: count }, (_, index) => {
    const number = index + 1
    return {
      title,
      subtitle: 'Full rollout',
      src: `./media/demos/simulation-demo/${prefix}-${number}.gif`,
    }
  }),
)

const realWorldDemoGroups = [
  { prefix: 'bowl-placement', count: 3, title: 'Bowl Placement', subtitle: 'Rigid-object placement' },
  { prefix: 'cup-collection', count: 4, segments: 3, title: 'Cup Collection', subtitle: 'Bimanual spatial coordination' },
  { prefix: 'flower-arrangement', count: 3, title: 'Flower Arrangement', subtitle: 'Precision placement' },
  { prefix: 'sandwich-making', count: 3, title: 'Sandwich Making', subtitle: 'Long-horizon coordination' },
  { prefix: 'towel-storage', count: 3, title: 'Towel Storage', subtitle: 'Deformable-object manipulation' },
]

const realWorldDemos = realWorldDemoGroups.flatMap(({ prefix, count, segments = 1, title, subtitle }) =>
  Array.from({ length: count }, (_, index) =>
    Array.from({ length: segments }, (_, segment) => ({
      title,
      subtitle,
      src: `./media/demos/realworld-demo/${prefix}-${index + 1}${segments > 1 ? `-${segment + 1}` : ''}.gif`,
    })),
  ).flat(),
)

export const project = {
  meta: {
    shortTitle: 'RoboSTD',
    venue: 'ICRA Submission',
    year: '2027',
    status: isAnonymousBuild ? 'Anonymous review version' : 'Public version',
  },
  hero: {
    titlePrefix: 'Robo',
    titleAccent: 'STD',
    subtitle: {
      lead: 'Zero-Shot ',
      single: 'Single',
      connector: '-to-',
      dual: 'Dual',
      tail: ' Transfer via Sagittal Mirroring for Bimanual Learning',
    },
    tagline:
      'Transform existing single-arm demonstrations into executable, coordinated bimanual supervision — without collecting additional task-specific bimanual demonstrations.',
  },
  metrics: [
    { value: '0', label: 'Additional Bimanual Demos', note: 'zero-shot data construction' },
    { value: '66.5%', label: 'RoboTwin Success', note: 'only 1.1 pp below Oracle' },
    { value: '2.11×', label: 'Cup Collection Gain', note: 'over the single-arm baseline' },
    { value: '13.0 pp', label: 'Final Arm-Side Gap', note: 'reduced from 28.3 pp' },
  ],
  projectVideo: {
    src: './media/videos/RoboSTD.mp4',
    poster: './media/videos/RoboSTD_poster.jpg',
    title: 'RoboSTD project video',
  },
  authors: isAnonymousBuild ? anonymousAuthors : publicAuthors,
  affiliation: isAnonymousBuild
    ? 'Submitted to the IEEE International Conference on Robotics and Automation'
    : '1 Beijing Academy of Artificial Intelligence · 2 Beijing University of Posts and Telecommunications',
  authorNote: isAnonymousBuild ? 'Author information withheld for anonymous review.' : '* Equal contribution · † Corresponding authors',
  links: [
    {
      label: 'Paper',
      note: 'PDF',
      href: './media/paper/RoboSTD.pdf',
      icon: 'paper',
      download: true,
    },
    { label: 'arXiv', note: 'Coming soon', href: '#', icon: 'archive', disabled: true },
    {
      label: 'Code',
      note: isAnonymousBuild ? 'Anonymous repository' : 'GitHub repository',
      href: isAnonymousBuild ? 'https://anonymous.4open.science/r/RoboSTD_code/' : 'https://github.com/HuaHungy/RoboSTD.git',
      icon: 'code',
      external: true,
    },
  ],
  teaser: {
    src: './media/paper/overview.svg',
    alt: 'Overview of RoboSTD for zero-shot Single-to-Dual transfer',
    caption:
      'RoboSTD turns single-arm skills into executable contralateral skills, then coordinates them into pseudo-bimanual supervision.',
  },
  abstract: [
    'Bimanual robot learning is limited by the cost of collecting coordinated dual-arm demonstrations, while abundant single-arm data cannot be directly reused because of geometric and coordination gaps.',
    'RoboSTD addresses this challenge in two stages. First, Sagittal-Plane Mirroring transfers observations, proprioception, and actions into physically executable contralateral skills. Second, LLM-Guided Spatio-Temporal Reconstruction organizes original and mirrored skills through arm assignment, precedence, and conflict constraints.',
    'The resulting pseudo-bimanual supervision supports effective Single-to-Dual transfer without additional task-specific bimanual demonstrations, improving workspace coverage, long-horizon coordination, and arm-side balance.',
  ],
  motivation: {
    title: 'Why Single-to-Dual Transfer?',
    paragraphs: [
      'Single-arm data are abundant, but they provide only unilateral supervision. Our analysis of Open X-Embodiment further reveals substantial left-right spatial imbalance, which can also appear as arm-side preference in pretrained robot policies.',
      'Reusing these data for bimanual learning therefore requires solving two distinct challenges:',
    ],
    challenges: [
      {
        title: 'Physically Valid Transfer',
        question: 'How can a single-arm skill be mapped into an executable skill for the opposite arm?',
      },
      {
        title: 'Structured Bimanual Coordination',
        question: 'How should original and transferred skills be assigned and ordered across space and time?',
      },
    ],
    closing:
      'RoboSTD addresses these two challenges with sagittal-plane mirroring and LLM-guided spatio-temporal reconstruction.',
    figure: {
      src: './media/paper/motivation.svg',
      alt: 'Motivation for RoboSTD and illustration of spatial bias mitigation',
      caption: 'From biased single-arm demonstrations to balanced bimanual supervision.',
    },
  },
  method: {
    title: 'Method',
    lead: 'Two stages: transfer single-arm skills across arms, then coordinate them across space and time.',
    stages: [
      {
        title: 'Stage 1 — Sagittal-Plane Mirroring.',
        body: 'RoboSTD jointly transforms observations, proprioceptive states, and actions to create a physically executable skill for the contralateral arm. The mapping is derived from robot geometry and mounting configuration rather than a simple image flip.',
        output: 'Executable contralateral pseudo-demonstrations',
      },
      {
        title: 'Stage 2 — LLM-Guided Spatio-Temporal Reconstruction.',
        body: 'GPT-4.1 infers arm assignment, precedence, and conflict constraints from the task context. These constraints guide the rearrangement of original and mirrored skills on a shared timeline and generate stage-level language supervision.',
        output: 'Coordinated pseudo-bimanual supervision',
      },
    ],
    points: [
      { label: 'Stage 1', title: 'Observation–State–Action Consistency' },
      { label: 'Stage 2', title: 'Arm Assignment · Precedence · Conflict' },
      { label: 'Final Output', title: 'Coordinated Pseudo-Bimanual Supervision' },
    ],
    figure: {
      src: './media/paper/method.svg',
      alt: 'RoboSTD method pipeline',
      caption:
        'RoboSTD mirrors single-arm trajectories, predicts coordination constraints, and constructs synchronized pseudo-bimanual supervision.',
    },
  },
  experiments: [
    {
      label: 'Q1',
      title: 'Q1 — Is RoboSTD Supervision Effective?',
      paragraphs: [
        'Across 10 RoboTwin 2.0 tasks, RoboSTD achieves 66.5% average success using only 50 source single-arm demonstrations per task, closely matching the 67.6% Oracle trained with 100 native bimanual demonstrations.',
      ],
      highlights: [
        { label: '', value: '39.1% → 66.5%' },
        { label: '', value: 'Only 1.1 pp below Oracle' },
      ],
      figure: {
        src: './media/paper/simulation.svg',
        alt: 'RoboTwin task suite and task-wise simulation results',
        caption: 'Task-wise success under Single-Arm, RoboSTD, and Oracle supervision.',
      },
    },
    {
      label: 'Q2',
      title: 'Q2 — From Skill Transfer to Bimanual Coordination',
      paragraphs: [
        'On low-coordination tasks, RoboSTD-generated supervision transfers single-arm skills effectively to real-world bimanual manipulation.',
        'On high-coordination tasks, RoboSTD w/o LLM remains below the full method, showing that geometric transfer alone is insufficient. Arm assignment, precedence, conflict constraints, and stage-level language supervision become critical when both arms must share workspace or execute long-horizon tasks.',
      ],
      highlights: [
        { label: 'Spatial Coordination', value: '2.00 → 2.76 → 4.22 cups' },
        { label: 'Temporal Coordination', value: '55% → 63% → 74% completion' },
      ],
      figure: {
        src: './media/paper/real-world.svg',
        alt: 'Real-world evaluation on low- and high-coordination tasks',
        caption: 'Real-world evaluation on the dual-arm AgileX ALOHA platform.',
      },
    },
    {
      label: 'Q3',
      title: 'Q3 — Can RoboSTD Mitigate Arm-Side Spatial Preference?',
      paragraphs: [
        'We evaluate the three RoboTwin tasks with the largest right-to-left gaps identified in our preliminary analysis. RoboSTD produces broader left-right trajectory coverage and substantially reduces the arm-side performance gap.',
      ],
      highlights: [
        { label: 'Density Gap', value: '45.8% → 25.6%' },
        { label: 'Bias Gap', value: '28.3 pp → 13.0 pp' },
      ],
      note: 'The weaker left-arm performance improves while right-arm performance remains largely preserved.',
      figure: {
        src: './media/paper/bias-mitigation.svg',
        alt: 'Trajectory density and arm-side performance before and after RoboSTD',
        caption: 'RoboSTD improves left-arm performance while preserving the stronger right arm.',
      },
    },
  ],
  videoGroups: [
    {
      id: 'simulation',
      title: 'Simulation Demonstrations',
      subtitle: 'RoboTwin 2.0 task rollouts',
      rows: 2,
      tasks: simulationDemos,
    },
    {
      id: 'real-world',
      title: 'Real-World Demonstrations',
      subtitle: 'AgileX ALOHA task rollouts',
      rows: 2,
      tasks: realWorldDemos,
    },
  ],
  experimentsLead: 'Does RoboSTD provide effective, coordinated, and balanced bimanual supervision?',
  conclusion: {
    lead: 'From executable transfer to coordinated pseudo-bimanual supervision.',
    paragraphs: [
      'RoboSTD turns existing single-arm demonstrations into coordinated pseudo-bimanual supervision through two complementary stages: sagittal-plane mirroring for physically valid contralateral transfer and LLM-guided spatio-temporal reconstruction for structured bimanual coordination.',
      'Across simulation and real-world tasks, RoboSTD approaches native bimanual supervision, improves spatial and long-horizon coordination, and mitigates arm-side spatial preference — without additional task-specific bimanual demonstrations.',
    ],
    takeaways: [
      { label: 'Transfer', text: 'Physically executable contralateral skills' },
      { label: 'Coordinate', text: 'Arm assignment, precedence, and conflict constraints' },
      { label: 'Scale', text: 'No additional task-specific bimanual demonstrations' },
    ],
  },
  bibtex: isAnonymousBuild
    ? `@inproceedings{anonymous2027robostd,
  title     = {RoboSTD: Zero-Shot Single-to-Dual Transfer via
               Sagittal Mirroring for Bimanual Learning},
  author    = {Anonymous Authors},
  booktitle = {IEEE International Conference on Robotics and Automation},
  year      = {2027}
}`
    : `@inproceedings{wu2027robostd,
  title     = {RoboSTD: Zero-Shot Single-to-Dual Transfer via Sagittal Mirroring for Bimanual Learning},
  author    = {Wu, Hongyu and Liu, Chong and Xie, Shaoxuan and Luo, Shaqi and Feng, Hanyu and Wu, Shihan and Yao, Guocai},
  booktitle = {2027 IEEE International Conference on Robotics and Automation},
  year      = {2027}
}`,
}
