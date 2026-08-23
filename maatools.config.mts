import type { FullConfig } from '@nekosu/maa-tools'

const config: FullConfig = {
  cwd: import.meta.dirname,
  interfacePath: 'assets/interface.json',
  check: {
    override: {
      'dynamic-image': 'ignore',
    },
  },
}

export default config
