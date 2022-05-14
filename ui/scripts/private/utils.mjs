import path from 'path'
import { fileURLToPath } from 'url'

export const getDirname = fileURL => {
  const __filename = fileURLToPath(fileURL)
  const __dirname = path.dirname(__filename)
  return __dirname
}
