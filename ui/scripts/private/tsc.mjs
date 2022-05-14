import { exec } from 'child_process'
import chalk from 'chalk'
const { yellowBright, white } = chalk

function compile(directory) {
  return new Promise((resolve, reject) => {
    const process = exec('tsc', {
      cwd: directory,
    });

    process.stdout.on('data', data => {
        console.log(yellowBright(`[tsc] `) + white(data.toString()))
    });

    process.on('exit', exitCode => {
      if (exitCode > 0) {
        reject(exitCode)
      } else {
        resolve()
      }
    });
  });
}

export default compile
