export interface Upr {
  fps: number,  // 钢琴卷帘 frames per second
  pianoroll: string[],
  qpm: number,
  timeSignature: number[],  // [分子, 分母]
  keySignature: number,  // [0, 23]范围的调号序号
  updatedAt: number,
}

export interface KeySignatureOption {
  value: number,
  label: string,
}

export interface instrumentMap {
  [key: string]: string
  // 索引签名, key也可以换成其他属性名，限制索引类型跟属性类型都是字符串

  'bass-electric': string 
  'bassoon': string 
  'cello': string 
  'clarinet': string 
  'contrabass': string,
  'flute': string 
  'french-horn': string 
  'guitar-acoustic': string 
  'guitar-electric': string 
  'guitar-nylon': string 
  'harmonium': string 
  'harp': string 
  'organ': string 
  'piano': string 
  'saxophone': string 
  'trombone': string 
  'trumpet': string 
  'tuba': string 
  'violin': string 
  'xylophone': string,
}
