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
