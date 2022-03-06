### todo
`配置运行环境，不用虚拟环境
`尝试运行各个项目看看效果
`选定最好的为基准进行改进
`seq2seq机器翻译模型 > attention机制 > transformer > bert模型及预训练方式 > bert下游任务微调
研究tasks.py，看训练具体设置，对照着进行数据预处理函数，在preprocess里调用并将结果写入h5py
照着add_transcription_task_to_registry实现一遍就得到数据集
使用Maestro的MIDI合成几个wav进行测试


想法
  结合bytedance跟huggingfaceT5的例子
  学习率使用1e-4 and 3e-4 work
  束搜索
  1.5. 实验：改变输入序列长度
  像bytedance那样利用踏板信息
  使用小数据集时还应进一步减小模型尺寸
  旋律提取 统一音色训练：各种乐器，嗓音
  使用预训练模型
  TPU上训练时要“pad all examples of the dataset to the same length or make use of pad_to_multiple_of to have a small number of predefined bucket sizes to fit all examples in”
  安装apex(linux)，模型会自己使用，使得训练更快
  如何提高transformer的计算速度和内存使用效率？提示：可以参考论文 [[Tay et al., 2020]](https://zh-v2.d2l.ai/chapter_references/zreferences.html#tay-dehghani-bahri-ea-2020)。

具体如何根据已有的模型、数据集进行可行性分析
注意记录数据，展示解决问题的思路、难点
每个小改进都单独做一下实验给出对比数据
实验在四月底前完成



明治十七年の上海アリス（红魔乡3面Boss）
39 佐渡の二ッ岩（神灵庙ExBoss）
38 幻視の夜　～ Ghostly Eyes（永夜抄1面道中）
29 旧地獄街道を行く（地灵殿3面道中）
27 少女さとり　～ 3rd eye（地灵殿4面Boss）
25 妖怪の山　～ Mysterious Mountain（风神录4面Boss）
24 針小棒大の天守閣（辉针城6面道中）
22 恋色マスタースパーク（永夜抄4B面Boss）
21 感情の摩天楼　～ Cosmic Mind（星莲船6面Boss）
緑眼のジェラシー（地灵殿2面Boss）
19 ピュアヒューリーズ　～ 心の在処（绀珠传6面Boss）
17 華狭間のバトルフィールド（深秘录华扇角色曲）
16 神さびた古戦場　～ Suwa Foughten Field（风神录6面Boss）
15 風神少女（文花帖Level10）
14 幻想郷の二ッ岩（心绮楼猯藏角色曲）
13 東方妖々夢　～ Ancient Temple（妖妖梦5面道中）
12 遠野幻想物語（妖妖梦2面道中）
11 無間の鐘　～ Infinite Nightmare（文花帖DS Level9 10 11 12 Ex）
蟠桃pt10 月まで届け、不死の煙（永夜抄ExBoss）
9 地蔵だけが知る哀嘆（鬼形兽1面道中）
8 彼岸帰航　～ Riverside View（花映塚小町主题曲）
7 永遠の三日天下（弹幕天邪鬼终盘主题曲）
6 幼心地の有頂天（绯想天LS主题曲）
5 六十年目の東方裁判　～ Fate of Sixty Years（花映塚四季映姬主题曲）
4 始原のビート　～ Pristine Beat（辉针城ExBoss）
3 有頂天変　～ Wonderful Heaven（绯想天天子主题曲）
2 亡失のエモーション（心绮楼秦心主题曲）
1 輝く針の小人族　～ Little Princess（辉针城6面Boss）
寄世界于偶像 ～ Idoratrize World

nvidia-smi

### meow

ctrl+M math
ctrl+B bold
ctrl+i italic
ctrl+shift+v preview


### 旧TODO
基于深度学习的歌曲转谱 ×
基于深度学习的声学音乐信号到音乐符号转换

  todo
    声源分离、旋律提取、音符分割、乐谱转换
  
   问题
    名字要高大上一点: 
    保证可行性: 应该可以
    有一定工作量，能用开源代码，开源数据，但要有不同
    
    中文搜不到东西
    一开始不知道英语表达真的很为难，一步步从相关的搜到真正对应的主题
    
    能不能用别人的模型，或者稍作修改，然后数据自己爬取+A2SA对齐(模拟MAESTRO)
    差别在于所识别的音乐类型不同(因为用不同的数据集表现不同，难以通用)
    这样工作量是否足够，可以增加其他功能(非算法)来弥补吗
    最后效果如果不好影响大吗
    
    或者根据别人的模型自创，结合已有数据集训练，用来做想要的效果
    
  原因
    有趣
    不少爱好者、初学者识谱能力有限，但希望得到喜爱的歌曲的乐谱
    通常是流行类的，简单，而且也只需要主旋律的简谱。
    现在有一些类似的软件，但都很复杂，涉及较多的乐理知识，适合专业用户。
    通常以midi结果导出，包括字节跳动最新的那个，
    要用其他方法再从midi转换为简谱，略麻烦

  目前已有
    MIDIto五线谱、五线谱toMIDI、歌曲toMIDI
  
  目标
    最新：  
    做那种对纯音乐、哼唱的提取，不局限于钢琴，也不用处理钢琴的踏板  
    先提取主旋律再转录，两个部分都尽量用深度学习  
    主旋律：CFP SFTF CRNN/patch-CNN HCQT  
    音符：字节那个去掉padal部分

    用深度学习从mp3转换到midi，再使用music21/lilypond实现midi转换到乐谱图片
    先参考开源模型实现，配合数据集(现成、爬取、购买 或者由midi重建mp3再作为数据集)
    
    根据歌曲输出简谱
    后面可增加melody extraction部分，提高准确度
    想办法统一音色只留音高(非必须)
    
    搜集各种midi文件，原曲，学习原曲到midi的直接转换，再从midi直接读取主旋律
    利用爬虫从网上搜集简谱与音乐，但简谱很多都是pdf还得识别
    模仿giantMIDI，做一个二次元细分领域的

    获得主旋律的简谱
    封装成应用，根据歌曲/纯音乐输出简谱
    支持演奏

  
  问题
    1没有现成模型参考-->就参考字节跳动那个改改
    2简谱都是pdf，而且不好找，数据量至少要1000--> 对简谱进行识别，只保留音高，其他细节都去掉-->用现成的Music21
    3歌曲
    
    或者先跟着字节跳动做，再从midi提取成简谱
    
  方法
    AnthemScore 利用机器学习将音频转换为各种乐谱 [RNN实现]
    Wavetone 日本的轻量级软件，实现音乐到MIDI的转变
    Piano transcription 钢琴曲mp3转换MIDI，带有踏板信息 [音频CNN]
    把音乐数据傅里叶变换之后得到的结果再深度学习
    two stage的乐谱识别算法：音乐>主旋律>乐谱; one stage：音乐>乐谱
    
  数据集
    1字节跳动，GiantMIDI-Piano，古典钢琴MIDI数据库，midi格式占用小，准确 midi本身就相当于谱子，包含了很多成分，直接转简谱也不轻松
    市面上自动转midi的也不少，或许有用
    2谷歌 MAESTRO 数据集：ENABLING FACTORIZED PIANO MUSIC MODELINGAND GENERATION WITH THE MAESTRO DATASET
    ps. 数据集里的midi文件应该没有对齐的说法，就是学习音乐到midi的对应而已
    考虑单音数据集https://zhuanlan.zhihu.com/p/78014428
    BitMidi有midi文件 --> 不全是钢琴
    MidiNet
    最重要还是搞清楚数据集结构，然后就是改模型。
    或许可以对比自创模型+已有数据库 与 微改模型+爬取数据的区别
    或许能从b站爬取相关视频，提取音源，匹配midi...
    musescore上有不少乐谱
    研究能否自己爬取数据使之符合MAESTRO 数据集格式

校园美食类微信小程序
  现成，前端部分完全自己的代码，后端部分可以重写，软工作业

虚拟形象驱动KalidoKit https://cuijiahua.com/blog/2021/11/ai-24.html
  （或同类的，仅有面部捕捉的 https://cuijiahua.com/blog/2021/02/ai-12.html）
  有趣，大部分代码依赖开源，较为熟悉，但硬件恐怕支持不了

视频网站
  熟悉，完全自己的代码
  
基于区块链的投票系统
  对区块链略有兴趣，且更透明的投票也有一定需求

