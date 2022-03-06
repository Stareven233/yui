from transformers import T5Model, T5Config
# TODO 换成T5_tiny: D:\A日常\大学\毕业设计\code\t5x\t5x\examples\t5\t5_1_1

config = T5Config.from_json_file('./yui/config/model.json')
# config.to_json_file('config.json', False)
model = T5Model(config=config)
