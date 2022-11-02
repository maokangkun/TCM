from utils import *
seed = 42
set_metaverse_seed(seed)

tokenizer = AutoTokenizer.from_pretrained("lucashu/TcmYiAnBERT")
model = AutoModelForMaskedLM.from_pretrained("lucashu/TcmYiAnBERT")

data = pd.read_csv('../data/tcmspt.csv').fillna('未知')
text_concat = '年龄：' + data['Age'].astype(str) + '。性别：' +data['Gender'] + '。病史：' + data['Medical History'] + '。舌诊：' + data['Tongue'] + '。脉诊：' + data['Pulse'] + '。'
feat = text_concat.to_list()
label = data.Type.to_list()
show_distribution(label)
train_x, train_y, valid_x, valid_y, test_x, test_y = train_valid_test_split(feat, label)
# print(train_y)

# print(dir(tokenizer))


# from transformers import pipeline
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# out = classifier(feat[0])

out = tokenizer.encode(feat[0])
print(model(torch.tensor(out)))
# print(out.tokens)
# breakpoint()

