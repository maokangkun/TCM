from utils import *
seed = 42
set_metaverse_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
lr = 1e-3
epochs = 500

# tokenizer = AutoTokenizer.from_pretrained("lucashu/TcmYiAnBERT")
# mlm_model = AutoModelForMaskedLM.from_pretrained("lucashu/TcmYiAnBERT").to(device)
# model = AutoModel.from_pretrained("lucashu/TcmYiAnBERT").to(device)

med_word_vector = {}
with open('../downloads/Medical.txt', 'r') as f:
    for line in f.readlines()[1:]:
        if len((arr := line.split())) == 513:
            word = arr[0]
            vector = list(map(float, arr[1:]))
            med_word_vector[word] = vector

breakpoint()


data = pd.read_csv('../data/tcmspt.csv').fillna('未知')
text_concat = '年龄：' + data['Age'].astype(str) + '。性别：' +data['Gender'] + '。病史：' + data['Medical History'] + '舌诊：' + data['Tongue'] + '。脉诊：' + data['Pulse'] + '。'
feat = text_concat.to_list()
label = data.Type.to_list()
label_map = {k:i for i, k in enumerate(set(label))}
label_idx = [label_map[k] for k in label]
out_dim = len(label_map)
# in_dim = model.pooler.dense.out_features

print(f'seed: {seed}')
print(f'tokenizer: {tokenizer}')
# print(f'model: {model}')
print(f'total data: {len(data)}')
print('label distribution:')
show_distribution(label)
test_example(tokenizer, mlm_model, feat)
