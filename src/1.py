from utils import *
seed = 42
set_metaverse_seed(seed)

tokenizer = AutoTokenizer.from_pretrained("lucashu/TcmYiAnBERT")
mlm_model = AutoModelForMaskedLM.from_pretrained("lucashu/TcmYiAnBERT")
model = AutoModel.from_pretrained("lucashu/TcmYiAnBERT")

data = pd.read_csv('../data/tcmspt.csv').fillna('未知')
text_concat = '年龄：' + data['Age'].astype(str) + '。性别：' +data['Gender'] + '。病史：' + data['Medical History'] + '舌诊：' + data['Tongue'] + '。脉诊：' + data['Pulse'] + '。'
feat = text_concat.to_list()
label = data.Type.to_list()

train_x, train_y, valid_x, valid_y, test_x, test_y = train_valid_test_split(feat, label)

print(f'seed: {seed}')
print(f'tokenizer: {tokenizer}')
# print(f'model: {model}')
print('[cyan]data[/]')
print(f'total: {len(data)}')
print('label distribution:')
show_distribution(label)

print(f'example: {feat[0]}')
tokenized_text = tokenizer.tokenize(feat[0])
token_idx = tokenizer.convert_tokens_to_ids(tokenized_text)
token_tensor = torch.tensor([token_idx])

encode_idx = tokenizer.encode(feat[0])
encode_text = ''.join(tokenizer.convert_ids_to_tokens(encode_idx))
encode_tensor = torch.tensor([encode_idx])

out = mlm_model(token_tensor)
out_token_idx = out.logits[0].argmax(1).tolist()
out_token = tokenizer.convert_ids_to_tokens(out_token_idx)
for i in range(len(tokenized_text)):
    if out_token[i] != tokenized_text[i]:
        out_token[i] = '[red]' + out_token[i] + '[/]'
out_text = ''.join(out_token)

out2 = mlm_model(encode_tensor)
out2_token_idx = out2.logits[0].argmax(1).tolist()
out2_token = tokenizer.convert_ids_to_tokens(out2_token_idx)
for i in range(len(tokenized_text)):
    if out2_token[i+1] != tokenized_text[i]:
        out2_token[i+1] = '[red]' + out2_token[i+1] + '[/]'
out2_text = ''.join(out2_token)

print(f'tokenized: {tokenized_text}')
print(f'tokenized index: {token_idx}')
print(f'encode index: {encode_idx}')
print(f'encode text: {encode_text}')
print(f'model out: {out_token_idx}')
print(f'model out text: {out_text}')
print(f'model out (encode): {out2_token_idx}')
print(f'model out text: {out2_text}')

# breakpoint()

