from utils import *
seed = 42
set_metaverse_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
lr = 1e-3
epochs = 500

tokenizer = AutoTokenizer.from_pretrained("lucashu/TcmYiAnBERT")
# mlm_model = AutoModelForMaskedLM.from_pretrained("lucashu/TcmYiAnBERT").to(device)
model = AutoModel.from_pretrained("lucashu/TcmYiAnBERT").to(device)

data = pd.read_csv('../data/tcmspt.csv').fillna('未知')
text_concat = '年龄：' + data['Age'].astype(str) + '。性别：' +data['Gender'] + '。病史：' + data['Medical History'] + '舌诊：' + data['Tongue'] + '。脉诊：' + data['Pulse'] + '。'
feat = text_concat.to_list()
label = data.Type.to_list()
label_map = {k:i for i, k in enumerate(set(label))}
label_idx = [label_map[k] for k in label]
out_dim = len(label_map)
in_dim = model.pooler.dense.out_features

print(f'seed: {seed}')
print(f'tokenizer: {tokenizer}')
# print(f'model: {model}')
print(f'total data: {len(data)}')
print('label distribution:')
show_distribution(label)
# test_example(tokenizer, mlm_model, feat)

batch_encode = tokenizer.batch_encode_plus(feat, padding=True)
input_ids = torch.tensor(batch_encode['input_ids']).to(device)
token_type_ids = torch.tensor(batch_encode['token_type_ids']).to(device)
attention_mask = torch.tensor(batch_encode['attention_mask']).to(device)
model.eval()
with torch.no_grad():
    out = model(input_ids, token_type_ids, attention_mask)
    embedding = out.last_hidden_state[:,0]
    pooled = out.pooler_output
# train_feat, train_label, valid_feat, valid_label, test_feat, test_label = train_valid_test_split(embedding, label_idx)
train_feat, train_label, valid_feat, valid_label, test_feat, test_label = train_valid_test_split(pooled, label_idx)
train_data = TCMDataset(train_feat, train_label)
valid_data = TCMDataset(valid_feat, valid_label)
test_data = TCMDataset(test_feat, test_label)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.fc(x)

mlp = MLP(in_dim, out_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

print(f'train data: {len(train_data)}')
print(f'valid data: {len(valid_data)}')
print(f'test data: {len(test_data)}')
print(f'input dim: {in_dim}')
print(f'output dim: {out_dim}')
print(f'batch size: {batch_size}')
print(f'learning rate: {lr}')
print(f'epochs: {epochs}')
print(f'mlp: {mlp}')

def train_a_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 10 == 0:
        loss, current = loss.item(), batch * len(x)
        print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
    train_loss /= num_batches
    correct /= size
    return train_loss, correct

def test_a_epoch(dataloader, model, loss_fn, prefix='Test'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{prefix} dataset:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct

result = {'train': [[],[]], 'valid': [[],[]], 'test': [[],[]]}
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n-------------------------------")
    train_loss, train_acc = train_a_epoch(train_dataloader, mlp, loss_fn, optimizer)
    valid_loss, valid_acc = test_a_epoch(valid_dataloader, mlp, loss_fn, 'Valid')
    test_loss, test_acc = test_a_epoch(test_dataloader, mlp, loss_fn)
    result['train'][0].append(train_loss)
    result['train'][1].append(train_acc)
    result['valid'][0].append(valid_loss)
    result['valid'][1].append(valid_acc)
    result['test'][0].append(test_loss)
    result['test'][1].append(test_acc)

plot(result, 'mlp-4')

# breakpoint()
