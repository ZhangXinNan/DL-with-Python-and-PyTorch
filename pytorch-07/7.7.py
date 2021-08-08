
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
 
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
 
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
 
    #初始化隐含状态State及C
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
 
    def forward(self, sentence):
        #获得词嵌入矩阵embeds
        embeds = self.word_embeddings(sentence)   
        #按lstm格式，修改embeds的形状
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        #修改隐含状态的形状，作为全连接层的输入
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #计算每个单词属于各词性的概率
        tag_scores = F.log_softmax(tag_space,dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return tensor



#定义训练数据
training_data = [
    ("The cat ate the fish".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("They read that book".split(), ["NN", "V", "DET", "NN"])
]
#定义测试数据
testing_data=[("They ate the fish".split())]

word_to_ix = {} # 单词的索引字典
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print('单词的索引字典：')
print(word_to_ix)


tag_to_ix = {"DET": 0, "NN": 1, "V": 2} # 手工设定词性标签数据字典
print('手工设定词性标签数据字典：')
print(tag_to_ix)

EMBEDDING_DIM=10
HIDDEN_DIM=3  #这里等于词性个数

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
 
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(training_data[0][0])
print(inputs)
print(tag_scores)
print(torch.max(tag_scores,1))

for epoch in range(400): # 我们要训练400次。
    for sentence, tags in training_data:
# 清除网络先前的梯度值
        model.zero_grad()
# 重新初始化隐藏层数据
        model.hidden = model.init_hidden()
# 按网络要求的格式处理输入数据和真实标签数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
# 实例化模型
        tag_scores = model(sentence_in)
# 计算损失，反向传递梯度及更新模型参数
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
 
# 查看模型训练的结果
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(training_data[0][0])
print(tag_scores)
print(torch.max(tag_scores,1))


test_inputs = prepare_sequence(testing_data[0], word_to_ix)
tag_scores01 = model(test_inputs)
print(testing_data[0])
print(test_inputs)
print(tag_scores01)
print(torch.max(tag_scores01,1))

