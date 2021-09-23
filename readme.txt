Namespace(batch_size=32, clip=0.35, 
dropout=0.5, 

emb_dropout=0.25, 

self.drop = nn.Dropout(emb_dropout)
        self.drop2 = nn.Dropout(emb_dropout*2)


下一步，可以lstm用小hidden units，用全连接降低TCN输出的维度，喂给LSTM的h.这样整个网络的参数不会太多，跟TCN差不多

emsize=600, 
epochs=100, 
ksize=3, 
levels=4, 
log_interval=100, 
lr=4, 
nhid=600, 
nhid_lstm=600, 
optim='SGD', 
seed=1111, 
seq_len=80, 
tied=True, 
validseqlen=40)
Weight tied