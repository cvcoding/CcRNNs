Namespace(batch_size=32, clip=0.35, 
dropout=0.5, 

emb_dropout=0.25, 

self.drop = nn.Dropout(emb_dropout)
        self.drop2 = nn.Dropout(emb_dropout*2)


��һ��������lstm��Сhidden units����ȫ���ӽ���TCN�����ά�ȣ�ι��LSTM��h.������������Ĳ�������̫�࣬��TCN���

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