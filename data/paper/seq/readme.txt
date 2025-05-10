该超参数用于处理
1）不同类别基模型与各种策略的对比实验
Seq2, NAR, Teach，Free, SS, PF, RL; 其中在此处，RL和PF均为标号1的进行实验，也即运行的是Seq2_RL_1和Seq_PF_1；
ELSTM, NAR, Teach，Free, SS, PF, RL
Informer,NAR, Teach，Free, SS, PF, RL

2）Seq2的消融处理
Seq2_RL_MLP：在采用强化学习进行选择时，只考虑到MLP模型
Seq2_RL_MSVR：在采用强化学习进行选择时，只考虑到MSVR模型
