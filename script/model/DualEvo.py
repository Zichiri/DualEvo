# 带有msa特征分支的GTM模型，GTM模型的输入特征中不包含msa特征
import torch
import torch.nn as nn
class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=4, num_neighbor=30, drop_rate=0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_neighbor = num_neighbor
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None,use_top=True):
        # q: bsz, protein_len, hid=heads*hidd'
        # q,k,v:[1,L,D_out]->[1,num_head,L,head_dim]
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)    # q: bsz, heads, protein_len, hid'
        v = self.transpose_for_scores(v)
        # attention_scores：[1,num_head,L,L]
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) # bsz, heads, protein_len, protein_len + bsz, 1, protein_len, protein_len
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # 按照距离矩阵添加掩码，每一行的注意力只保留num_neighbor个最近的残基的注意力
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            top_mask = (attention_weight_sorted_sorted < self.num_neighbor)
            attention_probs = attention_probs * top_mask
            # 归一化，保证每一行注意力分数和是1
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
        # outputs：[1,num_head,L,L]*[1,num_head,L,head_dim] = [1,num_head,L,head_dim]
        outputs = torch.matmul(attention_probs, v)
        # [1,num_head,L,head_dim]->[1,L,num_head,head_dim]
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        # [1,L,num_head,head_dim] -> [1,L,D_out]
        outputs = outputs.view(*new_output_shape)
        outputs = self.dp(outputs)
        outputs = self.ln(outputs)
        return outputs

# batch_size = B, residue数 = L, 输入特征维度 = D_in, 输出特征维度 = D_out = protein_out_dim,msa特征残差分支输出维度=mlp_out_dim
class GTM(nn.Module):
    def __init__(self, protein_in_dim, protein_out_dim=64,mlp_out_dim=64, target_dim=1, fc_layer_num=2, atten_layer_num=2, atten_head=4, num_neighbor=30, drop_rate1=0.2, drop_rate2=0):
        super().__init__()
        self.msa_dim = 3 * 768 # msa特征维度
        assert protein_in_dim >= self.msa_dim, "protein_in_dim must be >= 3*768"
        self.non_msa_dim = protein_in_dim - self.msa_dim
        # 输入层 [B, L, D_in]->[B, L, D_out]
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(self.non_msa_dim, elementwise_affine=True)
                                        ,nn.Linear(self.non_msa_dim, protein_out_dim)
                                        ,nn.LeakyReLU()
                                        )

        # MSA 分支（单独的 MLP），输入是最后的 3*768 维
        self.msa_mlp = nn.Sequential(
            nn.LayerNorm(self.msa_dim, elementwise_affine=True),
            nn.Linear(self.msa_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, mlp_out_dim),
            nn.LeakyReLU(),
        )

        # 多层前馈网络层 [B, L, D_out]->[B, L, D_out]
        # fc_layer_num表示层数，每层 LayerNorm → Dropout → Linear → LeakyReLU
        self.hidden_block = []
        for h in range(fc_layer_num-1):
            if h < fc_layer_num-1-1:
                self.hidden_block.extend([
                                          nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                                         ,nn.Dropout(drop_rate1)
                                         ,nn.Linear(protein_out_dim, protein_out_dim)
                                         ,nn.LeakyReLU()
                                         ])
            else:
                # 最后一层多加了一个LayerNorm层归一化
                self.hidden_block.extend([
                                          nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                                         ,nn.Dropout(drop_rate1)
                                         ,nn.Linear(protein_out_dim, protein_out_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                                         ])
        self.hidden_block = nn.Sequential(*self.hidden_block)

        # 多层自注意力模块
        self.layers = nn.ModuleList([Self_Attention(protein_out_dim, atten_head, num_neighbor, drop_rate2) for _ in range(atten_layer_num)])
        # 输出层
        self.logit = nn.Linear(protein_out_dim + mlp_out_dim,target_dim)


    def forward(self, protein_node_features, protein_dist_matrix, protein_masks, return_embedding=False):
        # 提取msa特征
        msa_feat = protein_node_features[:, :, -self.msa_dim:]
        non_msa_feat = protein_node_features[:, :, :-self.msa_dim]
        # [B, L, mlp_out_dim]
        msa_out = self.msa_mlp(msa_feat)
        # [1,L,len(node_f)] -> [1,L,D_out]
        protein_embedding = self.input_block(non_msa_feat)
        # [1,L,D_out] -> [1,L,D_out]
        protein_embedding = self.hidden_block(protein_embedding)
        # dist_weight [1,L,L] * [1,1,L] -> [1,L,L] 将距离转换成了注意力权重：距离越大，权重越小
        dist_weight = 1.0 / torch.sqrt(1.0+protein_dist_matrix) * protein_masks.unsqueeze(1)
        # 每一行进行归一化，使得每个残基对其他所有残基的距离权重加起来为 1
        dist_weight = dist_weight / torch.sum(dist_weight,axis=-1,keepdim=True)
        # protein_masks:[1,L] 都是0
        protein_masks = (1.0-protein_masks) * -10000
        # 这里的qkv直接使用了embedding,没有可训练参数
        for layer in self.layers:
            protein_embedding = layer(protein_embedding, protein_embedding, protein_embedding, protein_masks.unsqueeze(1).unsqueeze(1), dist_weight.unsqueeze(1)) #.squeeze(1)#,dist_weight.unsqueeze(1)
        # [B, L, protein_out_dim + mlp_out_dim]
        protein_out = torch.cat([protein_embedding, msa_out], dim=-1)
        y = self.logit(protein_out).squeeze(-1) # batch_size * L
        if return_embedding:
            return y, protein_out
        return y



