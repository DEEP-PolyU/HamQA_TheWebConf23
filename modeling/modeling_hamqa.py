from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from utils.utils import make_one_hot
from torch_geometric.nn import MessagePassing
from modeling.model_hyperbolic import *
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from modeling.model_hyperbolic import Hyperboloid, Manifold


def get_dim_act_curv(args):
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HyperGAT(nn.Module):
    def __init__(self, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        manifold = Hyperboloid
        super(HyperGAT, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj

class HamLayer(MessagePassing):
    def __init__(self, aggr="add"):
        super(HamLayer, self).__init__(aggr=aggr)

    def forward(self, x, edge_index, edge_embeddings):
        aggr_out = self.propagate(edge_index, x=(x, x), edge_attr=edge_embeddings)  # [N, emb_dim]
        return aggr_out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class HamQA(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim, enc_dim,
                 fc_dim, n_fc_layer, p_fc):
        super().__init__()
        self.message = HamQA_Messege_Passing(k, n_ntype, n_etype, hidden_size=enc_dim)
        self.fc = MLP(sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

    def forward(self, sent_vecs, concept_ids, node_type_ids, adj):
        graph_message = self.message(adj, node_type_ids)[:, 0]  # (batch_size, dim_node)
        question_context = self.fc(sent_vecs)
        message = question_context + graph_message
        return message


class LM_HamQA(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype, enc_dim,
                 fc_dim, n_fc_layer, p_fc, init_range=0.02, encoder_config={}):
        super().__init__()
        self.args = args
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder_type = HyperGAT if 'ham' in args.counter_type else MRN
        self.init_range = init_range
        self.decoder = HamQA(args, k, n_ntype, n_etype, self.encoder.sent_dim, enc_dim, fc_dim, n_fc_layer, p_fc)
        if init_range > 0:
            self.decoder.apply(self._init_weights)

    def forward(self, *inputs, detail=False):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [
            x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x, []) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        sent_vecs, _ = self.encoder(*lm_inputs)
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device),
               edge_type.to(node_type_ids.device))  # edge_index: [2, total_E]   edge_type: [total_E, ]
        logits = self.decoder(sent_vecs.to(node_type_ids.device), concept_ids, node_type_ids, adj)
        return logits.view(bs, nc)

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LM_HamQA_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        print('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path,
                                                                                          model_type, model_name,
                                                                                          max_seq_length,
                                                                                          args.load_sentvecs_model_path)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type,
                                                                                    model_name, max_seq_length,
                                                                                    args.load_sentvecs_model_path)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path,
                                                                                              max_node_num, num_choice,
                                                                                              args)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num,
                                                                                          num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in
                   [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in
                   [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path,
                                                                                           model_type, model_name,
                                                                                           max_seq_length,
                                                                                           args.load_sentvecs_model_path)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path,
                                                                                                max_node_num,
                                                                                                num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in
                       [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        print('max train seq length: ', self.train_encoder_data[1].sum(dim=2).max().item())
        print('max dev seq length: ', self.dev_encoder_data[1].sum(dim=2).max().item())
        if test_statement_path is not None:
            print('max test seq length: ', self.test_encoder_data[1].sum(dim=2).max().item())

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in
                           [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes,
                                                   self.train_qids, self.train_labels, tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                   torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                                   tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                       self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                                       tensors0=self.train_encoder_data,
                                                       tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                       torch.arange(len(self.test_qids)), self.test_qids,
                                                       self.test_labels, tensors0=self.test_encoder_data,
                                                       tensors1=self.test_decoder_data, adj_data=self.test_adj_data)


class HamQA_Messege_Passing(nn.Module):
    def __init__(self, k, n_ntype, n_etype, hidden_size):
        super().__init__()
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.hidden_size = hidden_size
        self.edge_encoder = nn.Sequential(MLP(n_etype + n_ntype * 2, hidden_size, 1, 1, 0, layer_norm=True),
                                          nn.Sigmoid())
        self.k = k
        self.gnn_layers = nn.ModuleList([HamLayer() for _ in range(k)])
        self.regulator = MLP(1, hidden_size, 1, 1, 0, layer_norm=True)

    def get_graph_edge_embedding(self, edge_index, edge_type, node_type_ids):
        edge_vec = make_one_hot(edge_type, self.n_etype)
        node_type = node_type_ids.view(-1).contiguous()
        head_type = node_type[edge_index[0]]
        tail_type = node_type[edge_index[1]]
        head_vec = make_one_hot(head_type, self.n_ntype)
        tail_vec = make_one_hot(tail_type, self.n_ntype)
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))
        return edge_embeddings

    def forward(self, adj, node_type_ids):
        _batch_size, _n_nodes = node_type_ids.size()
        n_node_total = _batch_size * _n_nodes
        edge_index, edge_type = adj

        edge_embeddings = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids)
        aggr_out = torch.zeros(n_node_total, 1).to(node_type_ids.device)
        for i in range(self.k):
            aggr_out = self.gnn_layers[i](aggr_out, edge_index, edge_embeddings)
        aggr_out = self.regulator(aggr_out).view(_batch_size, _n_nodes, -1)
        return aggr_out


class MRN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim, enc_dim, fc_dim, n_fc_layer, p_fc):
        super().__init__()
        self.fc = MLP(sent_dim, fc_dim, 1, 0, p_fc, layer_norm=True)
        if args.counter_type == '1hop':
            self.mlp = MLP(n_etype * n_ntype ** 2, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)
        elif args.counter_type == '2hop':
            self.mlp = MLP(n_etype ** 2 * n_ntype ** 3 + n_etype * n_ntype ** 2, fc_dim, 1, n_fc_layer, p_fc,
                           layer_norm=True)
        else:
            raise NotImplementedError

    def forward(self, sent_vecs, edge_counts, node_type_ids, adj):
        return self.mlp(edge_counts.to(node_type_ids.device)) + self.fc(sent_vecs)
