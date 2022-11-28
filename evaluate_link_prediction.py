try: import cPickle as pickle
except: import pickle
import metrics
import evaluation_util
import numpy as np
import networkx as nx

#import sys
#sys.path.insert(0, './')
#from gem.utils import embed_util





def get_reconstructed_adj(X=None, node_l=None):

    node_num = X.shape[0]

    adj_mtx_r = np.zeros((node_num, node_num))
    for v_i in range(node_num):
        for v_j in range(node_num):
            if v_i == v_j:
                continue
            adj_mtx_r[v_i, v_j] = np.dot(X[v_i, :], X[v_j, :])
    return adj_mtx_r

def evaluateStaticLinkPrediction(train_digraph, test_digraph, X, node_l=None,
                                 n_sample_nodes=None,
                                 sample_ratio_e=None,
                                 no_python=False,
                                 is_undirected=True):

    eval_edge_pairs = None
    estimated_adj = get_reconstructed_adj(X, node_l)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )

    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(e[0], e[1])]

    MAP = metrics.computeMAP(filtered_edge_list, test_digraph)
    prec_curv, _ = metrics.computePrecisionCurve(
        filtered_edge_list,
        test_digraph
    )
    return MAP


def expLP(digraph, graph_embedding,
          n_sample_nodes, rounds,
          res_pre, m_summ, train_ratio=0.8,
          no_python=False, is_undirected=True):
    print('\tLink Prediction')
    summ_file = open('%s_%s.lpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    MAP = [None] * rounds
    prec_curv = [None] * rounds
    for round_id in range(rounds):
        MAP[round_id], prec_curv[round_id] = \
            evaluateStaticLinkPrediction(digraph, graph_embedding,
                                         train_ratio=train_ratio,
                                         n_sample_nodes=1024,
                                         no_python=no_python,
                                         is_undirected=is_undirected)
    summ_file.write('\t%f/%f\t%s\n' % (
        np.mean(MAP),
        np.std(MAP),
        metrics.getPrecisionReport(
            prec_curv[0],
            len(prec_curv[0])
        )
    ))
    summ_file.close()
    pickle.dump([MAP, prec_curv],
                open('%s_%s.lp' % (res_pre, m_summ),
                     'wb'))
