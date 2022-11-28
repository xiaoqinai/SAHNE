try: import cPickle as pickle
except: import pickle
import metrics
import evaluation_util
import networkx as nx
import numpy as np






def get_reconstructed_adj(X=None, node_l=None):

    node_num = X.shape[0]

    adj_mtx_r = np.zeros((node_num, node_num))
    for v_i in range(node_num):
        for v_j in range(node_num):
            if v_i == v_j:
                continue
            adj_mtx_r[v_i, v_j] = np.dot(X[v_i, :], X[v_j, :])
    return adj_mtx_r


def evaluateStaticGraphReconstruction(digraph,
                                      X_stat, node_l=None, file_suffix=None,
                                      sample_ratio_e=None, is_undirected=True,
                                      is_weighted=False):

    eval_edge_pairs = None
    if file_suffix is None:
        estimated_adj = get_reconstructed_adj(X_stat, node_l)

    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )
    MAP = metrics.computeMAP(predicted_edge_list, digraph)
#    prec_curv, _ = metrics.computePrecisionCurve(predicted_edge_list, digraph)
    # If weighted, compute the error in reconstructed weights of observed edges
    return MAP


def expGR(digraph, graph_embedding,
          X, n_sampled_nodes, rounds,
          res_pre, m_summ,
          is_undirected=True):
    print('\tGraph Reconstruction')
    summ_file = open('%s_%s.grsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    if digraph.number_of_nodes() <= n_sampled_nodes:
        rounds = 1
    MAP = [None] * rounds
    prec_curv = [None] * rounds
    err = [None] * rounds
    err_b = [None] * rounds
    n_nodes = [None] * rounds
    n_edges = [None] * rounds
    for round_id in range(rounds):
        sampled_digraph, node_l = graph_util.sample_graph(
            digraph,
            n_sampled_nodes=n_sampled_nodes
        )
        n_nodes[round_id] = sampled_digraph.number_of_nodes()
        n_edges[round_id] = sampled_digraph.number_of_edges()
        print('\t\tRound: %d, n_nodes: %d, n_edges:%d\n' % (round_id,
                                                            n_nodes[round_id],
                                                            n_edges[round_id]))
        sampled_X = X[node_l]
        MAP[round_id], prec_curv[round_id], err[round_id], err_b[round_id] = \
            evaluateStaticGraphReconstruction(sampled_digraph, graph_embedding,
                                              sampled_X, node_l,
                                              is_undirected=is_undirected)
    try:
        summ_file.write('Err: %f/%f\n' % (np.mean(err), np.std(err)))
        summ_file.write('Err_b: %f/%f\n' % (np.mean(err_b), np.std(err_b)))
    except TypeError:
        pass
    summ_file.write('%f/%f\t%s\n' % (np.mean(MAP), np.std(MAP),
                                     metrics.getPrecisionReport(prec_curv[0],
                                                                n_edges[0])))
    pickle.dump([n_nodes,
                 n_edges,
                 MAP,
                 prec_curv,
                 err,
                 err_b],
                open('%s_%s.gr' % (res_pre, m_summ), 'wb'))
