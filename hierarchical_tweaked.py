import numpy as np
import sklearn.cluster.hierarchical 
from sklearn.cluster.hierarchical import _complete_linkage
from sklearn.cluster.hierarchical import _average_linkage
from sklearn.cluster.hierarchical import _fix_connectivity
from sklearn.cluster.hierarchical import _hc_cut
from sklearn.cluster.hierarchical import * 
from sklearn.cluster import _hierarchical

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.externals.joblib import Memory

###############################################################################
# Hierarchical tree building functions

def ward_tree_matrix(X, connectivity=None, n_clusters=None, return_distance=False):
    """Ward clustering based on a Feature matrix.
    Recursively merges the pair of clusters that minimally increases
    within-cluster variance.
    The inertia matrix uses a Heapq-based representation.
    This is the structured version, that takes into account some topological
    structure between samples.
    Read more in the :ref:`User Guide <hierarchical_clustering>`.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered
    connectivity : sparse matrix (optional).
        connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.
    n_clusters : int (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.
    return_distance: bool (optional)
        If True, return the distance between the clusters.
    Returns
    -------
    children : 2D array, shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`
    n_components : int
        The number of connected components in the graph.
    n_leaves : int
        The number of leaves in the tree
    parents : 1D array, shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.
    distances : 1D array, shape (n_nodes-1, )
        Only returned if return_distance is set to True (for compatibility).
        The distances between the centers of the nodes. `distances[i]`
        corresponds to a weighted euclidean distance between
        the nodes `children[i, 1]` and `children[i, 2]`. If the nodes refer to
        leaves of the tree, then `distances[i]` is their unweighted euclidean
        distance. Distances are updated in the following way
        (from scipy.hierarchy.linkage):
        The new entry :math:`d(u,v)` is computed as follows,
        .. math::
           d(u,v) = \\sqrt{\\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \\frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \\frac{|v|}
                               {T}d(s,t)^2}
        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    if connectivity is None:
        from scipy.cluster import hierarchy     # imports PIL

        if n_clusters is not None:
            warnings.warn('Partial build of the tree is implemented '
                          'only for structured clustering (i.e. with '
                          'explicit connectivity). The algorithm '
                          'will build the full tree and only '
                          'retain the lower branches required '
                          'for the specified number of clusters',
                          stacklevel=2)
        out = hierarchy.ward(X)
        children_ = out[:, :2].astype(np.intp)

        if return_distance:
            distances = out[:, 2]
            return children_, 1, n_samples, None, distances
        else:
            return children_, 1, n_samples, None

    connectivity, n_components = _fix_connectivity(X, connectivity)
    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        if n_clusters > n_samples:
            raise ValueError('Cannot provide more clusters than samples. '
                             '%i n_clusters was asked, and there are %i samples.'
                             % (n_clusters, n_samples))
        n_nodes = 2 * n_samples - n_clusters

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.intp, order='C')
    coord_col = np.array(coord_col, dtype=np.intp, order='C')

    # build moments as a list
    moments_1 = np.zeros(n_nodes, order='C')
    moments_1[:n_samples] = 1
    moments_2 = np.zeros((n_nodes, n_features), order='C')
    moments_2[:n_samples] = X
    inertia = np.empty(len(coord_row), dtype=np.float64, order='C')
    _hierarchical.compute_ward_dist(moments_1, moments_2, coord_row, coord_col,
                                    inertia)
    inertia = list(six.moves.zip(inertia, coord_row, coord_col))
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []
    if return_distance:
        distances = np.empty(n_nodes - n_samples)

    not_visited = np.empty(n_nodes, dtype=np.int8, order='C')
    number_of_desc = []

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            #print inertia # lengte van new heap
            # kprint n_samples
            inert, i, j = heappop(inertia)

            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j] = k, k
        children.append((i, j))

        i_n = 1
        j_n = 1
        if i >= n_samples:
            i_n = number_of_desc[i-n_samples]
        if j >= n_samples:
            j_n = number_of_desc[j-n_samples]
        number_of_desc.append(i_n+j_n)

        # print _hierarchical._hc_get_descendent(i, children, k)
        # print _hierarchical._hc_get_descendent(j, children, k)
        used_node[i] = used_node[j] = False
        if return_distance:  # store inertia value
            distances[k - n_samples] = inert

        # update the moments
        moments_1[k] = moments_1[i] + moments_1[j]
        moments_2[k] = moments_2[i] + moments_2[j]

        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _hierarchical._get_parents(A[i], coord_col, parent, not_visited)
        _hierarchical._get_parents(A[j], coord_col, parent, not_visited)
        # List comprehension is faster than a for loop
        [A[l].append(k) for l in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.intp, order='C')
        coord_row = np.empty(coord_col.shape, dtype=np.intp, order='C')
        coord_row.fill(k)
        n_additions = len(coord_row)
        ini = np.empty(n_additions, dtype=np.float64, order='C')

        _hierarchical.compute_ward_dist(moments_1, moments_2,
                                        coord_row, coord_col, ini)

        # List comprehension is faster than a for loop
        [heappush(inertia, (ini[idx], k, coord_col[idx]))
            for idx in range(n_additions)]

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    # sort children to get consistent output with unstructured version
    children = [c[::-1] for c in children]
    children = np.array(children)  # return numpy array for efficient caching

    if return_distance:
        # 2 is scaling factor to compare w/ unstructured version
        distances = np.sqrt(2. * distances)
        return children, n_components, n_leaves, parent, distances, number_of_desc
    else:
        return children, n_components, n_leaves, parent

_TREE_BUILDERS = dict(
    ward=ward_tree_matrix,
    complete=_complete_linkage,
    average=_average_linkage)

class AgglomerativeClusteringTreeMatrix(BaseEstimator, ClusterMixin):
    """
    Agglomerative Clustering
    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.
    Read more in the :ref:`User Guide <hierarchical_clustering>`.
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    connectivity : array-like or callable, optional
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.
    affinity : string or callable, default: "euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or 'precomputed'.
        If linkage is "ward", only "euclidean" is accepted.
    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    compute_full_tree : bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of clusters and using caching, it may
        be advantageous to compute the full tree.
    linkage : {"ward", "complete", "average"}, optional, default: "ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
    pooling_func : callable, default=np.mean
        This combines the values of agglomerated features into a single
        value, and should accept an array of shape [M, N] and the keyword
        argument ``axis=1``, and reduce it to an array of size [M].
    Attributes
    ----------
    labels_ : array [n_samples]
        cluster labels for each point
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_components_ : int
        The estimated number of connected components in the graph.
    children_ : array-like, shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`
    """

    def __init__(self, n_clusters=2, affinity="euclidean",
                 memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, compute_full_tree='auto',
                 linkage='ward', pooling_func=np.mean):
        self.n_clusters = n_clusters
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.pooling_func = pooling_func

    def fit(self, X, y=None):
        """Fit the hierarchical clustering on the data
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The samples a.k.a. observations.
        Returns
        -------
        self
        """
        # X = check_array(X, ensure_min_samples=2, estimator=self)
        memory = self.memory
        # if isinstance(memory, six.string_types):
        #    memory = Memory(cachedir=memory, verbose=0)

        #if self.n_clusters <= 0:
        #    raise ValueError("n_clusters should be an integer greater than 0."
        #                     " %s was provided." % str(self.n_clusters))

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError("%s was provided as affinity. Ward can only "
                             "work with euclidean distances." %
                             (self.affinity, ))

        if self.linkage not in _TREE_BUILDERS:
            raise ValueError("Unknown linkage type %s."
                             "Valid options are %s" % (self.linkage,
                                                       _TREE_BUILDERS.keys()))
        tree_builder = _TREE_BUILDERS[self.linkage]

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(
                connectivity, accept_sparse=['csr', 'coo', 'lil'])

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            # Early stopping is likely to give a speed up only for
            # a large number of clusters. The actual threshold
            # implemented here is heuristic
            compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        kwargs = {}
        kwargs['return_distance'] = True
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self.affinity
        self.children_, self.n_components_, self.n_leaves_, parents, self.distance, self.n_desc = \
            memory.cache(tree_builder)(X, connectivity,
                                       n_clusters=None, **kwargs)
        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters, self.children_,
                                   self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reassign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self
