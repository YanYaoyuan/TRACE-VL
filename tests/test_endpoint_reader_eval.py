import numpy as np
import pytest

from trace_vl.core_vlm.endpoint_reader_eval import cluster_paired


def test_byte_cluster_bootstrap_keeps_source_weighted_point_estimate():
    # Unequal cluster sizes: preserve the three-source mean, not the mean of two cluster means.
    result = cluster_paired([1., 1., -2.], ['same-video', 'same-video', 'different-video'])
    assert result['sources'] == 3 and result['byte_clusters'] == 2
    assert result['mean'] == pytest.approx(0.)
    assert result['ci97_5'][0] <= result['ci95'][0] <= result['mean']
    assert result['ci97_5'][1] >= result['ci95'][1] >= result['mean']
    assert cluster_paired([1., 1., -2.], ['same-video', 'same-video', 'different-video']) == result


def test_unclustered_duplicates_cannot_be_presented_as_independent_sources():
    with pytest.raises(ValueError, match='two independent'):
        cluster_paired([1., -1.], ['same-video', 'same-video'])
    for values, clusters in [([np.nan, 0.], ['a', 'b']), ([1., 2.], ['a'])]:
        with pytest.raises(ValueError, match='paired finite'):
            cluster_paired(values, clusters)
