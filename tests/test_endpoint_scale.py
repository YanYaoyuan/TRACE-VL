import pytest
import torch
from test_endpoint_reader import ANSWER, TAIL, bb as bb, blocks, episode

from trace_vl.core_vlm.train_endpoint_reader import source_observations, training_sources
from trace_vl.core_vlm.train_endpoint_scale import source_batches


def test_gradient_synchronization_precedes_global_clip_and_optimizer_step(bb):
    reader = episode(bb, blocks(6))
    loss, _ = reader.answer_loss(TAIL, ANSWER)
    loss.backward()
    gradients = [p.grad.clone() for p in reader.trainable]
    norm = float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in gradients])))
    before = [p.detach().clone() for p in reader.trainable]
    called = []

    def synchronize(parameters):
        for parameter, gradient in zip(parameters, gradients, strict=True):
            torch.testing.assert_close(parameter.grad, gradient)
            parameter.grad.mul_(2.)
        called.append(True)

    optimizer = torch.optim.SGD(reader.trainable, lr=.001)
    result = reader.train_step([(TAIL, ANSWER)], optimizer, max_grad_norm=norm,
                                synchronize_gradients=synchronize)
    assert called == [True]
    assert result['gradient_norm_before_clip'] == pytest.approx(2 * norm, rel=1e-5)
    factor = min(1., norm / (2 * norm + 1e-6))
    for parameter, old, gradient in zip(reader.trainable, before, gradients, strict=True):
        torch.testing.assert_close(parameter, old - .001 * factor * 2 * gradient)


def test_failed_collective_closes_episode_without_stepping(bb):
    reader = episode(bb, blocks(5))
    before = [p.detach().clone() for p in reader.trainable]

    def unavailable(parameters):
        raise RuntimeError('collective unavailable')

    optimizer = torch.optim.SGD(reader.trainable, lr=.01)
    with pytest.raises(RuntimeError, match='collective unavailable'):
        reader.train_step([(TAIL, ANSWER)], optimizer, synchronize_gradients=unavailable)
    assert all(torch.equal(p, old) for p, old in zip(reader.trainable, before, strict=True))
    with pytest.raises(RuntimeError, match='closed'):
        reader.answer_loss(TAIL, ANSWER)


def test_source_parallel_schedule_covers_every_source_once_without_padding():
    sources = [[{'source_id': str(i), 'query_time': float(400 + (i * 37) % 250)}] for i in range(64)]
    batches = source_batches(sources, 4)
    assert len(batches) == 16 and all(len(batch) == 4 for batch in batches)
    assert [r[0]['source_id'] for batch in batches[:2] for r in batch] == list(map(str, range(8)))
    assert sorted(int(r[0]['source_id']) for batch in batches for r in batch) == list(range(64))
    assert source_batches(sources, 4) == batches
    with pytest.raises(ValueError, match='without padding'):
        source_batches(sources[:-1], 4)


def test_cached_pts_metadata_is_checked_before_decode_or_model_use(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='cached full-frame PTS metadata differs'):
        source_observations(None, [{'metadata': str(path), 'metadata_sha256': 'wrong'}], parity=False)


def test_expanded_training_role_is_accepted_but_holdout_is_not():
    def row(i):
        return dict(id=str(i), source_id=str(i), source_group_id=str(i), split='candidate_train_scale',
                    question='q', answer='a', query_time=400., endpoint_rule='actual_full_video_container_end',
                    evidence_intervals=None, memory_necessity_verified=False, video=str(i), video_sha256=str(i))
    assert len(training_sources([row(0), row(1)])) == 2
    bad = row(1)
    bad['split'] = 'candidate_dev'
    with pytest.raises(ValueError):
        training_sources([row(0), bad])
