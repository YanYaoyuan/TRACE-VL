import copy

import pytest
import torch
from test_endpoint_reader import ANSWER, HEAD, TAIL, blocks
from test_endpoint_reader import bb as bb

from trace_vl.core_vlm.endpoint_reader import EndpointReaderEpisode


def frozen_copy(backbone):
    prefix = copy.deepcopy(backbone)
    prefix.model.eval().requires_grad_(False)
    return prefix


def make_episode(query, items, prefix=None):
    result = EndpointReaderEpisode(query, HEAD, 'reservoir', recent=2, capacity=2,
                                   seed=17, prefix_bb=prefix)
    for item in items:
        result.append(item)
    result.seal(items[-1].end)
    return result


def assert_same_cache(first, second):
    assert first[2:] == second[2:]
    assert len(first[0]) == len(second[0]) and len(first[1]) == len(second[1])
    for pair_a, pair_b in zip(first[0], second[0], strict=True):
        assert all(torch.equal(a, b) for a, b in zip(pair_a, pair_b, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(first[1], second[1], strict=True))


def test_equal_initial_adapters_match_shared_reader_loss_and_query_gradients(bb):
    query, prefix = copy.deepcopy(bb), frozen_copy(bb)
    items = blocks(8)
    shared = make_episode(bb, items)
    separated = make_episode(query, items, prefix)
    assert_same_cache(shared.stream.snapshot(), separated.stream.snapshot())
    shared_loss, _ = shared.answer_loss(TAIL, ANSWER)
    separate_loss, _ = separated.answer_loss(TAIL, ANSWER)
    torch.testing.assert_close(separate_loss, shared_loss, atol=3e-5, rtol=1e-4)
    shared_loss.backward()
    separate_loss.backward()
    shared_params = {name: p for name, p in bb.model.named_parameters() if p.requires_grad}
    query_params = {name: p for name, p in query.model.named_parameters() if p.requires_grad}
    assert shared_params.keys() == query_params.keys()
    for name in shared_params:
        assert shared_params[name].grad is not None and query_params[name].grad is not None
        torch.testing.assert_close(query_params[name].grad, shared_params[name].grad, atol=3e-5, rtol=1e-4)
    assert all(p.grad is None for p in prefix.model.parameters())


def test_question_gradients_reach_only_query_lora_and_preserve_live_prefix(bb):
    prefix, items = frozen_copy(bb), blocks(8)
    reader = make_episode(bb, items, prefix)
    before, rng = reader.stream.snapshot(), reader.stream.rng.getstate()
    loss, _ = reader.answer_loss(TAIL, ANSWER)
    loss.backward()
    reader.stream.verify(before)
    assert reader.stream.rng.getstate() == rng
    assert all(item.tokens.grad is None for item in items)
    assert all(p.grad is None for p in prefix.model.parameters())
    assert all(p.grad is None for p in bb.model.parameters() if not p.requires_grad)
    query_params = [p for p in bb.model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in query_params)
    assert sum(float(p.grad.abs().sum()) for p in query_params) > 0
    assert {id(p) for p in reader.trainable} == {id(p) for p in query_params}


def test_rebuilt_ingestion_is_bit_identical_after_query_optimizer_update(bb):
    prefix, items = frozen_copy(bb), blocks(8)
    reader = make_episode(bb, items, prefix)
    before, audit = reader.stream.snapshot(), reader.stream.audit()
    prefix_parameters = [p.detach().clone() for p in prefix.model.parameters()]
    query_parameters = [p.detach().clone() for p in reader.trainable]
    optimizer = torch.optim.SGD(reader.trainable, lr=.1)
    reader.train_step([(TAIL, ANSWER), ([13, 14], [42, 2])], optimizer)
    assert any(not torch.equal(a, b) for a, b in zip(query_parameters, reader.trainable, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(prefix_parameters, prefix.model.parameters(), strict=True))
    assert all(p.grad is None for p in prefix.model.parameters())
    with pytest.raises(RuntimeError, match='closed'):
        reader.answer_loss(TAIL, ANSWER)
    rebuilt = make_episode(bb, items, prefix)
    assert rebuilt.stream.audit() == audit
    assert_same_cache(before, rebuilt.stream.snapshot())
    assert rebuilt.answer_loss(TAIL, ANSWER)[0].requires_grad


def test_external_prefix_weight_mutation_invalidates_cached_prefix(bb):
    prefix = frozen_copy(bb)
    reader = make_episode(bb, blocks(4), prefix)
    with torch.no_grad():
        next(prefix.model.parameters()).add_(.01)
    with pytest.raises(RuntimeError, match='stale'):
        reader.answer_loss(TAIL, ANSWER)
    with pytest.raises(RuntimeError, match='closed'):
        reader.answer_loss(TAIL, ANSWER)


def test_prefix_mode_change_invalidates_episode(bb):
    prefix = frozen_copy(bb)
    reader = make_episode(bb, blocks(4), prefix)
    prefix.model.train()
    with pytest.raises(RuntimeError, match='prefix model mode'):
        reader.answer_loss(TAIL, ANSWER)
    with pytest.raises(RuntimeError, match='closed'):
        reader.answer_loss(TAIL, ANSWER)


def test_trainable_or_non_eval_prefix_is_refused_before_cache_creation(bb):
    with pytest.raises(ValueError, match='fully frozen'):
        EndpointReaderEpisode(bb, HEAD, prefix_bb=copy.deepcopy(bb))
    prefix = frozen_copy(bb)
    prefix.model.train()
    with pytest.raises(ValueError, match='eval mode'):
        EndpointReaderEpisode(bb, HEAD, prefix_bb=prefix)


@pytest.mark.parametrize('change', ['hidden', 'dtype', 'token', 'cache_geometry'])
def test_incompatible_prefix_signature_is_refused(bb, change):
    prefix = frozen_copy(bb)
    if change == 'hidden':
        prefix.hidden *= 2
    elif change == 'dtype':
        prefix.dtype = torch.float64
    elif change == 'token':
        prefix.vision_start += 1
    else:
        prefix.text_config.num_key_value_heads += 1
    with pytest.raises(ValueError, match='configurations must match'):
        EndpointReaderEpisode(bb, HEAD, prefix_bb=prefix)


def test_independent_parameter_objects_sharing_storage_are_refused(bb):
    prefix = frozen_copy(bb)
    query_parameters = dict(bb.model.named_parameters())
    name = next(name for name, p in query_parameters.items() if not p.requires_grad)
    prefix_parameter = dict(prefix.model.named_parameters())[name]
    assert prefix_parameter is not query_parameters[name]
    # Separate Parameter identities are insufficient if storage was aliased.
    prefix_parameter.data = query_parameters[name].data
    with pytest.raises(ValueError, match='storage must be independent'):
        EndpointReaderEpisode(bb, HEAD, prefix_bb=prefix)


def test_optimizer_cannot_own_frozen_prefix_parameters(bb):
    prefix = frozen_copy(bb)
    reader = make_episode(bb, blocks(4), prefix)
    optimizer = torch.optim.SGD([*reader.trainable, next(prefix.model.parameters())], lr=.1)
    with pytest.raises(ValueError, match="exactly the episode's"):
        reader.train_step([(TAIL, ANSWER)], optimizer)
