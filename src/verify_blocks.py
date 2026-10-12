"""CPU-only Stage 7 allocation and shared KV isolation checks."""
import torch

from distengine.block_manager import BlockManager
from distengine.paged_kv_cache import KVStorage, PagedKVCache


def raises(error_type, operation):
    try:
        operation()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def snapshot(manager):
    return manager.free_blocks.copy(), {
        key: value.copy() for key, value in manager.request_blocks.items()
    }


def invariant(manager):
    assigned = [block for table in manager.request_blocks.values() for block in table]
    all_blocks = manager.free_blocks + assigned
    assert len(all_blocks) == len(set(all_blocks)) == manager.num_blocks
    assert set(all_blocks) == set(range(manager.num_blocks))


def main():
    for bad in (True, 1.5, 0, -1, '4', None):
        raises(ValueError, lambda: BlockManager(bad, 4))
        raises(ValueError, lambda: BlockManager(4, bad))
    manager = BlockManager(5, 4)
    assert [manager._required_blocks(n) for n in (0, 3, 4, 5)] == [0, 1, 1, 2]
    assert manager.can_allocate(20) and not manager.can_allocate(21)
    for bad in (True, 1.5, -1, '4', None):
        raises(ValueError, lambda: manager.can_allocate(bad))
        raises(ValueError, lambda: manager.allocate('invalid', bad))
        raises(ValueError, lambda: manager.ensure_capacity('A', bad))
    for bad in ('', None, True, 1, []):
        for operation in (lambda: manager.allocate(bad, 1),
                          lambda: manager.ensure_capacity(bad, 1),
                          lambda: manager.get_block_table(bad),
                          lambda: manager.free(bad)):
            raises(ValueError, operation)
    manager.allocate('empty', 0)
    assert manager.get_block_table('empty') == []
    manager.free('empty')
    a = manager.allocate('A', 3)
    b = manager.allocate('B', 5)
    assert set(a).isdisjoint(b)
    a.clear()
    assert len(manager.get_block_table('A')) == 1
    assert len(manager.ensure_capacity('A', 4)) == 1
    assert len(manager.ensure_capacity('A', 5)) == 2
    assert len(manager.ensure_capacity('A', 0)) == 2
    before = snapshot(manager)
    raises(ValueError, lambda: manager.allocate('A', 1))
    raises(RuntimeError, lambda: manager.allocate('too_big', 8))
    raises(RuntimeError, lambda: manager.ensure_capacity('A', 16))
    assert snapshot(manager) == before
    invariant(manager)
    for operation in (lambda: manager.get_block_table('missing'),
                      lambda: manager.ensure_capacity('missing', 1),
                      lambda: manager.free('missing')):
        raises(KeyError, operation)
    print('PASS: validation, boundaries, copies, ownership, growth, atomic exhaustion')

    storage = KVStorage(2, 2, 3, 5, 4, torch.device('cpu'), torch.float32)
    def view(table, shared=storage):
        return PagedKVCache(2, 2, 3, 5, 4, table,
                            torch.device('cpu'), torch.float32, storage=shared)
    ca, cb = view(manager.get_block_table('A')), view(manager.get_block_table('B'))
    assert ca.k_cache is cb.k_cache and ca.v_cache is cb.v_cache
    expected = {name: [torch.empty(1, 2, 0, 3) for _ in range(2)] for name in ('A', 'B')}
    for count in (3, 2):
        for name, cache, offset in (('A', ca, 10), ('B', cb, 100)):
            for layer in range(2):
                k = torch.arange(count * 6, dtype=torch.float32).reshape(1, 2, count, 3) + offset
                expected[name][layer] = torch.cat((expected[name][layer], k), dim=2)
                cache.append(layer, k, k + 1000)
        for name, cache in (('A', ca), ('B', cb)):
            for layer in range(2):
                k, v = cache.read(layer)
                assert torch.equal(k, expected[name][layer])
                assert torch.equal(v, expected[name][layer] + 1000)
    old_table = ca.block_table.copy()
    old_lengths = ca.lengths.copy()
    ca.update_block_table(manager.ensure_capacity('A', 9))
    assert ca.block_table[:len(old_table)] == old_table and ca.lengths == old_lengths
    for invalid in (old_table, list(reversed(ca.block_table)), ca.block_table + [ca.block_table[0]],
                    ca.block_table + [5], ca.block_table + [True]):
        before_table = ca.block_table.copy()
        raises(ValueError, lambda: ca.update_block_table(invalid))
        assert ca.block_table == before_table and ca.lengths == old_lengths
    for layer in range(2):
        k = torch.full((1, 2, 4, 3), 42.)
        ca.append(layer, k, k + 1000)
        expected['A'][layer] = torch.cat((expected['A'][layer], k), dim=2)
        assert torch.equal(ca.read(layer)[0], expected['A'][layer])
        assert torch.equal(cb.read(layer)[0], expected['B'][layer])
    manager.free('A')
    # Retire ca before its physical blocks are reused.
    del ca
    raises(KeyError, lambda: manager.free('A'))
    assert manager.num_free_blocks == 3
    cc = view(manager.allocate('C', 1))
    assert cc.sequence_length == 0 and cc.read(0)[0].shape[2] == 0
    for layer in range(2):
        k = torch.full((1, 2, 1, 3), -7.)
        cc.append(layer, k, k - 1)
        assert torch.equal(cc.read(layer)[0], k)
        assert torch.equal(cc.read(layer)[1], k - 1)
        assert torch.equal(cb.read(layer)[0], expected['B'][layer])
    cc.reset()
    assert cc.sequence_length == 0 and cb.sequence_length == 5
    empty = view([])
    assert empty.capacity == 0 and empty.sequence_length == 0
    raises(ValueError, lambda: empty.append(0, k, k))
    manager.allocate('empty', 0)
    empty.update_block_table(manager.ensure_capacity('empty', 1))
    empty.append(0, k, k)
    assert torch.equal(empty.read(0)[0], k)
    manager.free('empty')
    del empty
    incompatible = KVStorage(1, 2, 3, 5, 4, torch.device('cpu'), torch.float32)
    raises(ValueError, lambda: view([0], incompatible))
    invariant(manager)
    print('PASS: shared storage, interleaved isolation, mapping growth, free/reuse, reset')


if __name__ == '__main__':
    main()
