use padoc::trace::CompressedTrace;
use padoc::arena::{ArenaNode, NodeArena, NodeId};

fn walk_count(arena: &NodeArena, id: NodeId, count: &mut u64) {
    *count += 1;
    if *count > 2_000_000 { panic!("too many: {}", count); }
    match arena.get(id) {
        ArenaNode::Root { children } => {
            for &cid in arena.children(*children) { walk_count(arena, cid, count); }
        }
        ArenaNode::Cpu { children, slots, .. } => {
            let (c, s) = (*children, *slots);
            for &cid in arena.children(c) { walk_count(arena, cid, count); }
            for &cid in arena.children(s) { walk_count(arena, cid, count); }
        }
        ArenaNode::SameCpu { children, slots_start, slots_len, .. } => {
            let (c, ss, sl) = (*children, *slots_start, *slots_len);
            for &cid in arena.children(c) { walk_count(arena, cid, count); }
            for slot in arena.slots_slice(ss, sl) {
                for &cid in arena.children(slot.children) { walk_count(arena, cid, count); }
            }
        }
        _ => {}
    }
}

fn main() {
    let ct = CompressedTrace::read_from_path("/mnt/treasure/ljx/artifacts_v7_sparse/leworldmodel_full.padoc.zst").unwrap();
    let arenas = ct.arenas.as_ref().unwrap();
    for (_rank, pids) in arenas {
        for (_pid, tids) in pids {
            for (tid, phs) in tids {
                for (ph, (arena, root_id)) in phs {
                    if arena.node_count() > 100 {
                        let mut count = 0u64;
                        println!("walking {} nodes (tid={} ph={})...", arena.node_count(), tid, ph);
                        walk_count(arena, *root_id, &mut count);
                        println!("  walked {} OK", count);
                    }
                }
            }
        }
    }
    println!("All done!");
}
