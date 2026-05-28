use padoc::trace::Trace;
use std::env;
use std::path::Path;

fn load_trace(path: &Path) -> Trace {
    if path.is_dir() {
        Trace::from_dir(path).unwrap()
    } else {
        Trace::from_file(path).unwrap()
    }
}

fn main() {
    let path = env::args().nth(1).unwrap();
    let kernel = env::args().nth(2).unwrap();
    
    eprintln!("Loading: {}", path);
    let trace = load_trace(Path::new(&path));
    
    let mut ts_list: Vec<i64> = Vec::new();
    for (_rank, _pid, _tid, _ph, events) in trace.iter_streams() {
        for ev in events {
            if ev.name.contains(&kernel) {
                ts_list.push(ev.ts);
            }
        }
    }
    ts_list.sort();
    
    eprintln!("{}: {} instances", kernel, ts_list.len());
    for (i, ts) in ts_list.iter().enumerate() {
        println!("{},{}", i, ts);
    }
}
