perf record -F 520 --call-graph dwarf "$@"
perf script > flamegraphs/out.perf
/private/home/mruberry/git/FlameGraph/stackcollapse-perf.pl flamegraphs/out.perf > flamegraphs/out.folded
/private/home/mruberry/git/FlameGraph/flamegraph.pl flamegraphs/out.folded > flamegraphs/flame_graph.svg