import pstats
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(20)
p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('cumulative').print_stats(20)
p.print_callees('particle.py')
