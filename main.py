from MainClasses.part1 import Part1
from MainClasses.part2 import Part2Q1, Part2Q2

'''
d = Part1('Data/ndt_tests_corrigido.csv', 'Output/Part1')
d.write_latex_tables()
d.write_amount_of_points()
d.generate_plots_for_selection()
d.write_mle_models_to_file()
d.plot_mle_models()
d.run_bayesian_inference_100()
d.run_bayesian_inference_70_30()
'''

'''
d = Part2Q1('Data/ndt_tests_corrigido.csv', 'Output/Part2/Q1/throughput_down.txt', 'throughput_down')
d.write()

d = Part2Q1('Data/ndt_tests_corrigido.csv', 'Output/Part2/Q1/throughput_up.txt', 'throughput_up')
d.write()
'''

d = Part2Q2('Data/ndt_tests_corrigido.csv', 'Output/Part2/Q2/rtt_down.txt', 'rtt_down')
d.write()

d = Part2Q2('Data/ndt_tests_corrigido.csv', 'Output/Part2/Q2/rtt_up.txt', 'rtt_up')
d.write()