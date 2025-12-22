from MainClasses.part1 import Part1

d = Part1('Data/ndt_tests_corrigido.csv', 'Output/Part1')
d.write_latex_tables()
d.write_amount_of_points()
d.generate_plots_for_selection()
d.write_mle_models_to_file()
d.plot_mle_models()
d.run_bayesian_inference_100()
d.run_bayesian_inference_70_30()