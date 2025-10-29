from classes import *

d = AllData('Data/ndt_tests_corrigido.csv')
#d.write_latex_tables()
#d.write_amount_of_points()
d.generate_plots_for_selection()