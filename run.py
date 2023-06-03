import tkinter as tk
import numpy as np
from tkinter import filedialog
import Euler_solver

find_diff = True
get_weno_plot = True


def choose_file(file_choose, for_weno_plot_path=None):
    if file_choose is False:
        file_path = 'cfg.txt'
    elif for_weno_plot_path is not None:
        file_path = for_weno_plot_path
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()

    return file_path


def get_max_diff(for_weno_plot_path1=None, for_weno_plot_path2=None):
    x1, nx1, r1, u1, p1 = Euler_solver.solve(filename=choose_file(True, for_weno_plot_path1))
    x2, nx2, r2, u2, p2 = Euler_solver.solve(filename=choose_file(True, for_weno_plot_path2))
    assert nx1 == nx2
    assert np.array_equal(x1, x2)
    max_diff = np.max([np.max(np.abs(r1 - r2)), np.max(np.abs(u1 - u2)), np.max(np.abs(p1 - p2))])

    return max_diff


if find_diff is False:
    Euler_solver.solve(filename=choose_file(False))
else:
    if get_weno_plot is False:
        print(get_max_diff(get_weno_plot))
    else:
        path_HLLE_mask = 'cfg_HLLE_weno'
        path_HLLC_mask = 'cfg_HLLC_weno'
        max_diff_weno3 = get_max_diff(path_HLLE_mask+'3.txt', path_HLLC_mask+'3.txt')
        max_diff_weno5 = get_max_diff(path_HLLE_mask+'5.txt', path_HLLC_mask+'5.txt')
        max_diff_weno7 = get_max_diff(path_HLLE_mask+'7.txt', path_HLLC_mask+'7.txt')
        print(max_diff_weno3, max_diff_weno5, max_diff_weno7)


