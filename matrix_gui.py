import tkinter as tk
from matrix import *
from matrix_tools import *


class SolveLinearSystemWindow(tk.Frame):

    def __init__(self, master, *args, **kwargs):
        self.master = master
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.configure_gui()
        self.create_widgets()

    def configure_gui(self):
        self.master.title("Solve Linear System")
        self.master.geometry("500x500")

    def create_widgets(self):
        self.entry_list = []
        self.var_label_list = []

        self.first_first_entry = tk.Entry(self, width=5)
        self.first_first_label = tk.Label(self, text="x1 + ").grid(row=0, column=1)
        self.first_second_entry = tk.Entry(self, width=5)
        self.first_second_label = tk.Label(self, text="x2 = ").grid(row=0, column=3)
        self.first_third_entry = tk.Entry(self, width=5)
        self.second_first_entry = tk.Entry(self, width=5)
        self.second_first_label = tk.Label(self, text="x1 + ").grid(row=1, column=1)
        self.second_second_entry = tk.Entry(self, width=5)
        self.second_second_label = tk.Label(self, text="x2 = ").grid(row=1, column=3)
        self.second_third_entry = tk.Entry(self, width=5)
        self.button = tk.Button(self, text="Calculate", command=self.solve_linear_system)
        self.string_1 = tk.StringVar()
        self.string_2 = tk.StringVar()
        self.label_1 = tk.Label(self, textvariable=self.string_1)
        self.label_2 = tk.Label(self, textvariable=self.string_2)

        self.first_first_entry.grid(row=0, column=0)
        self.first_second_entry.grid(row=0, column=2)
        self.first_third_entry.grid(row=0, column=4)
        self.second_first_entry.grid(row=1, column=0)
        self.second_second_entry.grid(row=1, column=2)
        self.second_third_entry.grid(row=1, column=4)
        self.button.grid(row=2, column=0, columnspan=3)
        self.label_1.grid(row=3)
        self.label_2.grid(row=4)

    def solve_linear_system(self):
        data_A = []
        data_A.append(int(self.first_first_entry.get()))
        data_A.append(int(self.first_second_entry.get()))
        data_A.append(int(self.second_first_entry.get()))
        data_A.append(int(self.second_second_entry.get()))
        data_b = []
        data_b.append(int(self.first_third_entry.get()))
        data_b.append(int(self.second_third_entry.get()))
        A = Matrix(2, 2, data_A)
        b = Matrix(2, 1, data_b)
        solution = matrix_solve_equation(A, b)
        self.string_1.set("x1 = " + str(solution.store[0][0]))
        self.string_2.set("x2 = " + str(solution.store[1][0]))


root = tk.Tk()
window = SolveLinearSystemWindow(root).grid()

root.mainloop()
