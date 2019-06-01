# HEADER
# Note: Indices of rows and columns commences from 0 and ascend with an increment of 1.
# Remember to write up all error statements for each False return.
# Remember to write up Crammer's method.

from typing import List
from number_tools import *


class Matrix:

    def __init__(self, row: int, col: int, data=None):
        self.row, self.col = row, col
        f = lambda x, y: [[0 for _ in range(y)] for _ in range(x)]  # creates a zero matrix with dimension of row * col
        self.store = f(row, col)
        if data is not None:
            self.initialize(data)

    def __add__(self, arg):
        if type(arg) is not Matrix:
            raise Exception("incorrect matrix addition data input")
        return matrix_add(self, arg)

    def __sub__(self, arg):
        if type(arg) is not Matrix:
            raise Exception("incorrect matrix subtraction data input")
        return matrix_add(self, scalar_multiply(arg, -1))

    def __mul__(self, arg):
        if type(arg) is Matrix:
            return matrix_multiply(self, arg)
        elif is_number(arg):
            return scalar_multiply(self, arg)
        else:
            raise Exception("incorrect matrix multiplication data input")

    def __rmul__(self, arg):
        if is_number(arg):
            return scalar_multiply(self, arg)
        else:
            raise Exception("incorrect matrix multiplication data input")

    def size(self) -> List[int]:
        return [self.row, self.col]

    def initialize(self, data) -> bool:  # assigns elements by filling in each row left to right as moving down the rows
        if data is None:
            return False
        if len(data) != self.row * self.col:
            raise Exception("incorrect matrix initialization data input")
        for r in range(self.row):
            for c in range(self.col):
                self.store[r][c] = data[r * self.col + c]
        return True

    def insert(self, value, row: int, col: int) -> bool:
        # TODO: set the upper bounds for row and col
        if row < 0 or col < 0:
            return False
        self.store[row][col] = value
        return True

    def get_identity(self):
        if self.row != self.col:
            return False
        identity = Matrix(self.row, self.row)
        for i in range(self.row):
            identity.insert(1, i, i)
        return identity

    def get_transpose(self):
        transpose = Matrix(self.col, self.row)
        for c in range(self.col):
            for r in range(self.row):
                transpose.store[c][r] = self.store[r][c]
        return transpose

    def get_inverse(self):
        adjugate = self.get_adjugate()
        determinant = self.get_determinant()
        if determinant == 0:
            return False
        scalar_multiply(adjugate, 1 / determinant)
        return adjugate

    def get_determinant(self, c=0):  # uses first column elements as pivots
        if self.row != self.col:
            return False
        elif self.row == 1:
            return self.store[0][0]
        determinant = 0
        for r in range(self.row):
            determinant += ((-1) ** r) * self.store[r][c] * self.get_submatrix(r, c).get_determinant()
        return determinant

    def get_cofactor_matrix(self):
        cofactor_matrix = Matrix(self.row, self.col)
        for r in range(self.row):
            for c in range(self.col):
                cofactor_matrix.store[r][c] = ((-1) ** (r + c)) * self.get_submatrix(r, c).get_determinant()
        return cofactor_matrix

    def get_cofactor(self, r, c):   # (r: row) & (c: column)
        return self.get_cofactor_matrix().store[r][c]

    def get_adjugate(self):
        return self.get_cofactor_matrix().get_transpose()

    def get_submatrix(self, *args, num_row=1, num_col=1):   # keywords: "num_row" and "num_col"
        if len(args) != num_row + num_col:
            return False
        submatrix = Matrix(self.row - num_row, self.col - num_col)
        row_count = 0
        for r in range(self.row):
            if r in args[:num_row]:
                row_count += 1
                continue
            col_count = 0
            for c in range(self.col):
                if c in args[num_row:]:
                    col_count += 1
                    continue
                submatrix.store[r - row_count][c - col_count] = self.store[r][c]
        return submatrix

    def get_minor(self, *args, num_row=1, num_col=1):
        num_row_temp = num_row
        num_col_temp = num_col
        return self.get_submatrix(*args, num_row=num_row_temp, num_col=num_col_temp).get_determinant()

    def get_trace(self):
        if self.row != self.col:
            return False
        return sum([self.store[i][i] for i in range(self.row)])

    def get_elementary_matrix(self, type: int, i: int, j=0, scalar=0):  # remember to input arguments in this order
        elementary_matrix = Matrix(self.row, self.row).get_identity()
        if type < 1 or type > 3 or i < 0 or i >= elementary_matrix.row:
            return False
        elif type == 1:  # interchange ith row and jth row (requires i and j)
            if j < 0 or j > elementary_matrix.row:
                return False
            elementary_matrix.store[i][i] = elementary_matrix.store[j][j] = 0
            elementary_matrix.store[i][j] = elementary_matrix.store[j][i] = 1
        elif type == 2:  # multiply ith row by a factor of the scalar (requires i and scalar)
            if scalar == 0:  # this prevents error from not inputting j=0
                if is_number(j) and j != 0:
                    scalar = j
                else:
                    return False
            elementary_matrix.store[i][i] = scalar
        elif type == 3:  # add the scalar multiple of jth row to ith row (requires i, j, and scalar)
            if scalar == 0 or j < 0 or j > elementary_matrix.row:
                return False
            elementary_matrix.store[i][j] = scalar
        return elementary_matrix

    def get_store_data(self):  # returns the matrix as a 1D array
        return [self.store[i][j] for i in range(self.row) for j in range(self.col)]

    def get_row_data(self, row):  # the row count begins at 0
        if not 0 <= row <= self.row-1:
            return False
        return self.store[row]

    def get_col_data(self, col):  # the col count begins at 0
        if not 0 <= col <= self.col-1:
            return False
        column = [self.store[r][col] for r in range(self.row)]
        return column

    def get_block_data(self, row_1, col_1, row_2, col_2, linearize=False):
        # (row_1, col_1): top-left coordinate of the block
        # (row_2, col_2): bottom-right coordinate of the block
        # both coordinates are inclusive
        if not 0 <= row_1 <= self.row-1:
            return False
        if not 0 <= row_2 <= self.row-1:
            return False
        if not row_1 <= row_2:
            return False
        if not 0 <= col_1 <= self.col-1:
            return False
        if not 0 <= col_2 <= self.col-1:
            return False
        if not col_1 <= col_2:
            return False
        block_data = []
        for i in range(row_1, row_2+1):
            temp_data = self.get_row_data(i)
            if linearize:
                block_data.extend(temp_data[col_1:col_2+1])
            else:
                block_data.append(temp_data[col_1:col_2+1])
        return block_data

    def set_row_data(self, row, row_data):
        if len(row_data) != self.col:
            return False
        if not 0 <= row <= self.row-1:
            return False
        for j in range(self.col):
            self.insert(row_data[j], row, j)
        return True

    def set_col_data(self, col, col_data):
        if len(col_data) != self.row:
            return False
        if not 0 <= col <= self.col-1:
            return False
        for i in range(self.row):
            self.insert(col_data[i], i, col)
        return True

    def set_block_data(self, row_1, col_1, row_2, col_2, block_data, linearize=False):
        # (row_1, col_1): top-left coordinate of the block
        # (row_2, col_2): bottom-right coordinate of the block
        # both coordinates are inclusive
        if not 0 <= row_1 <= self.row - 1:
            return False
        if not 0 <= row_2 <= self.row - 1:
            return False
        if not row_1 <= row_2:
            return False
        if not 0 <= col_1 <= self.col - 1:
            return False
        if not 0 <= col_2 <= self.col - 1:
            return False
        if not col_1 <= col_2:
            return False

        block_data_row = row_2 + 1 - row_1
        block_data_col = col_2 + 1 - col_1
        if linearize:
            if len(block_data) != block_data_row * block_data_col:
                return False
        else:
            if len(block_data) != block_data_row or len(block_data[0]) != block_data_col:
                return False

        for i in range(row_1, row_2+1):
            for j in range(col_1, col_2+1):
                if linearize:
                    self.insert(block_data[block_data_col * (i - row_1) + (j - col_1)], i, j)
                else:
                    self.insert(block_data[i - row_1][j - col_1], i, j)

        return True

    def normalize(self, by_row=True, sum_to=1.0):
        # TODO: implement by_col and by_whole
        for r in range(self.row):
            row_data = self.get_row_data(r)
            row_sum = sum(row_data)
            if row_sum == 0:
                continue
            normalized_row_data = [sum_to * float(elem) / float(row_sum) for elem in row_data]
            self.set_row_data(r, normalized_row_data)
        return True

    def print(self, precision=False):  # requires editing to accommodate for: how to space each column
        for r in range(self.row):
            if precision is False:
                print(self.store[r])
            else:
                print(["{:.{}f}".format(elem, precision) for elem in self.store[r]])
                # print([round(elem, format_decimal) for elem in self.store[r]])


def matrix_is_same_dimension(*args):
    def matrix_is_same_dimension_basic(A: Matrix, B: Matrix):
        if A.row != B.row or A.col != B.col:
            return False
        return True
    if len(args) < 2:
        return False
    for i in range(len(args) - 1):
        if not matrix_is_same_dimension_basic(args[i], args[i + 1]):
            return False
    return True


def matrix_is_equal(*args):
    def matrix_is_equal_basic(A: Matrix, B: Matrix):
        if not matrix_is_same_dimension(A, B):
            return False
        if A.get_store_data() != B.get_store_data():
            return False
        return True
    if len(args) < 2:
        return False
    for i in range(len(args) - 1):
        if not matrix_is_equal_basic(args[i], args[i + 1]):
            return False
    return True


def matrix_add(*args):
    def matrix_add_basic(A: Matrix, B: Matrix):
        if not matrix_is_same_dimension(A, B):
            return False
        Sum_temp = Matrix(A.row, A.col)
        Sum_temp.store = [list(map(lambda x, y: x + y, A.store[r], B.store[r])) for r in range(A.row)]
        return Sum_temp
    if len(args) < 2:
        return False
    Sum = args[0]
    for M in args[1:]:
        if Sum is False:
            return False
        Sum = matrix_add_basic(Sum, M)
    return Sum


def matrix_multiply(*args):
    def matrix_multiply_basic(A: Matrix, B:Matrix):
        if A.col != B.row:
            return False
        Product_temp = Matrix(A.row, B.col)
        for i in range(A.row):
            for k in range(B.col):
                for j in range(A.col):
                    Product_temp.store[i][k] += A.store[i][j] * B.store[j][k]
        return Product_temp
    if len(args) < 2:
        return False
    Product = args[0]
    for M in args[1:]:
        if Product is False:
            return False
        Product = matrix_multiply_basic(Product, M)
    return Product


def scalar_multiply(A: Matrix, scalar):
    A.store = [[scalar * x for x in row] for row in A.store]
    return A


def matrix_augment(*args, horizontally=True):
    def matrix_augment_basic(A: Matrix, B: Matrix, horizontally=True):
        if not horizontally:
            if A.col != B.col:
                return False
            data = A.get_store_data() + B.get_store_data()
            Augmented_temp = Matrix(A.row + B.row, A.col, data)
            return Augmented_temp
        if A.row != B.row:
            return False
        data = []
        for i in range(A.row):
            for j in range(A.col + B.col):
                if j < A.col:
                    data += [A.store[i][j]]
                else:
                    data += [B.store[i][j - A.col]]
        Augmented_temp = Matrix(A.row, A.col + B.col, data)
        return Augmented_temp
    if len(args) < 2:
        return False
    Augmented_matrix = args[0]
    for M in args[1:]:
        if Augmented_matrix is False:
            return False
        Augmented_matrix = matrix_augment_basic(Augmented_matrix, M, horizontally)
    return Augmented_matrix


def matrix_solve_equation(A: Matrix, b: Matrix, use_ge=True, printed=False):
    if not use_ge:
        A_inverse = A.get_inverse()
        if A_inverse is False:
            return False
        return matrix_multiply(A_inverse, b)
    gje_data = matrix_do_back_substitution(A, b)
    if gje_data[1] is None:
        return False
    A_reduced, b_reduced, rank = gje_data[0], gje_data[1], gje_data[3]
    if rank < A_reduced.row:
        for r in range(rank, A_reduced.row):
            if b_reduced.store[r][0] != 0:
                if printed:
                    print("No Solution")
                return False
    if rank < A_reduced.col:
        if printed:
            print("Infinitely Many Solution")
        return False
    if printed:
        print("Unique Solution")
    return gje_data[1]


def matrix_do_gaussian_elimination(A: Matrix, b=None):
    REF_A = Matrix(A.row, A.col)
    REF_A.store = A.store
    elem_ops = []
    zero_col = 0
    for j in range(REF_A.col):
        pivot = j - zero_col
        while REF_A.store[pivot][j] == 0:
            pivot += 1
            if pivot >= REF_A.row:
                break
        if pivot >= REF_A.row:
            zero_col += 1
            continue
        if pivot != j - zero_col:
            elem_matrix_temp = REF_A.get_elementary_matrix(1, i=j - zero_col, j=pivot)
            elem_ops.append([1, j - zero_col, pivot, 0])
            REF_A = matrix_multiply(elem_matrix_temp, REF_A)
        if REF_A.store[j - zero_col][j] != 1:
            elem_matrix_temp = REF_A.get_elementary_matrix(2, i=j - zero_col, scalar=1 / REF_A.store[j - zero_col][j])
            elem_ops.append([2, j - zero_col, 0, 1 / REF_A.store[j - zero_col][j]])
            REF_A = matrix_multiply(elem_matrix_temp, REF_A)
        for i in range(j - zero_col + 1, REF_A.row):
            if REF_A.store[i][j] != 0:
                elem_matrix_temp = REF_A.get_elementary_matrix(
                    3, i=i, j=j - zero_col, scalar=(-1) * REF_A.store[i][j] / REF_A.store[j - zero_col][j])
                elem_ops.append([3, i, j - zero_col, (-1) * REF_A.store[i][j] / REF_A.store[j - zero_col][j]])
                REF_A = matrix_multiply(elem_matrix_temp, REF_A)
    REF_b = b
    if b is not None and A.row == b.row and b.col == 1:
        for elem_op_data in elem_ops:
            elem_op = expand_elem_op_data(REF_A, elem_op_data)
            REF_b = matrix_multiply(elem_op, REF_b)
    else:
        REF_b = None
    return [REF_A, REF_b, elem_ops, A.col - zero_col]  # if b was not inputted or was a wrong input, REF_b outputs None


def expand_elem_op_data(A: Matrix, elem_op_data: List[int]):
    # use this method to convert elem_ops data into an actual elementary matrix
    return A.get_elementary_matrix(
        type=elem_op_data[0], i=elem_op_data[1], j=elem_op_data[2], scalar=elem_op_data[3])


def matrix_get_rank(A: Matrix):  # rank = A.col (# of variables) - zero_col (nullity) from REF or RREF
    ge_data = matrix_do_gaussian_elimination(A)
    return ge_data[3]


def matrix_get_nullity(A: Matrix):
    ge_data = matrix_do_gaussian_elimination(A)
    return A.col - ge_data[3]


def matrix_get_REF(A: Matrix, b=None):
    ge_data = matrix_do_gaussian_elimination(A, b)
    if b is None:
        return ge_data[0]
    return [ge_data[0], ge_data[1]]


def matrix_do_back_substitution(A: Matrix, b=None):
    ge_data = matrix_do_gaussian_elimination(A, b)
    elem_ops = []
    RREF_A, RREF_b = ge_data[0:2]
    rank = ge_data[3]
    for i in range(rank - 1, -1, -1):
        for j in range(RREF_A.col):
            if int(round(RREF_A.store[i][j])) == 1:  # int and round to avoid any type of float value bugs
                for k in range(i):
                    if RREF_A.store[k][j] != 0:
                        elem_matrix_temp = RREF_A.get_elementary_matrix(3, k, i, (-1) * RREF_A.store[k][j])
                        elem_ops.append([3, k, i, (-1) * RREF_A.store[k][j]])
                        RREF_A = matrix_multiply(elem_matrix_temp, RREF_A)
                break
    if RREF_b is not None:
        for elem_op_data in elem_ops:
            elem_op = expand_elem_op_data(RREF_A, elem_op_data)
            RREF_b = matrix_multiply(elem_op, RREF_b)
    return [RREF_A, RREF_b, elem_ops, rank]  # may remove rank as the output


def matrix_get_RREF(A: Matrix, b=None):
    gje_data = matrix_do_back_substitution(A, b)  # gaussian elimination + back substitution = gauss-jordan elimination
    if b is None:
        return gje_data[0]
    return [gje_data[0], gje_data[1]]


if __name__ == '__main__':
    # data_A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # data_b = [1, 1, 1, 0, 2, 5, 2, 5, -1]
    # # data_C = [1, 1, 1, 0, 2, 5, 2, 5, 0]
    # A = Matrix(3, 3, data_A)
    # # b = Matrix(3, 3, data_A)
    # # C = Matrix(3, 3, data_C)
    # # D = A
    # # print(matrix_is_equal(A,b,D))
    # # print(data_A == data_b)
    # print(A.get_minor(1,1,num_row=1,num_col=1))
    # data_A = [1, 1, 0, 9, 0, -4, 0, -4, 1]
    # data_b = [-1, 9, 8]
    # data_D = [1, 2, 3, 4, 5, 6]
    # A = Matrix(3, 3, data_A)
    # b = Matrix(1, 3, data_b)
    # D = Matrix(2, 3, data_D)
    # # matrix_solve_equation(A, b)
    # # hermitian = Matrix(2, 2, [1, 1, 1, -1])
    # # hermitian.get_inverse().print()
    # C = matrix_augment(A, b, D, horizontally=False)
    # C.print()
    A = Matrix(4, 5, list(range(1, 21)))
    print(A.get_block_data(0, 1, 2, 2, linearize=False))
    A.print()
    A.set_block_data(0, 1, 2, 2, [-1, -2, -3, -4, -5, -6], linearize=True)
    A.print()
